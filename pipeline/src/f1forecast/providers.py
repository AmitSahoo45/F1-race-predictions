"""Bounded adapters for public FastF1 and Jolpica data."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import fastf1
import httpx
import pandas as pd
from fastf1 import _api as fastf1_api
from fastf1.core import Session, Telemetry
from fastf1.exceptions import RateLimitExceededError


class SourceValidationError(ValueError):
    """Expected malformed source observations, eligible for an alternate provider."""


@dataclass
class RawSession:
    season: int
    event_id: str
    circuit_id: str
    session_type: str
    session_start: pd.Timestamp
    session_end: pd.Timestamp
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    telemetry: dict[str, pd.DataFrame] = field(default_factory=dict)
    source: str = "FastF1"
    driver_ids: dict[str, str] = field(default_factory=dict)
    completed: bool = False
    source_files: list[dict[str, str | int]] = field(default_factory=list)
    completion_evidence: str | None = None
    source_labels: dict[str, str] = field(default_factory=dict)
    extraction_version: str | None = None
    telemetry_omission_reason: str | None = None


def session_is_complete(status: pd.DataFrame) -> bool:
    """Only FastF1's Finalised event proves that results are no longer live."""
    values = status.get("Status")
    return isinstance(values, pd.Series) and bool(values.astype(str).eq("Finalised").any())


def _fastest_lap_trace(lap: Any) -> pd.DataFrame:
    """Reproduce FastF1's merged lap channels without unused driver-ahead work."""
    car = cast(Telemetry, lap.get_car_data(pad=1, pad_side="both"))
    if car.empty:
        return pd.DataFrame(car)
    car = car.add_distance().add_relative_distance()
    try:
        positions = cast(Telemetry, lap.get_pos_data(pad=1, pad_side="both"))
    except (KeyError, ValueError, AttributeError):
        return pd.DataFrame(car.slice_by_lap(lap, interpolate_edges=True))
    if positions.empty:
        return pd.DataFrame(car.slice_by_lap(lap, interpolate_edges=True))
    merged = positions.merge_channels(car).slice_by_lap(lap, interpolate_edges=True)
    return pd.DataFrame(merged)


class JolpicaProvider:
    """Small, cached Jolpica client with a finite per-process request budget."""

    BASE_URL = "https://api.jolpi.ca/ergast/f1"

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        client: httpx.Client | None = None,
        min_interval_s: float = 0.3,
        max_requests: int = 450,
        cache_ttl_s: float = 3600,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            timeout=20, headers={"User-Agent": "F1Forecast/0.1 (public-data research)"}
        )
        self.min_interval_s = min_interval_s
        self.max_requests = max_requests
        self.cache_ttl_s = cache_ttl_s
        self.sleep = sleep
        self._requests = 0
        self._last_request = 0.0

    def _page(self, path: str, offset: int) -> dict:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", path)
        cached = self.cache_dir / f"{safe}_{offset}.json"
        if cached.exists() and time.time() - cached.stat().st_mtime < self.cache_ttl_s:
            return json.loads(cached.read_text(encoding="utf-8"))
        for attempt in range(3):
            if self._requests >= self.max_requests:
                raise RuntimeError("Jolpica request budget exhausted")
            elapsed = time.monotonic() - self._last_request
            if self._last_request and elapsed < self.min_interval_s:
                self.sleep(self.min_interval_s - elapsed)
            self._requests += 1
            self._last_request = time.monotonic()
            response = self.client.get(
                f"{self.BASE_URL}/{path}.json",
                params={"limit": 100, "offset": offset},
                headers={"User-Agent": "F1Forecast/0.1 (public-data research)"},
            )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 2:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 60.0) if retry_after else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                self.sleep(max(delay, 0.0))
                continue
            response.raise_for_status()
            payload = response.json()
            temp = cached.with_suffix(".tmp")
            temp.write_text(json.dumps(payload), encoding="utf-8")
            temp.replace(cached)
            return payload
        raise RuntimeError("unreachable retry state")

    def fetch_races(self, year: int) -> list[dict]:
        return self._fetch_collection(str(year), "RaceTable", "Races")

    def fetch_drivers(self, year: int) -> list[dict]:
        return self._fetch_collection(f"{year}/drivers", "DriverTable", "Drivers")

    def fetch_results(self, year: int, round_number: int, target: Literal["Q", "R"]) -> list[dict]:
        """Fetch a race's qualifying or race classifications without merging sources."""
        if target not in {"Q", "R"}:
            raise ValueError("Jolpica result target must be Q or R")
        route, key = (
            ("qualifying", "QualifyingResults") if target == "Q" else ("results", "Results")
        )
        path = f"{year}/{round_number}/{route}"
        rows: list[dict] = []
        offset = 0
        while True:
            root = self._page(path, offset)["MRData"]
            for race in root.get("RaceTable", {}).get("Races", []):
                rows.extend(race.get(key, []))
            offset += int(root.get("limit", 100))
            if offset >= int(root["total"]):
                return rows

    def _fetch_collection(self, path: str, table: str, collection: str) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            root = self._page(path, offset)["MRData"]
            batch = root.get(table, {}).get(collection, [])
            rows.extend(batch)
            total = int(root["total"])
            offset += int(root.get("limit", 100))
            if offset >= total or not batch:
                return rows


class FastF1Provider:
    """Loads a single real session and one representative lap trace per driver."""

    _MIN_SESSION_INTERVAL_S = 90
    _RATE_WINDOW_BACKOFF_S = 3601

    def __init__(
        self, cache_dir: Path | str, *, sleep: Callable[[float], None] | None = None
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep if sleep is not None else time.sleep
        self._schedules: dict[int, Any] = {}

    def _pace_fetch(self) -> None:
        """Persist a conservative start interval across resumable workers."""
        budget = self.cache_dir / ".fastf1-pacing.json"
        if budget.exists():
            previous = datetime.fromisoformat(
                json.loads(budget.read_text(encoding="utf-8"))["last_fetch_start_utc"]
            )
            elapsed = (datetime.now(UTC) - previous).total_seconds()
            remaining = self._MIN_SESSION_INTERVAL_S - elapsed
            if remaining > 0:
                logging.getLogger(__name__).info("FastF1 source pacing %.1f s", remaining)
                self._sleep(remaining)
        staged = budget.with_suffix(".staging")
        staged.write_text(
            json.dumps({"last_fetch_start_utc": datetime.now(UTC).isoformat()}),
            encoding="utf-8",
        )
        staged.replace(budget)

    def _load_session(self, year: int, event: str | int, session_type: str) -> Session:
        for attempt in range(2):
            try:
                session = fastf1.get_session(year, event, session_type)
                session.load(laps=True, telemetry=True, weather=True)
                return session
            except RateLimitExceededError:
                if attempt:
                    raise
                logging.getLogger(__name__).warning(
                    "FastF1 hourly request budget reached; retrying after %d s",
                    self._RATE_WINDOW_BACKOFF_S,
                )
                self._sleep(self._RATE_WINDOW_BACKOFF_S)
        raise RuntimeError("FastF1 rate-limit retry exhausted")

    def fetch_driver_list(
        self, year: int, event: str | int, session_type: str
    ) -> tuple[pd.DataFrame, dict[str, str | int]]:
        """Read only this practice session's F1 DriverList, including zero-lap entrants."""
        if session_type not in {"FP1", "FP2", "FP3"}:
            raise ValueError("driver-list roster source must be a practice session")
        fastf1.Cache.enable_cache(str(self.cache_dir))
        for attempt in range(2):
            try:
                if isinstance(event, int):
                    if year not in self._schedules:
                        self._schedules[year] = fastf1.get_event_schedule(year, backend="fastf1")
                    session = self._schedules[year].get_event_by_round(event).get_session(session_type)
                else:
                    session = fastf1.get_session(year, event, session_type)
                source = fastf1_api.driver_info(session.api_path)
                break
            except RateLimitExceededError:
                if attempt:
                    raise
                logging.getLogger(__name__).warning(
                    "FastF1 DriverList hourly budget reached; retrying after %d s",
                    self._RATE_WINDOW_BACKOFF_S,
                )
                self._sleep(self._RATE_WINDOW_BACKOFF_S)
        else:
            raise RuntimeError("FastF1 DriverList rate-limit retry exhausted")
        if not source:
            raise SourceValidationError("same-session FastF1 DriverList is empty")
        rows = []
        for number, entry in source.items():
            first, last = entry.get("FirstName"), entry.get("LastName")
            rows.append(
                {
                    "Abbreviation": entry.get("Tla"),
                    "DriverNumber": str(entry.get("RacingNumber") or number),
                    "TeamName": entry.get("TeamName"),
                    "TeamColor": entry.get("TeamColour"),
                    "FullName": f"{first} {last}" if first and last else entry.get("FullName"),
                }
            )
        frame = pd.DataFrame(rows)
        abbreviations = cast(pd.Series, frame["Abbreviation"])
        numbers = cast(pd.Series, frame["DriverNumber"])
        if (
            bool(abbreviations.isna().any())
            or bool(numbers.duplicated().any())
            or bool(abbreviations.duplicated().any())
        ):
            raise SourceValidationError("same-session FastF1 DriverList has ambiguous identities")
        canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        return frame, {
            "url": f"{fastf1_api.base_url}{session.api_path}{fastf1_api.pages['driver_list']}",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "bytes": len(canonical),
            "hash_kind": "canonical_driver_info_json",
            "source": "FastF1 same-session DriverList (cached or live; mirror fallback possible)",
            "accessed_at": datetime.now(UTC).isoformat(),
        }

    def fetch_schedule(
        self, year: int, event: int, session_type: str
    ) -> tuple[pd.Timestamp, dict[str, str | int]]:
        """Read the provider's explicit UTC date, never infer dates from lap timestamps."""
        fastf1.Cache.enable_cache(str(self.cache_dir))
        if year not in self._schedules:
            self._schedules[year] = fastf1.get_event_schedule(year, backend="fastf1")
        metadata = self._schedules[year].get_event_by_round(event)
        value = pd.Timestamp(metadata.get_session_date(session_type, utc=True))
        if not isinstance(value, pd.Timestamp):
            raise SourceValidationError("FastF1 UTC event schedule has no session date")
        start = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        canonical = json.dumps(
            metadata.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode("utf-8")
        return start, {
            "url": f"https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_{year}.json",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "bytes": len(canonical),
            "digest_basis": "canonical_event_schedule_row",
            "source": "FastF1 UTC event schedule",
            "accessed_at": datetime.now(UTC).isoformat(),
        }

    def fetch_session(self, year: int, event: str | int, session_type: str) -> RawSession:
        fastf1.Cache.enable_cache(str(self.cache_dir))
        self._pace_fetch()
        session = self._load_session(year, event, session_type)
        event_meta = session.event
        round_number = int(event_meta["RoundNumber"])
        start_value = pd.Timestamp(session.date)
        if not isinstance(start_value, pd.Timestamp):
            raise TypeError("FastF1 session has no start date")
        start = (
            start_value.tz_localize("UTC")
            if start_value.tzinfo is None
            else start_value.tz_convert("UTC")
        )
        laps = pd.DataFrame(session.laps).copy()
        end = start
        lap_start_dates = laps.get("LapStartDate")
        lap_times = laps.get("LapTime")
        if (
            isinstance(lap_start_dates, pd.Series)
            and isinstance(lap_times, pd.Series)
            and bool(lap_start_dates.notna().any())
        ):
            last = pd.Timestamp(pd.to_datetime(lap_start_dates, utc=True).max())
            durations = pd.to_timedelta(lap_times, errors="coerce")
            longest = cast(pd.Series, durations).max()
            if isinstance(last, pd.Timestamp):
                end = (
                    max(start, last + longest)
                    if isinstance(longest, pd.Timedelta)
                    else max(start, last)
                )
        telemetry: dict[str, pd.DataFrame] = {}
        driver_column = laps.get("Driver")
        for driver in (
            sorted(driver_column.dropna().unique()) if isinstance(driver_column, pd.Series) else []
        ):
            try:
                fastest = session.laps.pick_drivers(driver).pick_accurate().pick_fastest()
                if fastest is not None:
                    data = _fastest_lap_trace(fastest)
                    if data is not None and not data.empty:
                        telemetry[str(driver)] = pd.DataFrame(data).copy()
            except (KeyError, ValueError, AttributeError):
                continue
        completed = session_is_complete(pd.DataFrame(session.session_status))
        return RawSession(
            season=year,
            event_id=f"{year}-{round_number:02d}",
            circuit_id=str(event_meta.get("Location", "unknown")).lower().replace(" ", "_"),
            session_type=session_type.upper(),
            session_start=start,
            session_end=end,
            laps=laps,
            results=pd.DataFrame(session.results).copy(),
            weather=pd.DataFrame(session.weather_data).copy(),
            telemetry=telemetry,
            completed=completed,
            completion_evidence="FastF1 session_status Finalised" if completed else None,
            source_labels={
                "telemetry": "FastF1 car_data + pos_data merged; unused driver-ahead omitted"
            },
            extraction_version="fastf1-merged-v1",
        )
