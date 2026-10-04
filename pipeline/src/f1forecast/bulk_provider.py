"""Pinned public JSON adapter for retrospective, completed F1 weekends only.

This provider reads data files, never third-party Python or pickle. Its output
uses the same RawSession normalization and immutable archive as direct FastF1.
"""

# Pandas' stubs widen scalar/Series operations into incompatible unions here;
# the adapter validates JSON schemas and exercises normalization in offline tests.
# pyright: reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from f1forecast.calendar import event_from_race
from f1forecast.providers import JolpicaProvider, RawSession, SourceValidationError

# Reviewed public revisions. Changing one requires revalidation of the overlap
# comparisons and creates a different set of immutable source-file identities.
COMMITS = {
    2022: "5d53d094e5004aab25537089ccaeadc529cbad3a",
    2023: "fcb91b207f8c0415f85b057f77d78600c5bba57f",
    2024: "c47beccd9fd3cd79c247370af9029a6f27f6dfc6",
    2025: "e6b8b8bbaa173ab1c5fb7740d65e2c56eb65a51a",
    2026: "47909a8691e6af8131594f8ebf34cd0737a1e614",
}
SESSION_FOLDERS = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "R": "Race",
}
_JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


class MissingBulkSourceError(ValueError):
    """A pinned source file is absent; a direct FastF1 read may replace it."""


class PracticeRosterUnavailableError(ValueError):
    """No verified same-session pre-target roster; full direct read is needed."""


class PracticeRosterProvider(Protocol):
    def fetch_driver_list(
        self, year: int, event: str | int, session_type: str
    ) -> tuple[pd.DataFrame, dict[str, str | int]]: ...

    def fetch_schedule(
        self, year: int, event: int, session_type: str
    ) -> tuple[pd.Timestamp, dict[str, str | int]]: ...


def _canonical_source_record(url: str, payload: Any) -> dict[str, str | int]:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return {
        "url": url,
        "sha256": sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "digest_basis": "canonical_json_rows",
    }


def _seconds(values: pd.Series) -> pd.Series:
    clean = values.map(
        lambda value: np.nan if isinstance(value, str) and value == "None" else value
    )
    return cast(pd.Series, pd.to_numeric(clean, errors="coerce"))


def _timedelta(values: pd.Series) -> pd.Series:
    return pd.to_timedelta(_seconds(values), unit="s", errors="coerce")


def _q_time(value: Any) -> pd.Timedelta | None:
    if not value or value == "None":
        return None
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid Jolpica qualifying time: {value!r}")
    minutes, seconds = int(parts[0]), float(parts[1])
    if minutes < 0 or not 0 <= seconds < 60:
        raise ValueError(f"invalid Jolpica qualifying time: {value!r}")
    return pd.Timedelta(round((minutes * 60 + seconds) * 1_000_000), unit="us")


class _Limiter:
    """One shared 250 ms request-start interval across all worker threads."""

    def __init__(self, monotonic: Callable[[], float], sleep: Callable[[float], None]):
        self.monotonic = monotonic
        self.sleep = sleep
        self.next_at = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            wait = max(0.0, self.next_at - self.monotonic())
            if wait:
                self.sleep(wait)
            self.next_at = max(self.monotonic(), self.next_at) + 0.25


class TracingInsightsProvider:
    """Historical-only pinned JSON reader with Jolpica schedule and labels.

    `completed` means a published pinned JSON session from a weekend whose
    scheduled GP ended at least 24 hours earlier, with published classification
    rows for Q/R. It does not claim observation of FastF1's Finalised event.
    """

    def __init__(
        self,
        jolpica_provider: JolpicaProvider,
        *,
        practice_roster: PracticeRosterProvider,
        client: httpx.Client | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        event_name_overrides: dict[tuple[int, int], str] | None = None,
    ) -> None:
        self.jolpica = jolpica_provider
        self.practice_roster = practice_roster
        self.client = client or httpx.Client(
            timeout=30, headers={"User-Agent": "F1Forecast/0.1 (public-data research)"}
        )
        self.now = now or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.limiter = _Limiter(monotonic, sleep)
        self.event_name_overrides = event_name_overrides or {}

    def _read_json(self, url: str, commit: str) -> tuple[Any, dict[str, str | int]]:
        if not url.startswith("https://raw.githubusercontent.com/TracingInsights/"):
            raise ValueError("bulk source URL is outside the pinned public archive")
        for attempt in range(3):
            self.limiter.acquire()
            try:
                response = self.client.get(url)
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise RuntimeError(f"bulk source transport failed: {url}") from exc
                self.sleep(2**attempt)
                continue
            if response.status_code == 429:
                if attempt == 2:
                    raise RuntimeError(f"bulk source rate-limited: {url}")
                try:
                    retry_after = float(response.headers.get("Retry-After", "1"))
                except ValueError:
                    retry_after = 1.0
                self.sleep(max(0.25, retry_after))
                continue
            if response.status_code >= 500:
                if attempt == 2:
                    raise RuntimeError(f"bulk source HTTP {response.status_code}: {url}")
                self.sleep(2**attempt)
                continue
            if response.status_code == 404:
                raise MissingBulkSourceError(
                    f"required bulk source {url.rsplit('/', 1)[-1]}: HTTP 404"
                )
            if response.status_code != 200:
                raise ValueError(
                    f"required bulk source {url.rsplit('/', 1)[-1]}: HTTP {response.status_code}"
                )
            content = response.content
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise SourceValidationError(f"invalid bulk source JSON: {url}") from exc
            return payload, {
                "url": url,
                "sha256": sha256(content).hexdigest(),
                "bytes": len(content),
                "digest_basis": "response_bytes",
                "commit": commit,
            }
        raise RuntimeError("bulk source retry exhausted")

    @staticmethod
    def _laps(payload: Any) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise SourceValidationError("session_laptimes.json must be a columnar object")
        needed = {"drv", "dNum", "team", "lap", "time", "lSD", "lST", "iacc", "del"}
        if missing := needed - set(payload):
            raise SourceValidationError(f"session_laptimes.json missing {sorted(missing)}")
        try:
            source = pd.DataFrame(payload)
        except ValueError as exc:
            raise SourceValidationError("session_laptimes.json columns have unequal lengths") from exc
        if source.empty or source["drv"].nunique() < 2:
            raise SourceValidationError("session_laptimes.json has insufficient driver coverage")
        for identity in ("drv", "dNum"):
            values = source[identity]
            if values.isna().any() or values.astype(str).str.strip().isin({"", "None", "nan"}).any():
                raise SourceValidationError("missing bulk driver identity")
        n = len(source)

        def col(name: str, default: Any = np.nan) -> pd.Series:
            return source[name] if name in source else pd.Series([default] * n)

        laps = pd.DataFrame(
            {
                "Driver": source["drv"].astype(str),
                "DriverNumber": source["dNum"].astype(str),
                "Team": source["team"].astype(str),
                "LapNumber": pd.to_numeric(source["lap"], errors="raise").astype(int),
                "LapTime": _timedelta(source["time"]),
                "LapStartTime": _timedelta(source["lST"]),
                "LapStartDate": pd.to_datetime(
                    source["lSD"].replace("None", pd.NaT), utc=True, errors="coerce"
                ),
                "Compound": col("compound").replace("None", None),
                "TyreLife": _seconds(col("life")),
                "Stint": _seconds(col("stint")),
                "IsAccurate": col("iacc", False).map(lambda value: value is True),
                "Deleted": col("del", True).map(lambda value: value is True),
                "PitInTime": _timedelta(col("pin")),
                "PitOutTime": _timedelta(col("pout")),
                "IsPersonalBest": col("pb", False).map(lambda value: value is True),
            }
        )
        if laps.duplicated(["Driver", "LapNumber"]).any():
            raise SourceValidationError("duplicate bulk lap identity")
        return laps

    @staticmethod
    def _weather(payload: Any) -> pd.DataFrame:
        if not isinstance(payload, dict) or not {"wAT", "wTT"}.issubset(payload):
            raise SourceValidationError("weather.json needs ambient and track temperatures")
        frame = pd.DataFrame(
            {
                "AirTemp": pd.to_numeric(pd.Series(payload["wAT"]), errors="coerce"),
                "TrackTemp": pd.to_numeric(pd.Series(payload["wTT"]), errors="coerce"),
            }
        )
        if frame.empty or frame["AirTemp"].isna().all() or frame["TrackTemp"].isna().all():
            raise SourceValidationError("weather.json contains no usable conditions")
        return frame

    @staticmethod
    def _results(rows: list[dict], target: str) -> pd.DataFrame:
        if not rows or len(rows) < 2:
            raise SourceValidationError(f"Jolpica {target} classifications unavailable")
        records = []
        for row in rows:
            driver = row.get("Driver") or {}
            code = driver.get("code")
            if not code:
                raise SourceValidationError("Jolpica classification missing driver code")
            position = pd.to_numeric(row.get("position"), errors="coerce")
            record = {
                "Abbreviation": code,
                "DriverNumber": str(row.get("number", driver.get("permanentNumber", ""))),
                "TeamName": (row.get("Constructor") or {}).get("name"),
                "Position": int(position) if pd.notna(position) else pd.NA,
                "Status": row.get("status")
                or ("Classified" if target == "Q" and pd.notna(position) else None),
            }
            if target == "Q":
                record.update({phase: _q_time(row.get(phase)) for phase in ("Q1", "Q2", "Q3")})
            records.append(record)
        result = pd.DataFrame(records)
        if result["Abbreviation"].duplicated().any():
            raise SourceValidationError("duplicate Jolpica classification driver")
        return result

    @staticmethod
    def _trace(payload: Any) -> pd.DataFrame:
        source = payload.get("tel") if isinstance(payload, dict) else None
        if not isinstance(source, dict):
            raise SourceValidationError("telemetry JSON missing tel object")
        required = {"distance", "speed", "throttle", "brake"}
        if missing := required - set(source):
            raise SourceValidationError(f"telemetry JSON missing {sorted(missing)}")
        try:
            frame = pd.DataFrame(
                {
                    "Distance": source["distance"],
                    "Speed": source["speed"],
                    "Throttle": source["throttle"],
                    "Brake": source["brake"],
                }
            )
        except ValueError as exc:
            raise SourceValidationError("telemetry JSON channels have unequal lengths") from exc
        if frame.empty:
            raise SourceValidationError("telemetry JSON has no samples")
        for source_name, output_name in (("x", "X"), ("y", "Y")):
            if source_name in source:
                if len(source[source_name]) != len(frame):
                    raise SourceValidationError("telemetry position channel has unequal length")
                frame[output_name] = source[source_name]
        return frame

    def fetch_session(self, year: int, event: str | int, session_type: str) -> RawSession:
        kind = session_type.upper()
        if kind not in SESSION_FOLDERS or year not in COMMITS:
            raise ValueError("bulk adapter supports pinned 2022-2026 FP1/FP2/FP3/Q/R only")
        try:
            round_number = int(event)
        except (TypeError, ValueError) as exc:
            raise ValueError("bulk adapter requires an official numeric round") from exc
        race = next(
            (row for row in self.jolpica.fetch_races(year) if int(row["round"]) == round_number),
            None,
        )
        if race is None:
            raise ValueError("bulk session is absent from Jolpica calendar")
        calendar = event_from_race(race)
        source_session = next((item for item in calendar.sessions if item.kind == kind), None)
        grand_prix = next((item for item in calendar.sessions if item.kind == "R"), None)
        if source_session is None or grand_prix is None:
            raise ValueError("Jolpica calendar lacks target or Grand Prix schedule")
        now = pd.Timestamp(self.now())
        if now.tzinfo is None or now < pd.Timestamp(grand_prix.end) + timedelta(hours=24):
            raise ValueError("bulk adapter requires a fully past Grand Prix weekend")
        session_start, schedule_source = self.practice_roster.fetch_schedule(year, round_number, kind)
        if session_start.tzinfo is None or session_start >= now:
            raise SourceValidationError("FastF1 schedule must provide a past UTC session start")
        commit = COMMITS[year]
        event_name = self.event_name_overrides.get((year, round_number), race["raceName"])
        if not isinstance(event_name, str) or not event_name:
            raise ValueError("Jolpica calendar lacks a usable event name")
        base = (
            f"https://raw.githubusercontent.com/TracingInsights/{year}/{commit}/"
            f"{quote(event_name, safe='')}/{quote(SESSION_FOLDERS[kind], safe='')}"
        )
        source_files: list[dict[str, str | int]] = [schedule_source]
        payloads = {}
        for name in ("session_laptimes.json", "weather.json", "drivers.json"):
            payloads[name], evidence = self._read_json(f"{base}/{name}", commit)
            source_files.append(evidence)
        drivers = payloads["drivers.json"]
        if not isinstance(drivers, dict) or not isinstance(drivers.get("drivers"), list):
            raise SourceValidationError("drivers.json missing driver roster")
        laps = self._laps(payloads["session_laptimes.json"])
        weather = self._weather(payloads["weather.json"])
        calendar_url = f"{_JOLPICA_BASE}/{year}.json"
        source_files.append(_canonical_source_record(calendar_url, race))
        identities = self.jolpica.fetch_drivers(year)
        source_files.append(
            _canonical_source_record(f"{_JOLPICA_BASE}/{year}/drivers.json", identities)
        )
        by_code = {row.get("code"): row.get("driverId") for row in identities if row.get("code")}
        driver_ids = {
            str(code): str(by_code[code]) for code in laps["Driver"].unique() if by_code.get(code)
        }
        results = pd.DataFrame()
        if kind.startswith("FP"):
            try:
                results, roster_source = self.practice_roster.fetch_driver_list(
                    year, round_number, kind
                )
            except (SourceValidationError, httpx.HTTPError, OSError) as exc:
                raise PracticeRosterUnavailableError(
                    f"{year}-{round_number:02d}-{kind} same-session FastF1 DriverList unavailable"
                ) from exc
            if (
                results.empty
                or "Abbreviation" not in results
                or results["Abbreviation"].duplicated().any()
            ):
                raise PracticeRosterUnavailableError("same-session FastF1 DriverList invalid")
            if not set(laps["Driver"]).issubset(set(results["Abbreviation"])):
                raise PracticeRosterUnavailableError(
                    "same-session FastF1 DriverList omits observed lap driver"
                )
            source_files.append(roster_source)
            for code in results["Abbreviation"].astype(str):
                if by_code.get(code):
                    driver_ids[code] = str(by_code[code])
        if kind in {"Q", "R"}:
            result_rows = self.jolpica.fetch_results(
                year, round_number, cast(Literal["Q", "R"], kind)
            )
            route = "qualifying" if kind == "Q" else "results"
            source_files.append(
                _canonical_source_record(
                    f"{_JOLPICA_BASE}/{year}/{round_number}/{route}.json", result_rows
                )
            )
            results = self._results(result_rows, kind)
            for row in result_rows:
                driver = row.get("Driver") or {}
                if driver.get("code") and driver.get("driverId"):
                    driver_ids[str(driver["code"])] = str(driver["driverId"])
        telemetry: dict[str, pd.DataFrame] = {}
        if kind.startswith("FP"):
            eligible = laps.loc[laps["IsAccurate"] & laps["LapTime"].notna()]
            choices = []
            for driver, group in eligible.groupby("Driver"):
                lap = group.sort_values(["LapTime", "LapNumber"]).iloc[0]
                choices.append((str(driver), int(lap["LapNumber"])))

            def fetch_trace(choice: tuple[str, int]):
                driver, lap_number = choice
                payload, evidence = self._read_json(
                    f"{base}/{quote(driver, safe='')}/{lap_number}_tel.json", commit
                )
                return driver, self._trace(payload), evidence

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(fetch_trace, choice) for choice in choices]
                for future in as_completed(futures):
                    driver, trace, evidence = future.result()
                    telemetry[driver] = trace
                    source_files.append(evidence)
        last_lap = laps["LapStartDate"].max()
        longest_lap = laps["LapTime"].max()
        end = session_start + (source_session.end - source_session.start)
        if pd.notna(last_lap):
            end = max(end, last_lap + longest_lap if pd.notna(longest_lap) else last_lap)
        if end >= now:
            raise ValueError("bulk session has a future observed end")
        source_files.sort(key=lambda item: str(item["url"]))
        source = f"TracingInsights/{year}@{commit}"
        source_labels = {
            "laps": source,
            "weather": source,
            "telemetry": source if telemetry else "not fetched; target-session telemetry excluded",
            "results": "Jolpica official classification"
            if kind in {"Q", "R"}
            else "FastF1 same-session DriverList",
            "schedule": "FastF1 UTC event schedule; Jolpica event/circuit identity",
            "driver_identity": "Jolpica driverId with source driver code fallback",
        }
        return RawSession(
            season=year,
            event_id=f"{year}-{round_number:02d}",
            circuit_id=race["Circuit"]["circuitId"],
            session_type=kind,
            session_start=session_start,
            session_end=end,
            laps=laps,
            results=results,
            weather=weather,
            telemetry=telemetry,
            source=source,
            driver_ids=driver_ids,
            completed=True,
            source_files=source_files,
            source_labels=source_labels,
            completion_evidence=(
                "Published pinned historical JSON after scheduled Grand Prix end + 24h"
                "; Jolpica classification present for Q/R; FastF1 Finalised not asserted"
            ),
            extraction_version="fastf1-merged-v1",
            telemetry_omission_reason="target-session telemetry excluded" if kind in {"Q", "R"} else None,
        )


def restore_source_stints(archived_laps: pd.DataFrame, payload: Any) -> pd.DataFrame | None:
    """Restore only source stint IDs after an exact driver/lap identity check."""
    if not isinstance(payload, dict) or "stint" not in payload:
        return None
    source = TracingInsightsProvider._laps(payload)
    if source["Stint"].isna().all():
        return None
    keys = ["Driver", "LapNumber"]
    archived_index = pd.MultiIndex.from_frame(archived_laps[keys])
    source = source.set_index(keys)
    if (
        archived_index.has_duplicates
        or source.index.has_duplicates
        or len(source) != len(archived_laps)
        or set(source.index) != set(archived_index)
    ):
        raise SourceValidationError("source stint identity mismatch with archived laps")
    output = archived_laps.copy()
    output["Stint"] = source["Stint"].reindex(archived_index).to_numpy()
    return output
