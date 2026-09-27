"""Validate and normalize public session observations before archiving them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from .archive import latest_snapshot, load_summaries, write_snapshot
from .providers import FastF1Provider, JolpicaProvider, RawSession

SESSION_TYPES = {"FP1", "FP2", "FP3", "Q", "SQ", "S", "R"}


@dataclass
class SessionSnapshot:
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    sources: dict
    manifest: dict


class SessionProvider(Protocol):
    def fetch_session(self, year: int, event: str | int, session_type: str) -> RawSession: ...


def _utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp):
        raise TypeError("missing timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Narrow Pandas' ambiguous scalar/Series/DataFrame column overload."""
    value = frame[name]
    if not isinstance(value, pd.Series):
        raise TypeError(f"duplicate column name: {name}")
    return value


def _missing(value: object) -> bool:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        raise TypeError("expected scalar value")
    return cast(bool, pd.isna(value))


def _scalar_float(value: object) -> float | None:
    return None if _missing(value) else float(str(value))


def _number(series: pd.Series, low: float, high: float, label: str) -> pd.Series:
    values = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    invalid = values.notna() & ~values.between(low, high)
    if bool(invalid.any()):
        raise ValueError(f"{label} outside expected units/range [{low}, {high}]")
    return values


def normalize_session(
    raw: RawSession,
    retrieved_at: datetime | pd.Timestamp,
    *,
    provenance: str = "retrospective",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Return driver summaries, source tables, and measured coverage flags."""
    if raw.session_type not in SESSION_TYPES:
        raise ValueError(f"unsupported session type: {raw.session_type}")
    if provenance not in {"retrospective", "prospective"}:
        raise ValueError("invalid provenance")
    start, end, retrieval = _utc(raw.session_start), _utc(raw.session_end), _utc(retrieved_at)
    if end < start or retrieval < end:
        raise ValueError("session boundaries or retrieval chronology invalid")
    laps = raw.laps.copy()
    if laps.empty:
        raise ValueError("session has no laps")
    required = {"Driver", "DriverNumber", "LapNumber", "LapTime"}
    if not required.issubset(laps.columns):
        raise ValueError(f"missing lap columns: {sorted(required - set(laps.columns))}")
    if bool(cast(pd.Series, laps[["Driver", "LapNumber"]].duplicated()).any()):
        raise ValueError("duplicate driver lap identity")
    if bool(_column(laps, "Driver").isna().any()) or bool(
        _column(laps, "DriverNumber").isna().any()
    ):
        raise ValueError("missing driver identity")
    identities = cast(pd.DataFrame, laps[["Driver", "DriverNumber"]].drop_duplicates())
    if bool(_column(identities, "Driver").duplicated().any()) or bool(
        _column(identities, "DriverNumber").duplicated().any()
    ):
        raise ValueError("ambiguous driver identity")
    lap_seconds = pd.to_timedelta(_column(laps, "LapTime"), errors="coerce").dt.total_seconds()
    laps["lap_seconds"] = _number(lap_seconds, 20.0, 300.0, "lap time seconds")
    if (
        "LapStartTime" in laps
        and bool(_column(laps, "LapStartTime").notna().any())
        and bool(cast(pd.Series, laps[["Driver", "LapStartTime"]].dropna().duplicated()).any())
    ):
        raise ValueError("duplicate lap timestamp")
    usable = _column(laps, "lap_seconds").notna()
    for column, accepted in (("IsAccurate", True), ("Deleted", False)):
        if column in laps:
            usable &= _column(laps, column).fillna(not accepted).eq(accepted)
    for column in ("PitInTime", "PitOutTime"):
        if column in laps:
            usable &= _column(laps, column).isna()
    laps["usable"] = usable
    if "TyreLife" in laps:
        laps["TyreLife"] = _number(_column(laps, "TyreLife"), 0.0, 100.0, "tyre age laps")
    weather = raw.weather.copy()
    for column, low, high in (("AirTemp", -30, 60), ("TrackTemp", -30, 90)):
        if column in weather:
            weather[column] = _number(_column(weather, column), low, high, column)
    results = raw.results.copy()
    if (
        not results.empty
        and "Abbreviation" in results
        and bool(_column(results, "Abbreviation").duplicated().any())
    ):
        raise ValueError("duplicate result driver identity")
    result_lookup = (
        results.set_index("Abbreviation") if "Abbreviation" in results else pd.DataFrame()
    )
    qualifying_best = {}
    if raw.session_type == "Q":
        for phase in ("Q1", "Q2", "Q3"):
            if phase in results:
                phase_values = _column(results, phase).dropna()
                if not phase_values.empty:
                    times = pd.to_timedelta(phase_values, errors="coerce").dt.total_seconds()
                    qualifying_best[phase] = times.min()
    telemetry_rows: list[pd.DataFrame] = []
    circuit = pd.DataFrame(columns=pd.Index(["distance_m", "x", "y"]))
    missing_telemetry = []
    missing_laps = []
    rows = []
    drivers = set(_column(laps, "Driver").dropna().astype(str))
    if not results.empty and "Abbreviation" in results:
        drivers.update(_column(results, "Abbreviation").dropna().astype(str))
    for driver in sorted(drivers):
        driver_laps = laps.loc[_column(laps, "Driver") == driver]
        if driver_laps.empty:
            missing_laps.append(driver)
        valid = driver_laps.loc[_column(driver_laps, "usable")].sort_values("LapNumber")
        result = (
            cast(pd.Series, result_lookup.loc[driver])
            if not result_lookup.empty and driver in result_lookup.index
            else None
        )
        team = (
            str(_column(driver_laps, "Team").dropna().iloc[0])
            if "Team" in laps and bool(_column(driver_laps, "Team").notna().any())
            else (
                str(result.get("TeamName"))
                if result is not None and not _missing(result.get("TeamName"))
                else None
            )
        )
        trace = raw.telemetry.get(driver, pd.DataFrame()).copy()
        trace_out = pd.DataFrame()
        if not trace.empty and "Distance" in trace:
            if "Date" in trace:
                sample_time = cast(
                    pd.Series, pd.to_datetime(_column(trace, "Date"), utc=True, errors="coerce")
                )
                if bool(sample_time.dropna().duplicated().any()):
                    raise ValueError("duplicate telemetry timestamp")
                trace_out["sample_time"] = sample_time
            # FastF1 interpolation can place the first sample centimetres before zero.
            trace_out["distance_m"] = _number(
                _column(trace, "Distance"), -1, 100000, "distance metres"
            ).clip(lower=0)
            for source, output, low, high in (
                ("Speed", "speed_kph", 0, 500),
                ("Throttle", "throttle_fraction", 0, 105),
                ("Brake", "brake", 0, 1),
                ("X", "x", -100000, 100000),
                ("Y", "y", -100000, 100000),
            ):
                if source in trace:
                    trace_out[output] = _number(_column(trace, source), low, high, output)
            if "throttle_fraction" in trace_out:
                # FastF1 interpolation can overshoot its nominal 0-100 percent channel.
                trace_out["throttle_fraction"] = (
                    _column(trace_out, "throttle_fraction").clip(upper=100) / 100.0
                )
            trace_out["driver_id"] = raw.driver_ids.get(driver, driver.lower())
            telemetry_rows.append(trace_out)
            if circuit.empty and {"x", "y"}.issubset(trace_out.columns):
                circuit = trace_out[["distance_m", "x", "y"]].dropna().copy()
        if not driver_laps.empty and (
            trace_out.empty
            or not {"speed_kph", "throttle_fraction", "brake"}.issubset(trace_out.columns)
        ):
            missing_telemetry.append(driver)
        long_run = pd.NA
        if len(valid) >= 3:
            # Consecutive valid laps are an observable stint proxy; pit laps were removed above.
            consecutive = _column(valid, "LapNumber").diff().fillna(1).eq(1)
            groups = (~consecutive).cumsum()
            runs = cast(
                pd.Series, valid.groupby(groups)["lap_seconds"].filter(lambda x: len(x) >= 3)
            )
            if not runs.empty:
                long_run = float(str(runs.median()))
        position = _scalar_float(result.get("Position")) if result is not None else None
        status = (
            str(result.get("Status"))
            if result is not None and not _missing(result.get("Status"))
            else None
        )
        if status and status.lower() in {"did not start", "disqualified", "dns", "dsq"}:
            position = None
        qualifying_gap: float | None = None
        if raw.session_type == "Q" and result is not None:
            for phase in ("Q3", "Q2", "Q1"):
                if (
                    phase in result
                    and not _missing(result[phase])
                    and not _missing(qualifying_best.get(phase))
                ):
                    phase_time = result[phase]
                    seconds = (
                        phase_time.total_seconds()
                        if isinstance(phase_time, pd.Timedelta)
                        else pd.Timedelta(str(phase_time)).total_seconds()
                    )
                    qualifying_gap = seconds - qualifying_best[phase]
                    break
        rows.append(
            {
                "event_id": raw.event_id,
                "season": raw.season,
                "circuit_id": raw.circuit_id,
                "driver_id": raw.driver_ids.get(driver, driver.lower()),
                "driver_code": driver,
                "team_id": team.lower().replace(" ", "_") if team else None,
                "session_id": f"{raw.event_id}-{raw.session_type}",
                "session_type": raw.session_type,
                "session_start": start,
                "session_end": end,
                "observed_at": end,
                "available_at": retrieval,
                "provenance": provenance,
                "position": int(position) if position is not None else pd.NA,
                "status": status,
                "pace_s": float(str(_column(valid, "lap_seconds").median()))
                if not valid.empty
                else pd.NA,
                "long_run_pace_s": long_run,
                "consistency_s": float(str(_column(valid, "lap_seconds").std()))
                if len(valid) >= 2
                else pd.NA,
                "usable_laps": len(valid),
                "tyre_age": float(str(_column(valid, "TyreLife").median()))
                if "TyreLife" in valid and bool(_column(valid, "TyreLife").notna().any())
                else pd.NA,
                "compound": str(_column(valid, "Compound").mode().iloc[0])
                if "Compound" in valid and bool(_column(valid, "Compound").notna().any())
                else None,
                "mean_speed": float(str(_column(trace_out, "speed_kph").mean()))
                if "speed_kph" in trace_out
                else pd.NA,
                "mean_throttle": float(str(_column(trace_out, "throttle_fraction").mean()))
                if "throttle_fraction" in trace_out
                else pd.NA,
                "brake_fraction": float(str(_column(trace_out, "brake").mean()))
                if "brake" in trace_out
                else pd.NA,
                "air_temp": float(str(_column(weather, "AirTemp").mean()))
                if "AirTemp" in weather and bool(_column(weather, "AirTemp").notna().any())
                else pd.NA,
                "track_temp": float(str(_column(weather, "TrackTemp").mean()))
                if "TrackTemp" in weather and bool(_column(weather, "TrackTemp").notna().any())
                else pd.NA,
                "qualifying_gap": qualifying_gap,
            }
        )
    summary = pd.DataFrame(rows)
    if bool(_column(summary, "driver_id").duplicated().any()):
        raise ValueError("duplicate normalized driver identity")
    telemetry = (
        pd.concat(telemetry_rows, ignore_index=True)
        if telemetry_rows
        else pd.DataFrame(columns=pd.Index(["distance_m", "driver_id"]))
    )
    tables = {
        "laps": laps,
        "results": results,
        "weather": weather,
        "telemetry": telemetry,
        "circuit": circuit,
    }
    coverage = {
        "drivers": len(summary),
        "laps": len(laps),
        "usable_laps": int(_column(laps, "usable").sum()),
        "missing_telemetry_drivers": missing_telemetry,
        "missing_lap_drivers": missing_laps,
        "weather_rows": len(weather),
        "circuit_points": len(circuit),
    }
    missing_results = raw.session_type in {"Q", "R"} and bool(
        _column(summary, "status").isna().any()
    )
    coverage["session_complete"] = raw.completed
    coverage["completion_evidence"] = raw.completion_evidence
    coverage["complete"] = raw.completed and not missing_telemetry and not missing_results
    coverage["missing_results"] = bool(missing_results)
    return summary, tables, coverage


def ingest_session(
    year: int,
    event: str | int,
    session: str,
    archive_dir: Path | str,
    *,
    fastf1_provider: SessionProvider | None = None,
    jolpica_provider: JolpicaProvider | None = None,
    retrieved_at: datetime | None = None,
    refresh: bool = False,
    provenance: str = "retrospective",
) -> SessionSnapshot:
    """Fetch one session unless already archived, then return its immutable snapshot."""
    root = Path(archive_dir)
    session_type = session.upper()
    if isinstance(event, int):
        existing = latest_snapshot(root, f"{year}-{event:02d}-{session_type}")
        if (
            existing
            and not refresh
            and existing[0].get("provenance") == provenance
            and existing[0].get("coverage", {}).get("complete") is True
            and existing[0].get("coverage", {}).get("session_complete") is True
        ):
            manifest, summary, tables = existing
            return SessionSnapshot(summary, tables, manifest["sources"], manifest)
    fastf1_provider = fastf1_provider or FastF1Provider(root / "fastf1-cache")
    raw = fastf1_provider.fetch_session(year, event, session_type)
    sources = {
        "laps": raw.source,
        "results": raw.source,
        "weather": raw.source,
        "telemetry": raw.source,
        "schedule": f"{raw.source} event metadata",
    }
    sources.update(raw.source_labels)
    unmatched_identities: list[str] = []
    if jolpica_provider is not None:
        round_number = int(raw.event_id.rsplit("-", 1)[1])
        race = next(
            (r for r in jolpica_provider.fetch_races(year) if int(r["round"]) == round_number), None
        )
        if race:
            raw.circuit_id = race["Circuit"]["circuitId"]
            sources["schedule"] = "Jolpica"
        drivers = jolpica_provider.fetch_drivers(year)
        by_code: dict[str, str] = {}
        for driver in drivers:
            code, identity = driver.get("code"), driver.get("driverId")
            if code and identity:
                if code in by_code and by_code[code] != identity:
                    raise ValueError(f"ambiguous Jolpica driver code: {code}")
                by_code[code] = identity
        fastf1_codes = set(raw.laps["Driver"].dropna().astype(str))
        if "Abbreviation" in raw.results:
            fastf1_codes.update(raw.results["Abbreviation"].dropna().astype(str))
        for code in fastf1_codes:
            if code in by_code and code in raw.driver_ids and raw.driver_ids[code] != by_code[code]:
                raise ValueError(f"conflicting source driver identity: {code}")
        raw.driver_ids = {
            **raw.driver_ids,
            **{code: by_code[code] for code in fastf1_codes if code in by_code},
        }
        unmatched_identities = sorted(fastf1_codes - raw.driver_ids.keys())
        sources["driver_identity"] = "Jolpica driverId with FastF1 code fallback"
    else:
        sources["driver_identity"] = (
            f"{raw.source} driver IDs" if raw.driver_ids else f"{raw.source} driver code"
        )
    retrieved = retrieved_at or datetime.now(UTC)
    summary, tables, coverage = normalize_session(raw, retrieved, provenance=provenance)
    coverage["unmatched_identity_drivers"] = unmatched_identities
    manifest = write_snapshot(
        root,
        {
            "event_id": raw.event_id,
            "session_id": f"{raw.event_id}-{session_type}",
            "season": year,
            "circuit_id": raw.circuit_id,
            "session_type": session_type,
            "session_start": _utc(raw.session_start).isoformat(),
            "session_end": _utc(raw.session_end).isoformat(),
            "retrieved_at": _utc(retrieved).isoformat(),
            "source": raw.source,
            "sources": sources,
            "source_files": raw.source_files,
            "extraction_version": raw.extraction_version,
            "coverage": coverage,
            "provenance": provenance,
        },
        summary,
        tables,
    )
    return SessionSnapshot(summary, tables, sources, manifest)


__all__ = ["SessionSnapshot", "ingest_session", "load_summaries", "normalize_session"]
