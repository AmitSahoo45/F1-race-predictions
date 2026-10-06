"""Convert observed archive tables into compact, validated site analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pandas as pd

from .archive import load_event_snapshots
from .contracts import (
    ActualResult,
    Analysis,
    DriverAnalysis,
    EventTelemetry,
    Stint,
    TelemetryPoint,
    TelemetryTrace,
)

PRACTICE = {"FP1", "FP2", "FP3"}


def _finite(value: object) -> float | None:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        raise TypeError("expected scalar numeric value")
    if cast(bool, pd.isna(value)):
        return None
    number = float(value) if isinstance(value, bool) else float(str(value))
    return number if math.isfinite(number) else None


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    value = frame[name]
    if not isinstance(value, pd.Series):
        raise TypeError(f"duplicate column name: {name}")
    return value


def _sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame.iloc[:: max(1, math.ceil(len(frame) / limit))].head(limit)


def export_analysis(
    event_id: str, archive_dir: Path | str, *, use_catalog: bool = True
) -> tuple[Analysis, EventTelemetry]:
    """Build observed analysis and its traces only from latest local source snapshots."""
    snapshots = load_event_snapshots(archive_dir, event_id, use_catalog=use_catalog)
    if not snapshots:
        raise ValueError(f"no archived sessions for {event_id}")
    snapshots.sort(key=lambda item: item[0]["session_start"])
    geometry = next(
        (
            tables.get("circuit")
            for _, _, tables in snapshots
            if tables.get("circuit") is not None and not tables["circuit"].empty
        ),
        None,
    )
    circuit_points: list[tuple[float, float]] = []
    if geometry is not None:
        for _, point in _sample(cast(pd.DataFrame, geometry[["x", "y"]].dropna()), 300).iterrows():
            x, y = _finite(point["x"]), _finite(point["y"])
            if x is not None and y is not None:
                circuit_points.append((x, y))
    summary = pd.concat([rows for _, rows, _ in snapshots], ignore_index=True)
    drivers = []
    traces = []
    for driver_id, history in summary.groupby("driver_id", sort=True):
        history = cast(pd.DataFrame, history)
        practice = history.loc[_column(history, "session_type").isin(tuple(PRACTICE))].sort_values(
            "session_start"
        )
        latest = practice.iloc[-1] if not practice.empty else None
        long_runs = practice.loc[_column(practice, "long_run_pace_s").notna()]
        long_run = long_runs.iloc[-1] if not long_runs.empty else None
        stints: list[Stint] = []
        trace = pd.DataFrame()
        trace_session = None
        for manifest, rows, tables in snapshots:
            if manifest["session_type"] not in PRACTICE:
                continue
            row = rows.loc[_column(rows, "driver_id") == driver_id]
            if row.empty:
                continue
            code = row.iloc[0]["driver_code"]
            laps = tables.get("laps", pd.DataFrame())
            if not laps.empty and {"Driver", "usable", "lap_seconds", "Compound", "Stint"}.issubset(
                laps.columns
            ):
                usable = laps.loc[
                    (_column(laps, "Driver") == code)
                    & _column(laps, "usable")
                    & _column(laps, "Compound").notna()
                    & _column(laps, "Stint").notna()
                    & _column(laps, "lap_seconds").notna()
                ]
                keys = ["Compound", "Stint"]
                for key, stint_laps in usable.groupby(keys, dropna=False, sort=False):
                    stint_laps = cast(pd.DataFrame, stint_laps)
                    compound = key[0] if isinstance(key, tuple) else key
                    pace = _finite(_column(stint_laps, "lap_seconds").median())
                    if pace is not None:
                        stints.append(
                            Stint(compound=str(compound), laps=len(stint_laps), pace_s=pace)
                        )
            candidate = tables.get("telemetry", pd.DataFrame())
            if not candidate.empty and "driver_id" in candidate:
                selected = candidate.loc[_column(candidate, "driver_id") == driver_id]
                if not selected.empty:
                    trace, trace_session = selected, str(manifest["session_id"])
        points = []
        required = {"distance_m", "speed_kph", "throttle_fraction", "brake"}
        if required.issubset(trace.columns):
            for _, sample in _sample(trace, 300).iterrows():
                distance = _finite(sample["distance_m"])
                speed = _finite(sample["speed_kph"])
                throttle = _finite(sample["throttle_fraction"])
                brake = _finite(sample["brake"])
                if (
                    distance is not None
                    and speed is not None
                    and throttle is not None
                    and brake is not None
                ):
                    points.append(
                        TelemetryPoint(
                            distance_m=distance,
                            speed_kph=speed,
                            throttle_pct=100 * throttle,
                            brake=bool(brake),
                        )
                    )
        if points and trace_session is not None:
            traces.append(
                TelemetryTrace(driver_id=str(driver_id), session_id=trace_session, points=points)
            )
        drivers.append(
            DriverAnalysis(
                driver_id=str(driver_id),
                practice_pace_s=_finite(latest["pace_s"]) if latest is not None else None,
                long_run_pace_s=_finite(long_run["long_run_pace_s"])
                if long_run is not None
                else None,
                stints=stints[:12],
            )
        )
    actuals = []
    for manifest, rows, _ in snapshots:
        if (
            not manifest.get("coverage", {}).get("session_complete", False)
            or rows.empty
            or rows.iloc[0]["session_type"] not in {"Q", "R"}
        ):
            continue
        target = "qualifying" if rows.iloc[0]["session_type"] == "Q" else "race"
        for _, row in rows.iterrows():
            if not cast(bool, pd.isna(row["status"])):
                position = _finite(row["position"])
                actuals.append(
                    ActualResult(
                        target=target,
                        driver_id=str(row["driver_id"]),
                        position=int(position) if position is not None else None,
                        status=str(row["status"]),
                    )
                )
    newest = max(manifest["retrieved_at"] for manifest, _, _ in snapshots)
    sources = ", ".join(sorted({str(manifest["source"]) for manifest, _, _ in snapshots}))
    result_sources = ", ".join(sorted({
        str(manifest.get("sources", {}).get("results", manifest["source"]))
        for manifest, _, _ in snapshots
        if manifest.get("session_type") in {"Q", "R"}
    }))
    analysis = Analysis(
        event_id=event_id,
        kind="observed",
        source=(
            f"{sources} public observations. "
            f"Classification sources: {result_sources or 'not archived'}. "
            f"Archived retrieval latest {newest}."
        ),
        circuit_points=circuit_points,
        drivers=drivers,
        actuals=actuals,
    )
    return analysis, EventTelemetry(event_id=event_id, traces=traces)
