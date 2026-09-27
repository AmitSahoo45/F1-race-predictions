"""Analysis artifacts must derive from archived observations only."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from f1forecast.analysis_export import export_analysis
from f1forecast.ingestion import ingest_session
from f1forecast.providers import RawSession

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


def source_session(kind: str) -> RawSession:
    laps = pd.DataFrame(
        [
            {
                "Driver": "NOR",
                "DriverNumber": "4",
                "Team": "McLaren",
                "LapNumber": lap,
                "LapTime": np.timedelta64(90 + lap, "s"),
                "LapStartTime": np.timedelta64(lap * 2, "m"),
                "Compound": "MEDIUM",
                "TyreLife": lap,
                "IsAccurate": True,
                "Deleted": False,
            }
            for lap in (1, 2, 3)
        ]
    )
    results = pd.DataFrame(
        [
            {
                "Abbreviation": "NOR",
                "DriverNumber": "4",
                "TeamName": "McLaren",
                "Position": 1,
                "Status": "Finished",
            }
        ]
    )
    if kind == "R":
        results.loc[0, "Status"] = "Disqualified"
    return RawSession(
        2025,
        "2025-01",
        "albert_park",
        kind,
        pd.Timestamp("2025-03-14T01:30:00Z"),
        pd.Timestamp("2025-03-14T02:30:00Z"),
        laps,
        results,
        pd.DataFrame([{"AirTemp": 22.0, "TrackTemp": 30.0}]),
        {
            "NOR": pd.DataFrame(
                {
                    "Distance": [0.0, 100.0],
                    "Speed": [200.0, 250.0],
                    "Throttle": [50.0, 100.0],
                    "Brake": [False, True],
                    "X": [1.0, 2.0],
                    "Y": [3.0, 4.0],
                }
            )
        },
        completed=True,
    )


def test_export_analysis_uses_real_archive_tables_and_retains_dsq(tmp_path):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            return source_session(kind)

    provider = Provider()
    for kind in ("FP1", "Q", "R"):
        ingest_session(2025, 1, kind, tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    analysis = export_analysis("2025-01", tmp_path)
    assert analysis.kind == "observed"
    assert analysis.circuit_points == [(1.0, 3.0), (2.0, 4.0)]
    assert "FastF1 public" in analysis.source
    driver = analysis.drivers[0]
    assert driver.driver_id == "nor"
    assert driver.practice_pace_s == 92.0
    assert driver.long_run_pace_s == 92.0
    assert driver.stints[0].compound == "MEDIUM"
    assert driver.stints[0].laps == 3
    assert [p.throttle_pct for p in driver.telemetry] == [50.0, 100.0]
    assert {(a.target, a.position, a.status) for a in analysis.actuals} == {
        ("qualifying", 1, "Finished"),
        ("race", None, "Disqualified"),
    }


def test_export_analysis_rejects_absent_event(tmp_path):
    with pytest.raises(ValueError, match="no archived"):
        export_analysis("2025-99", tmp_path)


def test_export_analysis_omits_partial_race_classification(tmp_path):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.completed = False
            return raw

    ingest_session(2025, 1, "R", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    assert export_analysis("2025-01", tmp_path).actuals == []


def test_export_analysis_filesystem_reader_uses_latest_complete_snapshot_without_catalog(
    tmp_path, monkeypatch
):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.completed = True
            return raw

    provider = Provider()
    ingest_session(2025, 1, "R", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    latest = ingest_session(
        2025,
        1,
        "R",
        tmp_path,
        fastf1_provider=provider,
        retrieved_at=NOW + timedelta(hours=1),
        refresh=True,
    )
    expected = export_analysis("2025-01", tmp_path)

    def forbid_catalog(*_args, **_kwargs):
        raise AssertionError("catalog read during active writer")

    monkeypatch.setattr("f1forecast.archive._connect", forbid_catalog)
    observed = export_analysis("2025-01", tmp_path, use_catalog=False)
    assert observed == expected
    assert latest.manifest["retrieved_at"] in observed.source
