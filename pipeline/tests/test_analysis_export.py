"""Analysis artifacts must derive from archived observations only."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from f1forecast.analysis_export import export_analysis
from f1forecast.contracts import EventTelemetry
from f1forecast.ingestion import ingest_session
from f1forecast.providers import RawSession
from f1forecast.publication import publish_telemetry, verify_telemetry

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
                "Stint": 1,
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
    analysis, telemetry = export_analysis("2025-01", tmp_path)
    assert analysis.kind == "observed"
    assert analysis.circuit_points == [(1.0, 3.0), (2.0, 4.0)]
    assert "FastF1 public" in analysis.source
    driver = analysis.drivers[0]
    assert driver.driver_id == "nor"
    assert driver.practice_pace_s == 92.0
    assert driver.long_run_pace_s == 92.0
    assert driver.stints[0].compound == "MEDIUM"
    assert driver.stints[0].laps == 3
    assert "telemetry" not in driver.model_dump()
    assert telemetry.event_id == "2025-01"
    assert [(t.driver_id, t.session_id) for t in telemetry.traces] == [("nor", "2025-01-FP1")]
    assert [p.throttle_pct for p in telemetry.traces[0].points] == [50.0, 100.0]
    assert {(a.target, a.position, a.status) for a in analysis.actuals} == {
        ("qualifying", 1, "Finished"),
        ("race", None, "Disqualified"),
    }


def test_export_analysis_rejects_absent_event(tmp_path):
    with pytest.raises(ValueError, match="no archived"):
        export_analysis("2025-99", tmp_path)


@pytest.mark.parametrize("missing", ["column", "values"])
def test_export_analysis_does_not_invent_stints_without_source_boundaries(tmp_path, missing):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            if missing == "column":
                raw.laps = raw.laps.drop(columns="Stint")
            else:
                raw.laps["Stint"] = float("nan")
            return raw

    ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    assert export_analysis("2025-01", tmp_path)[0].drivers[0].stints == []


def test_export_analysis_keeps_separate_same_compound_source_stints(tmp_path):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.laps["Stint"] = [1, 2, 2]
            return raw

    ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    stints = export_analysis("2025-01", tmp_path)[0].drivers[0].stints
    assert [(stint.laps, stint.pace_s) for stint in stints] == [(1, 91.0), (2, 92.5)]


def test_export_analysis_attributes_actual_bulk_and_classification_sources(tmp_path):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.source = "TracingInsights/2025@abc123"
            raw.source_labels["results"] = "Jolpica official classification"
            return raw

    ingest_session(2025, 1, "Q", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    analysis, _ = export_analysis("2025-01", tmp_path)
    assert "TracingInsights/2025@abc123" in analysis.source
    assert "Jolpica official classification" in analysis.source
    assert "FastF1 public" not in analysis.source


def test_export_analysis_omits_partial_race_classification(tmp_path):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.completed = False
            return raw

    ingest_session(2025, 1, "R", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    assert export_analysis("2025-01", tmp_path)[0].actuals == []


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
    assert latest.manifest["retrieved_at"] in observed[0].source


@pytest.mark.parametrize("use_catalog", [True, False])
def test_completed_export_retains_latest_finalized_practice_before_new_partial(
    tmp_path, use_catalog
):
    class Provider:
        completed = True
        extra_seconds = 0

        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.completed = self.completed
            raw.laps["LapTime"] = [
                np.timedelta64(90 + lap + self.extra_seconds, "s")
                for lap in raw.laps["LapNumber"]
            ]
            return raw

    provider = Provider()
    ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    provider.extra_seconds = 10
    latest_finalized = ingest_session(
        2025, 1, "FP1", tmp_path, fastf1_provider=provider,
        retrieved_at=NOW + timedelta(hours=1), refresh=True,
    )
    provider.completed = False
    provider.extra_seconds = 20
    ingest_session(
        2025, 1, "FP1", tmp_path, fastf1_provider=provider,
        retrieved_at=NOW + timedelta(hours=2), refresh=True,
    )

    analysis, telemetry = export_analysis(
        "2025-01", tmp_path, use_catalog=use_catalog, completed_only=True
    )
    assert analysis.drivers[0].practice_pace_s == 102.0
    assert latest_finalized.manifest["retrieved_at"] in analysis.source
    assert telemetry.traces[0].session_id == "2025-01-FP1"
    # Existing callers still see the newest snapshot unless they request this gate.
    latest, _ = export_analysis("2025-01", tmp_path, use_catalog=use_catalog)
    assert latest.drivers[0].practice_pace_s == 112.0


@pytest.mark.parametrize("use_catalog", [True, False])
def test_completed_export_rejects_only_partial_sessions(tmp_path, use_catalog):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.completed = False
            return raw

    ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW)
    with pytest.raises(ValueError, match="no archived"):
        export_analysis("2025-01", tmp_path, use_catalog=use_catalog, completed_only=True)


@pytest.mark.parametrize("use_catalog", [True, False])
def test_completed_export_keeps_finalized_practice_without_optional_telemetry(
    tmp_path, use_catalog
):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            raw = source_session(kind)
            raw.telemetry = {}
            return raw

    snapshot = ingest_session(
        2025, 1, "FP1", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW
    )
    assert snapshot.manifest["coverage"]["session_complete"] is True
    assert snapshot.manifest["coverage"]["complete"] is False
    analysis, telemetry = export_analysis(
        "2025-01", tmp_path, use_catalog=use_catalog, completed_only=True
    )
    assert analysis.drivers[0].practice_pace_s == 92.0
    assert telemetry.traces == []


def _exported(archive):
    class Provider:
        def fetch_session(self, _year, _event, kind):
            return source_session(kind)

    ingest_session(2025, 1, "FP1", archive, fastf1_provider=Provider(), retrieved_at=NOW)
    return export_analysis("2025-01", archive)


def test_event_telemetry_rejects_duplicate_drivers_and_sessions_from_other_events():
    point = {"distance_m": 0.0, "speed_kph": 200.0, "throttle_pct": 50.0, "brake": False}
    trace = {"driver_id": "nor", "session_id": "2025-01-FP1", "points": [point]}
    EventTelemetry.model_validate({"event_id": "2025-01", "traces": [trace]})
    with pytest.raises(ValueError, match="one published trace"):
        EventTelemetry.model_validate({"event_id": "2025-01", "traces": [trace, trace]})
    with pytest.raises(ValueError, match="session"):
        EventTelemetry.model_validate(
            {"event_id": "2025-01", "traces": [{**trace, "session_id": "2025-02-FP1"}]}
        )
    with pytest.raises(ValueError):
        EventTelemetry.model_validate({"event_id": "2025-01", "traces": [{**trace, "points": []}]})


def test_telemetry_is_published_beside_the_site_as_compact_json(tmp_path):
    _, telemetry = _exported(tmp_path / "archive")
    site_dir = tmp_path / "site-data"
    path = publish_telemetry(telemetry, site_dir)
    assert path == site_dir / "telemetry" / "2025-01.json"
    raw = path.read_bytes()
    assert b"\n " not in raw
    assert EventTelemetry.model_validate_json(raw) == telemetry
    written = path.stat().st_mtime_ns
    publish_telemetry(telemetry, site_dir)
    assert path.stat().st_mtime_ns == written


def test_telemetry_files_must_belong_to_published_analyses(tmp_path):
    analysis, telemetry = _exported(tmp_path / "archive")
    site_dir = tmp_path / "site-data"
    path = publish_telemetry(telemetry, site_dir)
    assert verify_telemetry([analysis], site_dir) == 1
    with pytest.raises(ValueError, match="no published analysis"):
        verify_telemetry([], site_dir)
    with pytest.raises(ValueError, match="absent from the analysis"):
        verify_telemetry([analysis.model_copy(update={"drivers": []})], site_dir)
    path.rename(path.with_name("2025-02.json"))
    with pytest.raises(ValueError, match="file name"):
        verify_telemetry([analysis], site_dir)
