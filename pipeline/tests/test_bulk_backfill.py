"""Resumable historical import chooses pinned bulk data and honest fallbacks."""

from datetime import UTC, datetime

import pytest
from f1forecast.bulk_backfill import _save_report, run_bulk_backfill
from f1forecast.bulk_provider import MissingBulkSourceError


def test_progress_checkpoint_retries_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    original = Path.replace
    attempts = []

    def replace(path, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("reader still has previous progress file open")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    target = tmp_path / "progress.json"
    _save_report(target, {"counts": {"skipped": 12}})
    assert len(attempts) == 2
    assert json.loads(target.read_text(encoding="utf-8"))["counts"]["skipped"] == 12


class _Calendar:
    def fetch_races(self, year):
        return [
            {
                "season": str(year),
                "round": "1",
                "raceName": "Test Grand Prix",
                "Circuit": {"circuitId": "test", "Location": {"country": "Test"}},
                "FirstPractice": {"date": "2022-03-18", "time": "12:00:00Z"},
                "Qualifying": {"date": "2022-03-19", "time": "15:00:00Z"},
                "date": "2022-03-20",
                "time": "15:00:00Z",
            }
        ]


class _Bulk:
    def fetch_session(self, year, event, session_type):
        if session_type == "Q":
            raise MissingBulkSourceError("weather.json absent")
        return object()


def test_backfill_skips_completed_primary_and_falls_back_only_for_missing_source(
    tmp_path, monkeypatch
):
    archive = tmp_path / "archive"
    existing = archive / "snapshots" / "2022-01-FP1_primary"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text(
        '{"session_id":"2022-01-FP1","source":"FastF1",'
        '"retrieved_at":"2022-03-21T00:00:00+00:00",'
        '"coverage":{"session_complete":true}}',
        encoding="utf-8",
    )
    calls = []

    def ingest(
        year,
        event,
        kind,
        destination,
        *,
        fastf1_provider,
        provenance,
        refresh,
        jolpica_provider=None,
    ):
        calls.append((kind, fastf1_provider.__class__.__name__, refresh))
        fastf1_provider.fetch_session(year, event, kind)
        return type(
            "Snapshot",
            (),
            {
                "manifest": {
                    "source": fastf1_provider.__class__.__name__,
                    "coverage": {"session_complete": True},
                }
            },
        )()

    class Direct:
        def fetch_session(self, year, event, kind):
            return object()

    monkeypatch.setattr("f1forecast.bulk_backfill.ingest_session", ingest)
    report = run_bulk_backfill(
        archive,
        years=(2022,),
        jolpica=_Calendar(),
        bulk=_Bulk(),
        direct=Direct(),
        now=datetime(2026, 9, 26, tzinfo=UTC),
        report_path=tmp_path / "progress.json",
    )
    assert calls == [("Q", "_Bulk", True), ("Q", "Direct", True), ("R", "_Bulk", True)]
    assert report["counts"] == {
        "skipped": 1,
        "bulk": 1,
        "direct": 0,
        "direct_fallback": 1,
        "failed": 0,
    }
    assert (tmp_path / "progress.json").exists()


def test_backfill_does_not_fallback_for_invalid_bulk_schema(tmp_path, monkeypatch):
    class Invalid:
        def fetch_session(self, year, event, kind):
            raise ValueError("lap schema invalid")

    class Direct:
        def fetch_session(self, year, event, kind):
            pytest.fail("schema error must be reported, not concealed by fallback")

    def ingest(
        year,
        event,
        kind,
        destination,
        *,
        fastf1_provider,
        provenance,
        refresh,
        jolpica_provider=None,
    ):
        fastf1_provider.fetch_session(year, event, kind)

    monkeypatch.setattr("f1forecast.bulk_backfill.ingest_session", ingest)
    report = run_bulk_backfill(
        tmp_path,
        years=(2022,),
        jolpica=_Calendar(),
        bulk=Invalid(),
        direct=Direct(),
        now=datetime(2026, 9, 26, tzinfo=UTC),
        report_path=tmp_path / "progress.json",
    )
    assert report["counts"]["failed"] == 3
    assert report["counts"]["direct_fallback"] == 0


def test_backfill_includes_sprint_sessions_using_direct_source(tmp_path, monkeypatch):
    class SprintCalendar(_Calendar):
        def fetch_races(self, year):
            row = super().fetch_races(year)[0]
            row["SprintQualifying"] = {"date": "2022-03-18", "time": "16:00:00Z"}
            row["Sprint"] = {"date": "2022-03-19", "time": "12:00:00Z"}
            return [row]

    direct = object()
    calls = []

    def ingest(year, event, kind, destination, **kwargs):
        calls.append((kind, kwargs["fastf1_provider"] is direct))
        return type(
            "Snapshot", (), {"manifest": {"source": "test", "coverage": {"session_complete": True}}}
        )()

    monkeypatch.setattr("f1forecast.bulk_backfill.ingest_session", ingest)
    run_bulk_backfill(
        tmp_path,
        years=(2022,),
        jolpica=SprintCalendar(),
        bulk=object(),
        direct=direct,
        now=datetime(2026, 9, 28, tzinfo=UTC),
    )
    assert ("SQ", True) in calls and ("S", True) in calls


def test_past_stop_deadline_returns_checkpoint_without_provider_calls(tmp_path):
    class Never:
        def fetch_races(self, year):
            pytest.fail("provider called after deadline")

    report = run_bulk_backfill(
        tmp_path,
        years=(2022,),
        jolpica=Never(),
        bulk=object(),
        direct=object(),
        stop_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert report["status"] == "stopped"
    assert report["sessions"] == []
