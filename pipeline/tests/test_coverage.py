"""Filesystem-only, calendar-relative archive completeness report."""

from datetime import UTC, datetime

import pandas as pd
from f1forecast.archive import write_snapshot
from f1forecast.coverage import build_coverage_report

NOW = datetime(2025, 4, 20, tzinfo=UTC)


class Calendar:
    def fetch_races(self, year):
        assert year == 2025
        return [
            {
                "season": "2025",
                "round": "1",
                "raceName": "Example Grand Prix",
                "Circuit": {
                    "circuitId": "example",
                    "Location": {"country": "Example"},
                },
                "FirstPractice": {"date": "2025-04-01", "time": "09:00:00Z"},
                "Qualifying": {"date": "2025-04-02", "time": "09:00:00Z"},
                "date": "2025-04-03",
                "time": "09:00:00Z",
            }
        ]


def _snapshot(root, session_id, source, *, completed, complete, source_files=None):
    write_snapshot(
        root,
        {
            "session_id": session_id,
            "retrieved_at": "2025-04-19T12:00:00+00:00",
            "source": source,
            "coverage": {
                "session_complete": completed,
                "complete": complete,
                "missing_telemetry_drivers": ["AAA"] if not complete else [],
                "unmatched_identity_drivers": ["BBB"] if not complete else [],
                "missing_lap_drivers": [],
                "weather_rows": 1,
                "circuit_points": 2,
                "missing_results": False,
            },
            "source_files": source_files or [],
            "extraction_version": "fastf1-merged-v1",
            "provenance": "retrospective",
        },
        pd.DataFrame([{"session_id": session_id}]),
        {},
    )


def test_coverage_report_compares_completed_calendar_to_latest_immutable_files(
    tmp_path, monkeypatch
):
    _snapshot(tmp_path, "2025-01-FP1", "FastF1", completed=True, complete=True)
    _snapshot(tmp_path, "2026-01-FP1", "FastF1", completed=True, complete=True)
    _snapshot(
        tmp_path,
        "2025-01-Q",
        "TracingInsights/2025@abc",
        completed=False,
        complete=False,
        source_files=[
            {
                "url": "https://example.test/pinned.json",
                "sha256": "a" * 64,
                "bytes": 500,
                "commit": "b" * 40,
            }
        ],
    )
    (tmp_path / "snapshots" / ".2025-01-R_bad.staging").mkdir()

    def forbid_catalog(*_args, **_kwargs):
        raise AssertionError("coverage must not open live DuckDB catalog")

    monkeypatch.setattr("f1forecast.archive._connect", forbid_catalog)
    report = build_coverage_report(tmp_path, [2025], now=NOW, calendar_provider=Calendar())
    assert report["requested_sessions"] == 3
    assert report["archived_sessions"] == 2
    assert report["finalized_sessions"] == 1
    assert report["fully_covered_sessions"] == 1
    assert report["missing_session_ids"] == ["2025-01-R"]
    assert report["incomplete_session_ids"] == ["2025-01-Q"]
    assert report["orphan_archived_session_ids"] == []
    assert report["per_year"]["2025"]["primary_sessions"] == 1
    assert report["per_year"]["2025"]["bulk_sessions"] == 1
    assert report["missing_telemetry_driver_sessions"] == 1
    assert report["unmatched_identity_driver_sessions"] == 1
    assert report["pinned_source_file_records"] == 1
    assert report["source_file_inventory"][0]["session_id"] == "2025-01-Q"
    assert report["source_file_inventory"][0]["sha256"] == "a" * 64
    assert report["extraction_versions"] == {"fastf1-merged-v1": 2}
    assert report["archive_bytes"] > 0
    assert "not historical availability" in report["availability_note"]
