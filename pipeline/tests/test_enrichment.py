"""Stint enrichment preserves raw evidence and resumes immutable transformations."""

import json
from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd
import pytest
from f1forecast.archive import latest_snapshot, write_snapshot
from f1forecast.bulk_provider import COMMITS
from f1forecast.job_control import JobControl, WorkStopped


def _archive(root, session_id="2022-01-FP1", *, bulk=True):
    laps = pd.DataFrame(
        {
            "Driver": ["LEC"] * 3 + ["VER"] * 3,
            "LapNumber": [1, 2, 3] * 2,
            "lap_seconds": [90, 91, 92, 100, 101, 102],
            "usable": [True] * 6,
            "Compound": ["SOFT"] * 3 + ["MEDIUM"] * 3,
        }
    )
    if not bulk:
        laps["Stint"] = [1] * 6
    payload = {
        "drv": laps.Driver.tolist(),
        "dNum": ["16"] * 3 + ["1"] * 3,
        "team": ["Ferrari"] * 3 + ["Red Bull"] * 3,
        "lap": laps.LapNumber.tolist(),
        "time": laps.lap_seconds.tolist(),
        "lSD": ["2022-03-18T12:05:00"] * 6,
        "lST": [300, 400, 500] * 2,
        "iacc": [True] * 6,
        "del": [False] * 6,
        "stint": [1] * 6,
    }
    content = json.dumps(payload).encode()
    url = f"https://raw.githubusercontent.com/TracingInsights/2022/{COMMITS[2022]}/session_laptimes.json"
    evidence = {
        "url": url,
        "commit": COMMITS[2022],
        "sha256": sha256(content).hexdigest(),
        "bytes": len(content),
        "digest_basis": "response_bytes",
    }
    summary = pd.DataFrame(
        {
            "driver_code": ["LEC", "VER"],
            "driver_id": ["leclerc", "max_verstappen"],
            "available_at": pd.to_datetime(["2026-09-28T12:00:00Z"] * 2),
            "long_run_pace_s": [999.0, 999.0],
            "consistency_s": [999.0, 999.0],
        }
    )
    manifest = write_snapshot(
        root,
        {
            "session_id": session_id,
            "season": 2022,
            "source": "TracingInsights/pinned" if bulk else "FastF1",
            "retrieved_at": "2026-09-28T12:00:00+00:00",
            "coverage": {"session_complete": True},
            "source_files": [evidence] if bulk else [],
        },
        summary,
        {"laps": laps, "results": pd.DataFrame({"evidence": ["preserve"]})},
    )
    return manifest, payload, evidence


class Reader:
    def __init__(self, payload, evidence):
        self.payload, self.evidence = payload, evidence

    def _read_json(self, url, commit):
        assert url == self.evidence["url"] and commit == self.evidence["commit"]
        return self.payload, self.evidence


def _run(root, report, reader, *, control=None):
    from f1forecast.enrichment import run_enrichment

    return run_enrichment(
        root,
        report,
        control=control or JobControl(),
        reader=reader,
        now=lambda: datetime(2026, 10, 4, 18, tzinfo=UTC),
    )


def test_enrichment_preserves_original_evidence_and_rebuilds_summaries(tmp_path):
    archive = tmp_path / "archive"
    old, payload, evidence = _archive(archive)
    directory = archive / "snapshots" / old["snapshot_id"]
    original_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
    report = _run(archive, tmp_path / "progress.json", Reader(payload, evidence))
    assert report["status"] == "completed"
    latest, summary, tables = latest_snapshot(archive, "2022-01-FP1")
    assert latest["snapshot_id"] != old["snapshot_id"]
    assert latest["enrichments"][-1]["parent_snapshot_id"] == old["snapshot_id"]
    assert latest["lap_summary_semantics"]
    assert summary.long_run_pace_s.tolist() == [91.0, 101.0]
    assert summary.consistency_s.tolist() == [1.0, 1.0]
    assert summary.compound_soft_fraction.tolist() == [1.0, 0.0]
    assert summary.available_at.tolist() == [pd.Timestamp("2026-09-28T12:00:00Z")] * 2
    assert tables["laps"].Stint.tolist() == [1] * 6
    assert tables["results"].evidence.tolist() == ["preserve"]
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original_bytes


def test_direct_snapshots_are_recomputed_without_provider_reads_and_rerun_is_idempotent(tmp_path):
    archive = tmp_path / "archive"
    _archive(archive, bulk=False)
    report_path = tmp_path / "progress.json"
    report = _run(archive, report_path, None)
    assert report["status"] == "completed"
    first = latest_snapshot(archive, "2022-01-FP1")
    assert first[1].long_run_pace_s.tolist() == [91.0, 101.0]
    report = _run(archive, report_path, None)
    assert report["status"] == "completed"
    assert latest_snapshot(archive, "2022-01-FP1")[0] == first[0]
    assert len(list((archive / "snapshots").glob("*/manifest.json"))) == 2


def test_source_hash_change_is_reported_without_writing_a_snapshot(tmp_path):
    archive = tmp_path / "archive"
    old, payload, evidence = _archive(archive)
    report = _run(
        archive, tmp_path / "progress.json", Reader(payload, {**evidence, "sha256": "b" * 64})
    )
    assert report["status"] == "completed_with_failures"
    assert "hash" in report["sessions"][0]["error"]
    assert latest_snapshot(archive, "2022-01-FP1")[0] == old
    assert len(list((archive / "snapshots").glob("*/manifest.json"))) == 1


def test_stop_during_download_saves_checkpoint_and_resumes_once(tmp_path):
    archive = tmp_path / "archive"
    old, payload, evidence = _archive(archive)
    stop = tmp_path / "stop.flag"

    class StoppingReader(Reader):
        def _read_json(self, url, commit):
            stop.touch()
            return super()._read_json(url, commit)

    report_path = tmp_path / "progress.json"
    report = _run(
        archive, report_path, StoppingReader(payload, evidence), control=JobControl(stop_file=stop)
    )
    assert report["status"] == "stopped"
    assert latest_snapshot(archive, "2022-01-FP1")[0] == old
    stop.unlink()
    assert _run(archive, report_path, Reader(payload, evidence))["status"] == "completed"
    assert _run(archive, report_path, Reader(payload, evidence))["status"] == "completed"
    assert len(list((archive / "snapshots").glob("*/manifest.json"))) == 2


def test_cli_requires_an_explicit_future_deadline_and_stop_file(tmp_path):
    from f1forecast.enrichment import main

    with pytest.raises(SystemExit):
        main(["--archive", str(tmp_path)])
    with pytest.raises((ValueError, WorkStopped), match="deadline"):
        main(
            [
                "--archive",
                str(tmp_path),
                "--stop-at",
                "2020-01-01T00:00:00+00:00",
                "--stop-file",
                str(tmp_path / "stop.flag"),
            ]
        )


def test_resume_registers_snapshot_published_before_catalog_failure(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    old, payload, evidence = _archive(archive)
    report_path = tmp_path / "progress.json"

    def catalog_unavailable(_root):
        raise OSError("simulated interruption before catalog registration")

    with monkeypatch.context() as patch:
        patch.setattr("f1forecast.archive._connect", catalog_unavailable)
        failed = _run(archive, report_path, Reader(payload, evidence))
    assert failed["status"] == "completed_with_failures"
    assert latest_snapshot(archive, "2022-01-FP1")[0] == old
    assert len(list((archive / "snapshots").glob("*/manifest.json"))) == 2

    resumed = _run(archive, report_path, Reader(payload, evidence))
    assert resumed["status"] == "completed"
    current, summary, _ = latest_snapshot(archive, "2022-01-FP1")
    assert current["snapshot_id"] != old["snapshot_id"]
    assert summary.long_run_pace_s.tolist() == [91.0, 101.0]
    assert len(list((archive / "snapshots").glob("*/manifest.json"))) == 2
