"""One-read, validated dataset freezes cannot replace existing user data."""

import json
from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd
import pytest
from f1forecast import datasets
from f1forecast.training import input_dataset_sha256


def _summaries():
    rows = []
    for kind, hour in (("FP1", 9), ("Q", 12), ("R", 15)):
        for position, driver in enumerate(("a", "b"), 1):
            start = pd.Timestamp(f"2024-03-01T{hour:02}:00:00Z")
            end = pd.Timestamp(f"2024-03-01T{hour + 1:02}:00:00Z")
            rows.append({
                "event_id": "2024-01", "season": 2024, "circuit_id": "example",
                "driver_id": driver, "driver_code": driver.upper(), "team_id": driver,
                "session_id": f"2024-01-{kind}", "session_type": kind,
                "session_start": start, "session_end": end, "observed_at": end,
                "available_at": pd.Timestamp("2026-10-04T12:00:00Z"),
                "provenance": "retrospective", "session_complete": True,
                "source_snapshot_id": f"2024-01-{kind}_original",
                "position": position if kind != "FP1" else None,
                "status": "Finished" if kind != "FP1" else None,
                "pace_s": 90.0 + position, "usable_laps": 5,
            })
    return pd.DataFrame(rows)


def test_training_rows_from_supplied_frame_is_pure_and_preserves_existing_prepare_api(
    tmp_path, monkeypatch,
):
    summaries = _summaries()
    original = summaries.copy(deep=True)
    rows = datasets.training_rows_from_summaries(summaries)
    assert len(rows) == 4
    assert set(rows.target) == {"qualifying", "race"}
    pd.testing.assert_frame_equal(summaries, original)
    monkeypatch.setattr(datasets, "load_summaries", lambda *_args, **_kwargs: summaries)
    path = tmp_path / "existing-api.parquet"
    prepared = datasets.prepare_training_data("archive", path, use_catalog=False)
    pd.testing.assert_frame_equal(prepared, rows)
    assert len(pd.read_parquet(path)) == 4


def test_freeze_reads_once_and_records_verified_artifacts_from_that_exact_frame(tmp_path, monkeypatch):
    original = _summaries()
    reads = []

    def advancing_archive(archive, *, use_catalog):
        reads.append((archive, use_catalog))
        if len(reads) == 1:
            return original
        return original.assign(source_snapshot_id="later_revision", pace_s=1.0)

    monkeypatch.setattr(datasets, "load_summaries", advancing_archive)
    destination = tmp_path / "freeze"
    manifest = datasets.freeze_training_data("archive", destination)
    history = pd.read_parquet(destination / "history.parquet")
    training = pd.read_parquet(destination / "training.parquet")
    assert reads == [("archive", False)]
    pd.testing.assert_frame_equal(history, original)
    assert sorted(history.source_snapshot_id.unique()) == [
        "2024-01-FP1_original", "2024-01-Q_original", "2024-01-R_original",
    ]
    assert manifest["source_snapshot_ids"] == sorted(history.source_snapshot_id.unique())
    assert manifest["history_rows"] == 6
    assert manifest["history_sessions"] == 3
    assert manifest["history_events"] == 1
    assert manifest["training_rows"] == 4
    assert manifest["training_events"] == 1
    assert manifest["target_events"] == {"qualifying": 1, "race": 1}
    assert manifest["input_dataset_sha256"] == input_dataset_sha256(training)
    assert manifest["validation"]["event_groups"] == {"qualifying": 1, "race": 1}
    assert manifest["feature_semantics"]["stints"] == "source-stints-compound-fractions-v1"
    assert datetime.fromisoformat(manifest["generated_at"]).tzinfo == UTC
    for filename in ("history.parquet", "training.parquet"):
        assert manifest["files"][filename]["sha256"] == sha256(
            (destination / filename).read_bytes()
        ).hexdigest()
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert sorted(path.name for path in tmp_path.iterdir()) == ["freeze"]


@pytest.mark.parametrize("existing", ["sentinel.txt", "manifest.json"])
def test_freeze_refuses_existing_data_without_reading_archive(tmp_path, monkeypatch, existing):
    destination = tmp_path / "freeze"
    destination.mkdir()
    saved = destination / existing
    saved.write_bytes(b"preserve exactly")

    def no_read(*_args, **_kwargs):
        raise AssertionError("occupied destination must be rejected before source reads")

    monkeypatch.setattr(datasets, "load_summaries", no_read)
    with pytest.raises(FileExistsError):
        datasets.freeze_training_data("archive", destination)
    assert saved.read_bytes() == b"preserve exactly"
    assert list(destination.iterdir()) == [saved]


def test_freeze_retry_preserves_completed_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "load_summaries", lambda *_args, **_kwargs: _summaries())
    destination = tmp_path / "freeze"
    datasets.freeze_training_data("archive", destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        datasets.freeze_training_data("archive", destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


@pytest.mark.parametrize("corruption", ["feature_cutoff", "qualifying_group", "race_group"])
def test_invalid_training_preflight_leaves_no_partial_freeze(tmp_path, monkeypatch, corruption):
    summaries = _summaries()
    monkeypatch.setattr(datasets, "load_summaries", lambda *_args, **_kwargs: summaries)
    build = datasets.training_rows_from_summaries

    def invalid_rows(frame):
        rows = build(frame)
        if corruption == "feature_cutoff":
            rows.loc[0, "source_latest_at"] = pd.Timestamp("2030-01-01T00:00Z")
        else:
            target = corruption.removesuffix("_group")
            index = rows.loc[rows.target.eq(target)].index[0]
            rows.loc[index, "target_session_end"] = pd.Timestamp("2030-01-01T00:00Z")
        return rows

    monkeypatch.setattr(datasets, "training_rows_from_summaries", invalid_rows)
    destination = tmp_path / "freeze"
    with pytest.raises(ValueError, match="cutoff|share"):
        datasets.freeze_training_data("archive", destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_artifact_write_cleans_staging_and_preserves_empty_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "load_summaries", lambda *_args, **_kwargs: _summaries())
    destination = tmp_path / "freeze"
    destination.mkdir()
    write = pd.DataFrame.to_parquet

    def failing_training_write(frame, path, *args, **kwargs):
        if path.name == "training.parquet":
            raise OSError("simulated storage failure")
        return write(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_training_write)
    with pytest.raises(OSError, match="storage failure"):
        datasets.freeze_training_data("archive", destination)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


def test_freeze_rejects_even_small_history_round_trip_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "load_summaries", lambda *_args, **_kwargs: _summaries())
    write = pd.DataFrame.to_parquet

    def corrupt_history_write(frame, path, *args, **kwargs):
        if path.name == "history.parquet":
            frame = frame.copy()
            frame.loc[0, "pace_s"] += 0.000001
        return write(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", corrupt_history_write)
    destination = tmp_path / "freeze"
    with pytest.raises(AssertionError, match="pace_s"):
        datasets.freeze_training_data("archive", destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
