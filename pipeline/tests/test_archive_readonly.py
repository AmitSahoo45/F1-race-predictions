"""A training snapshot can be read while the catalog has an active writer."""

import json

import pandas as pd
from f1forecast.archive import load_summaries


def test_readonly_summaries_use_latest_published_version_without_catalog(tmp_path, monkeypatch):
    for suffix, hour, value in [("old", 10, 1), ("new", 11, 2), ("partial.staging", 12, 3)]:
        identifier = f"2026-01-FP1_{suffix}"
        path = tmp_path / "snapshots" / ("." + identifier if "staging" in suffix else identifier)
        path.mkdir(parents=True)
        manifest = {
            "snapshot_id": identifier,
            "session_id": "2026-01-FP1",
            "retrieved_at": f"2026-09-28T{hour}:00:00+00:00",
            "coverage": {"session_complete": True},
        }
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        pd.DataFrame({"driver_id": ["a"], "value": [value]}).to_parquet(path / "summary.parquet")
    monkeypatch.setattr(
        "f1forecast.archive._connect",
        lambda *_: (_ for _ in ()).throw(AssertionError("catalog opened")),
    )
    result = load_summaries(tmp_path, use_catalog=False)
    assert result.value.tolist() == [2]
    assert result.source_snapshot_id.tolist() == ["2026-01-FP1_new"]
    assert result.session_complete.tolist() == [True]
