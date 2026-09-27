"""A real CatBoost fit through registry, forecast schema and immutable publication.

The tiny synthetic dataset proves integration, not F1 accuracy.
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from f1forecast.features import NUMERIC_FEATURES
from f1forecast.operations import make_forecast
from f1forecast.publication import publish_forecast, verify_archive
from f1forecast.registry import approve_model
from f1forecast.training import fit_final_model, walk_forward_backtest
from test_operations import example_event


def small_training_set():
    rows = []
    for i, season in enumerate([2022, 2022, 2023, 2024, 2025], 1):
        for j, driver in enumerate(["a", "b"], 1):
            row = {feature: float(j) for feature in NUMERIC_FEATURES}
            row.update({"event_id": f"{season}-{i:02d}", "driver_id": driver,
                "season": season, "target": "race", "position": j, "status": "Finished",
                "cutoff": f"{season}-0{i}-02T10:30:00Z",
                "source_latest_at": f"{season}-0{i}-01T12:00:00Z",
                "target_session_end": f"{season}-0{i}-02T13:00:00Z",
                "weekend_complete_at": f"{season}-0{i}-02T13:00:00Z",
                "feature_provenance": "reconstructed", "qualifying_position": j,
                "recent_race_rank": (j - 1), "recent_qualifying_rank": (j - 1),
                "baseline_score": 1 - j / 2, "used_session_ids": ("2026-01-Q",)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_registry_to_issued_forecast_and_hashes(tmp_path):
    rows = small_training_set()
    report = walk_forward_backtest(rows, "race", iterations=5, min_train_events=2)
    artifact = fit_final_model(rows, "race", training_cutoff="2026-01-01T00:00:00Z",
                               output_dir=tmp_path / "training", iterations=5, min_train_events=2)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report))
    history_path = tmp_path / "history.parquet"
    rows.to_parquet(history_path)
    registry = tmp_path / "models" / "registry.json"
    record = approve_model(artifact.metadata_path, report_path, history_path, registry)
    event = example_event()
    now = datetime(2026, 10, 4, 11, 15, tzinfo=UTC)
    prediction = make_forecast(event, event.targets[0], rows.iloc[-2:], record, registry.parent, now=now)
    assert prediction.kind == "issued"
    assert prediction.model_kind == "baseline"
    assert np.isclose(sum(d.win_probability for d in prediction.drivers), 1)
    publish_forecast(prediction, tmp_path / "forecasts", now=now)
    assert verify_archive(tmp_path / "forecasts") == 1
    # A different dataset must not borrow the same recipe's evaluation evidence.
    metadata = json.loads(artifact.metadata_path.read_text())
    original = metadata["input_dataset_sha256"]
    metadata["input_dataset_sha256"] = "f" * 64
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dataset"):
        approve_model(artifact.metadata_path, report_path, history_path, tmp_path / "different-data" / "registry.json")
    metadata["input_dataset_sha256"] = original
    metadata["baseline_temperature"] *= 2
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="baseline"):
        approve_model(artifact.metadata_path, report_path, history_path, tmp_path / "different-baseline" / "registry.json")
    metadata["baseline_temperature"] /= 2
    artifact.metadata_path.write_text(json.dumps(metadata))
    # A different recipe must not borrow the first recipe's evaluation.
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["training_config"]["iterations"] += 1
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="recipe"):
        approve_model(artifact.metadata_path, report_path, history_path, tmp_path / "other" / "registry.json")
