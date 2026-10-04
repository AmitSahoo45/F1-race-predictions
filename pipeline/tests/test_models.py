from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from f1forecast.datasets import prepare_training_data
from f1forecast.evaluation import evaluate_event, promotion_decision
from f1forecast.features import NUMERIC_FEATURES, missing_feature_groups
from f1forecast.modeling import fit_ranker, score_explanations
from f1forecast.probabilities import fit_temperature, normalized_order_nll, sample_orders
from f1forecast.training import (
    attach_results,
    fit_final_model,
    input_dataset_sha256,
    validate_training_rows,
    walk_forward_backtest,
)


def test_input_fingerprint_covers_provenance_and_ignores_dataframe_order():
    rows = pd.DataFrame(
        [
            {
                "event_id": "2024-02",
                "target": "race",
                "driver_id": "b",
                "feature_provenance": "reconstructed",
                "source_latest_at": "2024-04-01T10:00Z",
                "used_session_ids": ("2024-02-FP1",),
                "pace": np.nan,
            },
            {
                "event_id": "2024-01",
                "target": "race",
                "driver_id": "a",
                "feature_provenance": "prospective",
                "source_latest_at": "2024-03-01T10:00Z",
                "used_session_ids": ("2024-01-Q",),
                "pace": 1.25,
            },
        ]
    )
    original = input_dataset_sha256(rows)
    shuffled = rows.sample(frac=1, random_state=7)[list(reversed(rows.columns))]
    assert original == input_dataset_sha256(shuffled)
    changed = rows.copy()
    changed.loc[0, "feature_provenance"] = "prospective"
    assert input_dataset_sha256(changed) != original
    changed = rows.copy()
    changed.loc[0, "source_latest_at"] = "2024-04-01T11:00Z"
    assert input_dataset_sha256(changed) != original


def test_pl_samples_are_reproducible_normalized_and_nested():
    a = sample_orders(["a", "b", "c"], [2.0, 0.0, -2.0], temperature=1.0, seed=23)
    b = sample_orders(["a", "b", "c"], [2.0, 0.0, -2.0], temperature=1.0, seed=23)
    assert np.array_equal(a.position_probabilities, b.position_probabilities)
    assert a.position_probabilities.shape == (3, 3)
    assert np.allclose(a.position_probabilities.sum(axis=1), 1)
    assert np.allclose(a.position_probabilities.sum(axis=0), 1)
    assert np.all(
        (a.win_probabilities <= a.podium_probabilities)
        & (a.podium_probabilities <= a.top_ten_probabilities)
    )
    assert a.win_probabilities[0] > a.win_probabilities[1] > a.win_probabilities[2]


def test_normalized_full_order_loss_matches_direct_likelihood():
    scores = np.array([2.0, 1.0, -1.0])
    probability = (math.exp(2) / sum(math.exp(value) for value in scores)) * (
        math.exp(1) / (math.exp(1) + math.exp(-1))
    )
    assert normalized_order_nll(scores, [1, 2, 3], 1.0) == pytest.approx(
        -math.log(probability) / math.log(6)
    )


def test_temperature_fit_rejects_future_calibration_events():
    events = pd.DataFrame(
        {
            "event_id": ["2024-01", "2024-01", "2025-01", "2025-01"],
            "event_end": ["2024-01-01T00:00:00Z"] * 2 + ["2025-01-01T00:00:00Z"] * 2,
            "driver_id": ["a", "b", "a", "b"],
            "score": [1.0, 0.0, 0.0, 1.0],
            "position": [1, 2, 2, 1],
            "prediction_provenance": ["chronological_oof"] * 4,
            "feature_provenance": ["reconstructed"] * 4,
        }
    )
    with pytest.raises(ValueError, match="cutoff"):
        fit_temperature(events, training_cutoff="2025-01-01T00:00:00Z")
    assert 0.25 <= fit_temperature(events.iloc[:2], training_cutoff="2025-01-01T00:00:00Z") <= 4.0
    bad = events.iloc[:2].copy()
    bad["prediction_provenance"] = "in_sample"
    with pytest.raises(ValueError, match="provenance"):
        fit_temperature(bad, training_cutoff="2025-01-01T00:00:00Z")


def test_pairlogit_ranker_prefers_better_feature_on_training_events():
    rows = []
    for event in range(1, 7):
        for driver, rank in [("a", 1), ("b", 2), ("c", 3)]:
            rows.append(
                {
                    "event_id": f"2024-{event:02d}",
                    "driver_id": driver,
                    "recent_qualifying_rank": rank / 3,
                    "position": rank,
                }
            )
    frame = pd.DataFrame(rows)
    model = fit_ranker(frame, feature_columns=["recent_qualifying_rank"], iterations=80)
    score = model.predict(frame.iloc[:3][["recent_qualifying_rank"]])
    assert list(np.argsort(-score)) == [0, 1, 2]
    method, explanations = score_explanations(
        model,
        frame.iloc[:3],
        feature_columns=["recent_qualifying_rank"],
        reference_medians={"recent_qualifying_rank": 2 / 3},
    )
    assert method in {"SHAP score contribution", "median-reference score sensitivity"}
    assert all(item[0]["group"] == "recent qualifying form" for item in explanations)


def test_training_rejects_unfinished_or_leaky_events():
    frame = pd.DataFrame(
        {
            "event_id": ["2024-01", "2024-01"],
            "cutoff": ["2024-03-02T11:30Z"] * 2,
            "source_latest_at": ["2024-03-02T11:31Z", "2024-03-02T10:00Z"],
            "target_session_end": ["2024-03-02T13:00Z"] * 2,
            "weekend_complete_at": ["2024-03-03T16:00Z"] * 2,
            "feature_provenance": ["prospective"] * 2,
        }
    )
    with pytest.raises(ValueError, match="source"):
        validate_training_rows(frame, "2024-04-01T00:00Z")
    frame.loc[0, "source_latest_at"] = "2024-03-02T10:00Z"
    with pytest.raises(ValueError, match="complete"):
        validate_training_rows(frame, "2024-03-02T12:00Z")
    with pytest.raises(ValueError, match="weekend"):
        validate_training_rows(frame, "2024-03-03T15:00Z")


def test_full_order_evaluation_excludes_missing_classifications_without_fabricating_places():
    probabilities = sample_orders(["a", "b", "c"], [2.0, 1.0, 0.0], seed=7)
    result = evaluate_event(
        probabilities,
        {"a": 1, "b": 2, "c": None},
        statuses={"a": "Finished", "b": "Retired", "c": "DNS"},
    )
    assert not result.full_order
    assert result.excluded_drivers == {"c": "DNS"}
    assert result.position_mae is None
    assert result.normalized_nll is None
    disqualified = evaluate_event(
        probabilities, {"a": 1, "b": 2, "c": 3}, statuses={"c": "Disqualified"}
    )
    assert not disqualified.full_order
    assert disqualified.excluded_drivers == {"c": "Disqualified"}


def test_results_join_requires_recorded_gp_completion_for_qualifying_training():
    features = pd.DataFrame(
        {"event_id": ["2024-01"], "target_session_id": ["2024-01-Q"], "driver_id": ["a"]}
    )
    results = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "session_id": "2024-01-Q",
                "session_type": "Q",
                "driver_id": "a",
                "session_end": "2024-03-01T12:00Z",
                "position": 1,
                "status": "Finished",
            },
            {
                "event_id": "2024-01",
                "session_id": "2024-01-R",
                "session_type": "R",
                "driver_id": "a",
                "session_end": "2024-03-03T16:00Z",
                "position": 1,
                "status": "Finished",
            },
        ]
    )
    labelled = attach_results(features, results)
    assert labelled.loc[0, "weekend_complete_at"] == "2024-03-03T16:00Z"
    assert labelled.loc[0, "position"] == 1
    before_gp = attach_results(features, results.iloc[:1])
    assert pd.isna(before_gp.loc[0, "weekend_complete_at"])


@pytest.mark.parametrize("target,session_type", [("qualifying", "Q"), ("race", "R")])
def test_absent_classification_keeps_session_timing_and_is_excluded(target, session_type):
    features = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "season": 2024,
                "target": target,
                "target_session_id": f"2024-01-{session_type}",
                "driver_id": driver,
                "cutoff": "2024-03-02T11:30Z",
                "source_latest_at": "2024-03-01T10:00Z",
                "feature_provenance": "reconstructed",
                "baseline_score": 1 - index / 3,
                **{name: float(index) for name in NUMERIC_FEATURES},
            }
            for index, driver in enumerate(("a", "b", "c"), 1)
        ]
    )
    summaries = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "session_id": f"2024-01-{kind}",
                "session_type": kind,
                "driver_id": driver,
                "session_end": end,
                "position": position,
                "status": "Retired" if driver == "b" else "Finished",
                "session_complete": True,
            }
            for kind, end in (("Q", "2024-03-02T13:00Z"), ("R", "2024-03-03T16:00Z"))
            for position, driver in enumerate(("a", "b"), 1)
        ]
    )
    labelled = attach_results(features, summaries)
    expected_end = "2024-03-02T13:00Z" if target == "qualifying" else "2024-03-03T16:00Z"
    assert labelled["target_session_end"].eq(expected_end).all()
    assert labelled["driver_id"].tolist() == ["a", "b", "c"]
    assert pd.isna(labelled.loc[2, "position"])
    assert pd.isna(labelled.loc[2, "status"])
    assert labelled.loc[1, "status"] == "Retired"
    event = walk_forward_backtest(labelled, target, n_samples=100)["events"][0]
    assert not event["full_order"]
    assert set(event["excluded_drivers"]) == {"c"}
    assert event["required_inputs_available"]
    with pytest.raises(ValueError, match="missing classifications"):
        validate_training_rows(labelled, "2024-04-01T00:00Z")


def test_results_join_rejects_conflicting_finalized_session_ends():
    features = pd.DataFrame(
        {"event_id": ["2024-01"], "target_session_id": ["2024-01-R"], "driver_id": ["a"]}
    )
    summaries = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "session_id": "2024-01-R",
                "session_type": "R",
                "driver_id": driver,
                "session_end": end,
                "position": position,
                "status": "Finished",
            }
            for driver, position, end in (
                ("a", 1, "2024-03-03T16:00Z"),
                ("b", 2, "2024-03-03T17:00Z"),
            )
        ]
    )
    with pytest.raises(ValueError, match="session end"):
        attach_results(features, summaries)


def test_unpredicted_last_place_entrant_cannot_be_a_complete_training_or_scored_order(tmp_path):
    features = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "season": 2024,
                "target": "race",
                "target_session_id": "2024-01-R",
                "driver_id": driver,
                "cutoff": "2024-03-03T13:30Z",
                "source_latest_at": "2024-03-02T13:00Z",
                "feature_provenance": "reconstructed",
                "baseline_score": 1 - index / 2,
                **{name: float(index) for name in NUMERIC_FEATURES},
            }
            for index, driver in enumerate(("a", "b"), 1)
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "session_id": "2024-01-R",
                "session_type": "R",
                "driver_id": driver,
                "session_end": "2024-03-03T16:00Z",
                "position": position,
                "status": "Retired" if driver == "c" else "Finished",
            }
            for position, driver in enumerate(("a", "b", "c"), 1)
        ]
    )
    labelled = attach_results(features, actual)
    assert labelled["driver_id"].tolist() == ["a", "b"]
    # Preserve exactly the original forecast field through the CLI data boundary.
    path = tmp_path / "labelled.parquet"
    labelled.to_parquet(path, index=False)
    labelled = pd.read_parquet(path)
    report = walk_forward_backtest(labelled, "race", iterations=5, n_samples=100)
    event = report["events"][0]
    assert not event["full_order"]
    assert event["unexpected_actual_entrants"] == ["c"]
    assert event["predicted_field_count"] == 2
    assert event["actual_field_count"] == 3
    assert event["required_inputs_available"]
    assert event["excluded_drivers"] == {"c": "actual entrant absent from pre-session roster"}
    assert report["periods"]["development_2022_2024"]["baseline"]["evaluated_events"] == 0
    with pytest.raises(ValueError, match="insufficient complete prior events"):
        fit_final_model(
            labelled,
            "race",
            training_cutoff="2024-04-01T00:00Z",
            output_dir=tmp_path / "models",
            min_train_events=1,
            iterations=5,
        )
    labelled.loc[0, "qualifying_position"] = np.nan
    unavailable = walk_forward_backtest(labelled, "race", iterations=5, n_samples=100)["events"][0]
    assert not unavailable["required_inputs_available"]
    assert not unavailable["full_order"]
    assert unavailable["required_input_unavailability_reason"] == (
        "required qualifying positions are unavailable for one or more entrants"
    )


def test_unfinished_qualifying_and_race_cannot_become_training_labels_or_gp_completion():
    features = pd.DataFrame(
        [
            {"event_id": "2024-01", "target_session_id": "2024-01-Q", "driver_id": "a"},
            {"event_id": "2024-01", "target_session_id": "2024-01-R", "driver_id": "a"},
        ]
    )
    summaries = pd.DataFrame(
        [
            {
                "event_id": "2024-01",
                "session_id": "2024-01-Q",
                "session_type": "Q",
                "driver_id": "a",
                "session_end": "2024-03-02T13:00Z",
                "position": 1,
                "status": "Finished",
                "session_complete": False,
            },
            {
                "event_id": "2024-01",
                "session_id": "2024-01-R",
                "session_type": "R",
                "driver_id": "a",
                "session_end": "2024-03-03T16:00Z",
                "position": 1,
                "status": "Finished",
                "session_complete": False,
            },
        ]
    )
    both_partial = attach_results(features, summaries)
    assert both_partial["position"].isna().all()
    assert both_partial["target_session_end"].isna().all()
    assert both_partial["weekend_complete_at"].isna().all()

    summaries.loc[0, "session_complete"] = True
    q_final_r_partial = attach_results(features, summaries)
    assert q_final_r_partial.loc[0, "position"] == 1
    assert pd.isna(q_final_r_partial.loc[0, "weekend_complete_at"])
    assert pd.isna(q_final_r_partial.loc[1, "position"])

    summaries.loc[1, "session_complete"] = True
    fully_final = attach_results(features, summaries)
    assert fully_final["position"].tolist() == [1, 1]
    assert fully_final["weekend_complete_at"].eq("2024-03-03T16:00Z").all()


def test_historical_prepare_never_uses_partial_q_as_race_roster(tmp_path, monkeypatch):
    rows = []
    for session, start, end, complete in (
        ("FP1", "2024-03-01T09:00Z", "2024-03-01T10:00Z", True),
        ("Q", "2024-03-02T12:00Z", "2024-03-02T13:00Z", False),
        ("R", "2024-03-03T14:00Z", "2024-03-03T16:00Z", True),
    ):
        for driver, position, pace in (("a", 1, 90.0), ("b", 2, 91.0)):
            rows.append(
                {
                    "event_id": "2024-01",
                    "season": 2024,
                    "circuit_id": "bahrain",
                    "driver_id": driver,
                    "team_id": driver,
                    "session_id": f"2024-01-{session}",
                    "session_type": session,
                    "session_start": start,
                    "session_end": end,
                    "observed_at": end,
                    "available_at": "2025-01-01T00:00Z",
                    "provenance": "retrospective",
                    "session_complete": complete,
                    "position": position if session != "FP1" else None,
                    "status": "Finished" if session != "FP1" else None,
                    "pace_s": pace if session == "FP1" else None,
                    "qualifying_gap": (position - 1) * 0.2 if session == "Q" else None,
                }
            )
    summaries = pd.DataFrame(rows)
    monkeypatch.setattr("f1forecast.datasets.load_summaries", lambda _archive, **_kwargs: summaries)
    with pytest.raises(ValueError, match="no target"):
        prepare_training_data(tmp_path / "archive", tmp_path / "partial.parquet")

    summaries.loc[summaries["session_type"].eq("Q"), "session_complete"] = True
    summaries.loc[summaries["session_type"].eq("R"), "session_complete"] = False
    partial_r = prepare_training_data(tmp_path / "archive", tmp_path / "partial-r.parquet")
    assert partial_r["target"].eq("qualifying").all()
    assert set(partial_r["driver_id"]) == {"a", "b"}
    assert partial_r["position"].tolist() == [1, 2]
    assert partial_r["weekend_complete_at"].isna().all()


def test_promotion_requires_probability_gain_without_position_regression():
    baseline = {"probability_loss": 0.30, "position_mae": 1.0}
    assert promotion_decision({"probability_loss": 0.25, "position_mae": 0.9}, baseline)
    assert not promotion_decision({"probability_loss": 0.25, "position_mae": 1.1}, baseline)
    assert not promotion_decision({"probability_loss": 0.31, "position_mae": 0.9}, baseline)


def test_walk_forward_keeps_2025_locked_and_reports_2026_separately(tmp_path, monkeypatch):
    rows = []
    for event, season in enumerate([2022, 2022, 2023, 2024, 2025, 2026], 1):
        for driver, position in [("a", 1), ("b", 2)]:
            rows.append(
                {
                    "event_id": f"{season}-{event:02d}",
                    "season": season,
                    "target": "qualifying",
                    "driver_id": driver,
                    "position": position,
                    "status": "Finished",
                    "recent_qualifying_rank": position / 2,
                    "practice_pace_gap_s": float(position),
                    "baseline_score": 1 - position / 2,
                    "cutoff": f"{season}-03-{event:02d}T11:30:00Z",
                    "target_session_end": f"{season}-03-{event:02d}T13:00:00Z",
                    "weekend_complete_at": f"{season}-03-{event:02d}T16:00:00Z",
                    "source_latest_at": f"{season}-03-{event:02d}T10:00:00Z",
                    "feature_provenance": "reconstructed",
                    "used_session_ids": (f"{season}-{event:02d}-FP1",),
                    "used_snapshot_ids": (f"{season}-{event:02d}-FP1-snapshot",),
                }
            )
    frame = pd.DataFrame(rows)
    for feature in NUMERIC_FEATURES:
        if feature not in frame:
            frame[feature] = np.nan
    # CLI evaluation reads the prepared Parquet table, whose array columns
    # deserialize as numpy arrays rather than their original Python tuples.
    prepared = tmp_path / "prepared.parquet"
    frame.to_parquet(prepared, index=False)
    frame = pd.read_parquet(prepared)
    from f1forecast import training

    original_fit = training.fit_ranker
    fit_count = 0

    def counted_fit(*args, **kwargs):
        nonlocal fit_count
        fit_count += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(training, "fit_ranker", counted_fit)
    report = walk_forward_backtest(
        frame, "qualifying", iterations=12, min_train_events=2, n_samples=100, seed=8
    )
    assert report["evaluation_schema_version"] == "2"
    assert fit_count == 4  # One chronological fit for each eligible held event.
    locked = next(item for item in report["events"] if item["season"] == 2025)
    separate = next(item for item in report["events"] if item["season"] == 2026)
    assert locked["training_events"] == 4
    assert locked["calibration_oof_events"] == 2
    assert separate["training_events"] == 5
    assert locked["cutoff"] == "2025-03-05T11:30:00+00:00"
    assert locked["target_session_end"] == "2025-03-05T13:00:00+00:00"
    assert locked["training_cutoff"] == "2024-03-04T16:00:00+00:00"
    assert locked["training_latest_completed_at"] == "2024-03-04T16:00:00+00:00"
    assert pd.Timestamp(locked["training_cutoff"]) < pd.Timestamp(locked["cutoff"])
    assert locked["model_version"].startswith("qualifying-fold-")
    assert locked["source_snapshot_ids"] == ["2025-05-FP1-snapshot"]
    assert locked["used_session_ids"] == ["2025-05-FP1"]
    assert locked["required_inputs_available"]
    assert locked["required_input_unavailability_reason"] is None
    assert report["baseline_selection_provenance"] == "posthoc development 2022-2024"
    for forecast_name in ("baseline_forecast", "learned_forecast"):
        forecast = locked[forecast_name]
        assert set(forecast["explanations"]) == {"a", "b"}
        assert all(forecast["explanations"].values())
        assert forecast["explanation_method"]
        assert forecast["driver_ids"] == ["a", "b"]
        assert len(forecast["scores"]) == 2
        assert forecast["temperature"] > 0
        assert np.allclose(np.asarray(forecast["position_probabilities"]).sum(axis=1), 1)
        assert np.allclose(
            [probabilities[0] for probabilities in forecast["position_probabilities"]],
            forecast["win_probabilities"],
        )
    assert json.loads(json.dumps(report))["events"]  # Exportable without custom encoders.
    assert report["periods"]["locked_2025"]["learned"]["evaluated_events"] == 1
    assert report["input_dataset_sha256"] == input_dataset_sha256(frame)
    pattern = missing_feature_groups(frame.loc[frame["season"].eq(2025)], "qualifying")
    assert locked["missing_pattern"] == pattern
    evidence = next(item for item in report["missing_case_evidence"] if item["pattern"] == pattern)
    assert evidence["baseline_evaluated_event_count"] == 6
    assert evidence["learned_evaluated_event_count"] == 4
    assert evidence["paired_locked_2025"]["baseline"]["evaluated_events"] == 1
    artifact = fit_final_model(
        frame,
        "qualifying",
        training_cutoff="2027-01-01T00:00:00Z",
        output_dir=tmp_path,
        iterations=12,
        min_train_events=2,
        seed=8,
    )
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["input_dataset_sha256"] == report["input_dataset_sha256"]
    assert metadata["baseline_name"] == report["baseline_name"]
    assert metadata["baseline_temperature"] == pytest.approx(report["baseline_temperature"])
    # A finalized Q classification does not make its Grand Prix weekend complete.
    unfinished = frame.copy()
    unfinished.loc[unfinished["season"].eq(2025), "weekend_complete_at"] = pd.NaT
    incomplete_report = walk_forward_backtest(
        unfinished, "qualifying", iterations=12, min_train_events=2, n_samples=100, seed=8
    )
    assert incomplete_report["periods"]["locked_2025"]["baseline"]["evaluated_events"] == 0
    assert incomplete_report["periods"]["locked_2025"]["learned"]["evaluated_events"] == 0
    assert not incomplete_report["promotion_passed"]
    future_revised = frame.copy()
    future_revised.loc[future_revised["season"].eq(2026), "position"] = [2, 1]
    revised_report = walk_forward_backtest(
        future_revised, "qualifying", iterations=12, min_train_events=2, n_samples=100, seed=8
    )
    revised_locked = next(item for item in revised_report["events"] if item["season"] == 2025)
    assert revised_report["input_dataset_sha256"] != report["input_dataset_sha256"]
    assert revised_locked["model_version"] == locked["model_version"]
    assert revised_locked["training_dataset_sha256"] == locked["training_dataset_sha256"]
    assert revised_locked["learned_forecast"] == locked["learned_forecast"]
    unavailable = frame.copy()
    unavailable.loc[unavailable["season"].isin([2024, 2025]), "practice_pace_gap_s"] = np.nan
    unavailable_report = walk_forward_backtest(
        unavailable, "qualifying", iterations=12, min_train_events=2, n_samples=100, seed=8
    )
    unavailable_locked = next(
        item for item in unavailable_report["events"] if item["season"] == 2025
    )
    assert not unavailable_locked["required_inputs_available"]
    assert (
        unavailable_locked["evaluation_exclusion_reason"] == "required practice pace is unavailable"
    )
    assert unavailable_locked["learned_forecast"] is not None  # Diagnostic only.
    assert unavailable_report["periods"]["locked_2025"]["baseline"]["evaluated_events"] == 0
    assert unavailable_report["periods"]["locked_2025"]["learned"]["evaluated_events"] == 0
    assert not unavailable_report["promotion_passed"]
    unavailable_2026 = next(item for item in unavailable_report["events"] if item["season"] == 2026)
    assert unavailable_2026["calibration_oof_events"] == 1
    unavailable_artifact = fit_final_model(
        unavailable,
        "qualifying",
        training_cutoff="2027-01-01T00:00:00Z",
        output_dir=tmp_path / "unavailable",
        iterations=12,
        min_train_events=2,
        seed=8,
    )
    unavailable_metadata = json.loads(unavailable_artifact.metadata_path.read_text())
    assert unavailable_metadata["calibration_oof_events"] == 1
    assert (
        unavailable_metadata["baseline_temperature"] == unavailable_report["baseline_temperature"]
    )
    frame.loc[frame["season"].eq(2026), "source_latest_at"] = "2026-03-06T12:00:00Z"
    with pytest.raises(ValueError, match="source"):
        walk_forward_backtest(
            frame, "qualifying", iterations=12, min_train_events=2, n_samples=100, seed=8
        )
