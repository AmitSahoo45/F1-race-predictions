"""Offline contract tests; fixtures are never published as real evidence."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from f1forecast.contracts import Analysis, EventTelemetry
from f1forecast.historical import forecast_from_record, historical_model_choice, publish_history
from f1forecast.publication import verify_archive
from test_operations import example_event


def _report_and_record():
    report = {
        "evaluation_schema_version": "2",
        "target": "race",
        "input_dataset_sha256": "a" * 64,
        "training_config": {"iterations": 300, "random_seed": 42},
        "baseline_name": "qualifying_order",
    }
    record = {
        "required_inputs_available": True,
        "unexpected_actual_entrants": [],
        "event_id": "2026-01",
        "cutoff": "2026-10-04T11:30:00Z",
        "training_cutoff": "2026-09-01T00:00:00Z",
        "model_version": "race-fold-example",
        "source_snapshot_ids": ["2026-01-Q_source-version"],
        "used_session_ids": ["2026-01-Q"],
        "missing_pattern": [],
        "baseline_forecast": {
            "driver_ids": ["a", "b"],
            "scores": [1.0, 0.0],
            "temperature": 1.0,
            "position_probabilities": [[0.7, 0.3], [0.3, 0.7]],
            "expected_positions": [1.3, 1.7],
            "win_probabilities": [0.7, 0.3],
            "podium_probabilities": [1.0, 1.0],
            "top_ten_probabilities": [1.0, 1.0],
        },
    }
    return report, record


def test_historical_forecast_keeps_real_generation_time_and_reconstruction_label():
    report, record = _report_and_record()
    now = datetime(2026, 11, 1, tzinfo=UTC)
    result = forecast_from_record(example_event(), record, report, model_choice="baseline", now=now)
    assert result.kind == "reconstructed"
    assert result.generated_at == now > result.cutoff_at
    assert result.training_cutoff < result.cutoff_at
    assert result.drivers[0].driver_id == "a"
    assert result.drivers[0].p10 == 1 and result.drivers[0].p90 == 2
    assert result.input_snapshot_ids == ["2026-01-Q_source-version"]
    assert "retrospective-archive" in result.coverage.flags


def test_historical_export_rejects_wrong_event_or_missing_fold_instead_of_refitting():
    report, record = _report_and_record()
    with pytest.raises(ValueError, match="fold"):
        forecast_from_record(example_event(), record, report, model_choice="learned")
    record["event_id"] = "2026-99"
    with pytest.raises(ValueError, match="event"):
        forecast_from_record(example_event(), record, report, model_choice="baseline")


def test_historical_explanations_follow_saved_driver_ids_through_rank_sorting():
    report, record = _report_and_record()
    block = record["baseline_forecast"]
    block["explanations"] = {
        "b": [{"group": "qualifying result", "contribution": -0.4}],
        "a": [{"group": "qualifying result", "contribution": 0.4}],
    }
    block["explanation_method"] = "baseline ranking score"
    result = forecast_from_record(example_event(), record, report, model_choice="baseline")
    assert result.drivers[0].explanations[0].contribution == 0.4
    assert result.drivers[1].explanations[0].contribution == -0.4
    assert result.drivers[0].explanations[0].group == "qualifying result"
    del block["explanations"]["b"]
    with pytest.raises(ValueError, match="explanation"):
        forecast_from_record(example_event(), record, report, model_choice="baseline")


def test_historical_identity_changes_when_recipe_changes():
    report, record = _report_and_record()
    first = forecast_from_record(example_event(), record, report, model_choice="baseline")
    report["training_config"]["iterations"] = 500
    second = forecast_from_record(example_event(), record, report, model_choice="baseline")
    assert first.id != second.id


def _publishable(report, record):
    report.update(
        events=[record],
        promotion_passed=False,
        periods={
            "locked_2025": {
                "baseline": {"evaluated_events": 0},
                "learned": {"evaluated_events": 0},
            },
            "separate_2026": {
                "baseline": {
                    "evaluated_events": 1,
                    "probability_loss": 0.5,
                    "position_mae": 0.3,
                    "win_brier": 0.09,
                    "podium_brier": 0.0,
                    "top_ten_brier": 0.0,
                },
                "learned": {"evaluated_events": 0},
            },
        },
    )
    return report


def _offline_calendar(monkeypatch):
    race = {
        "season": "2026",
        "round": "1",
        "raceName": "Synthetic test event",
        "Circuit": {"circuitId": "test", "Location": {"country": "Test"}},
        "Qualifying": {"date": "2026-10-03", "time": "12:00:00Z"},
        "date": "2026-10-04",
        "time": "12:00:00Z",
    }
    monkeypatch.setattr("f1forecast.providers.JolpicaProvider.fetch_races", lambda *_: [race])
    monkeypatch.setattr("f1forecast.archive.load_event_snapshots", lambda *_a, **_k: [])
    monkeypatch.setattr("f1forecast.historical._drivers", lambda _: example_event().entrants)


def test_history_publishes_saved_evidence_and_preserves_immutable_artifacts_on_repeat(
    tmp_path, monkeypatch
):
    report = _publishable(*_report_and_record())
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _offline_calendar(monkeypatch)
    now = datetime(2026, 11, 1, tzinfo=UTC)
    destination = tmp_path / "site-data"
    first = publish_history(
        [str(report_path)], tmp_path / "archive", destination, years=[2026], now=now
    )
    assert first.demo_notice is None
    assert first.evaluation.status == "evaluated"
    assert first.events[0].targets[0].state == "unavailable"
    assert first.events[0].targets[1].state == "reconstructed"
    assert first.forecasts[0].generated_at == now
    assert first.forecasts[0].drivers[0].win_probability == 0.7
    immutable = destination / "forecasts" / "reconstructed" / f"{first.forecasts[0].id}.json"
    saved = immutable.read_bytes()
    second = publish_history(
        [str(report_path)],
        tmp_path / "archive",
        destination,
        years=[2026],
        now=datetime(2026, 11, 2, tzinfo=UTC),
    )
    assert second.forecasts == first.forecasts
    assert immutable.read_bytes() == saved

    assert verify_archive(destination / "forecasts") == 1
    assert (
        json.loads((destination / "evaluation-reports/race.json").read_text(encoding="utf-8"))
        == report
    )

    # Rejected replacement evidence must not leave new data with stale accuracy.
    saved_site = (destination / "site.json").read_bytes()
    empty_report = json.loads(json.dumps(report))
    empty_report["periods"]["separate_2026"]["baseline"] = {"evaluated_events": 0}
    report_path.write_text(json.dumps(empty_report), encoding="utf-8")
    with pytest.raises(ValueError, match="no evaluated"):
        publish_history(
            [str(report_path)], tmp_path / "archive", destination, years=[2026], now=now
        )
    assert (destination / "site.json").read_bytes() == saved_site
    assert immutable.read_bytes() == saved

    # A report must not silently move the event schedule to fit its own cutoff.
    report["events"][0]["cutoff"] = "2026-10-04T10:00:00Z"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule"):
        publish_history(
            [str(report_path)], tmp_path / "archive", tmp_path / "bad-site", years=[2026], now=now
        )
    assert immutable.read_bytes() == saved


@pytest.mark.parametrize(
    "fields",
    [
        {"unexpected_actual_entrants": ["missing-driver"]},
        {
            "required_inputs_available": False,
            "required_input_unavailability_reason": "required qualifying positions unavailable",
        },
        {"missing_pattern": None},
    ],
)
def test_historical_forecast_rejects_incomplete_roster_or_missing_required_inputs(fields):
    report, record = _report_and_record()
    record.update(fields)
    with pytest.raises(ValueError, match="unavailable"):
        forecast_from_record(example_event(), record, report, model_choice="baseline")


def test_historical_forecast_requires_explicit_current_availability_evidence():
    report, record = _report_and_record()
    del record["required_inputs_available"]
    record["missing_pattern"] = ["missing-qualifying-result"]
    with pytest.raises(ValueError, match="unavailable"):
        forecast_from_record(example_event(), record, report, model_choice="baseline")


def _case(pattern, *, learned_mae=0.9):
    def summary(loss, mae):
        return {
            "probability_loss": loss,
            "position_mae": mae,
            "win_brier": 0.1,
            "podium_brier": 0.1,
            "top_ten_brier": 0.1,
            "evaluated_events": 1,
        }

    learned, baseline = summary(0.2, learned_mae), summary(0.3, 1.0)
    return {
        "pattern": pattern,
        "baseline": baseline,
        "learned": learned,
        "paired_locked_2025": {"baseline": baseline, "learned": learned},
    }


def test_historical_model_choice_applies_the_live_missing_case_rule():
    report, record = _report_and_record()
    record["learned_forecast"] = dict(record["baseline_forecast"])
    report.update(
        promotion_passed=True,
        missing_case_evidence=[
            _case(["missing-telemetry"]),
            _case(["missing-tyre-context"], learned_mae=1.1),
        ],
    )
    for pattern, expected in (
        ([], "learned"),
        (["missing-telemetry"], "learned"),
        (["missing-tyre-context"], "baseline"),
        (["missing-observed-conditions"], None),
    ):
        record["missing_pattern"] = pattern
        assert historical_model_choice(record, report) == expected
    report["promotion_passed"] = False
    record["missing_pattern"] = ["missing-telemetry"]
    assert historical_model_choice(record, report) == "baseline"


def test_historical_reconstruction_rejects_a_case_its_model_did_not_validate():
    report, record = _report_and_record()
    record["learned_forecast"] = dict(record["baseline_forecast"])
    report["missing_case_evidence"] = [_case(["missing-tyre-context"], learned_mae=1.1)]
    record["missing_pattern"] = ["missing-tyre-context"]
    with pytest.raises(ValueError, match="missing-data"):
        forecast_from_record(example_event(), record, report, model_choice="learned")
    result = forecast_from_record(example_event(), record, report, model_choice="baseline")
    assert result.coverage.flags == ["retrospective-archive", "missing-tyre-context"]
    record["missing_pattern"] = ["missing-observed-conditions"]
    with pytest.raises(ValueError, match="missing-data"):
        forecast_from_record(example_event(), record, report, model_choice="baseline")


def test_history_applies_archived_times_that_move_a_session_past_its_calendar_slot(
    tmp_path, monkeypatch
):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_publishable(*_report_and_record())), encoding="utf-8")
    _offline_calendar(monkeypatch)
    moved = {
        "session_id": "2026-01-Q",
        "session_start": "2026-10-03T13:00:00+00:00",
        "session_end": "2026-10-03T14:00:00+00:00",
    }
    monkeypatch.setattr(
        "f1forecast.archive.load_event_snapshots", lambda *_a, **_k: [(moved, pd.DataFrame(), {})]
    )
    monkeypatch.setattr(
        "f1forecast.analysis_export.export_analysis",
        lambda event_id, *_a, **_k: (
            Analysis(
                event_id=event_id,
                kind="observed",
                source="Synthetic archive",
                circuit_points=[],
                drivers=[],
                actuals=[],
            ),
            EventTelemetry(event_id=event_id, traces=[]),
        ),
    )
    site = publish_history(
        [str(report_path)],
        tmp_path / "archive",
        tmp_path / "site-data",
        years=[2026],
        now=datetime(2026, 11, 1, tzinfo=UTC),
    )
    qualifying = next(s for s in site.events[0].sessions if s.id == "2026-01-Q")
    assert (qualifying.start, qualifying.end) == (
        datetime(2026, 10, 3, 13, tzinfo=UTC),
        datetime(2026, 10, 3, 14, tzinfo=UTC),
    )
    assert (tmp_path / "site-data" / "telemetry" / "2026-01.json").exists()


def test_history_marks_an_unvalidated_case_unavailable_before_writing(tmp_path, monkeypatch):
    report, record = _report_and_record()
    record["missing_pattern"] = ["missing-observed-conditions"]
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_publishable(report, record)), encoding="utf-8")
    _offline_calendar(monkeypatch)
    destination = tmp_path / "site-data"
    site = publish_history(
        [str(report_path)],
        tmp_path / "archive",
        destination,
        years=[2026],
        now=datetime(2026, 11, 1, tzinfo=UTC),
    )
    race = next(t for t in site.events[0].targets if t.target == "race")
    assert race.state == "unavailable"
    assert "missing-observed-conditions" in race.reason
    assert site.forecasts == []
    assert not list((destination / "forecasts").rglob("*.json"))
