from datetime import UTC, datetime, timedelta

import pytest
from f1forecast.contracts import Event, Forecast, SiteData
from pydantic import ValidationError


def forecast_payload():
    return {
        "id": "demo-race", "event_id": "demo", "target": "race",
        "kind": "demonstration", "generated_at": "2026-09-01T10:00:00Z",
        "cutoff_at": "2026-09-01T10:30:00Z", "training_cutoff": "2026-08-01T00:00:00Z",
        "model_version": "demo-v1", "model_kind": "baseline", "sample_count": 20000,
        "seed": 42, "coverage": {"available_sessions": ["Q"], "missing_sessions": [],
        "summary": "Synthetic demonstration", "flags": ["synthetic"]},
        "drivers": [{"driver_id": driver, "rank": i + 1, "expected_position": 1.5,
                     "p10": 1, "p90": 2, "win_probability": .5, "podium_probability": 1.,
                     "top10_probability": 1., "position_probabilities": [.5, .5],
                     "explanations": []} for i, driver in enumerate(["alpha", "beta"])],
    }


def test_valid_distribution_round_trips_and_uses_utc():
    result = Forecast.model_validate(forecast_payload())
    assert result.model_dump(mode="json")["cutoff_at"].endswith("Z")
    assert Forecast.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("change", ["sum", "marginal", "rank", "duplicate", "nan", "interval"])
def test_rejects_inconsistent_probability_artifacts(change):
    payload = forecast_payload()
    row = payload["drivers"][0]
    if change == "sum":
        row["position_probabilities"] = [.2, .2]
    elif change == "marginal":
        row["win_probability"] = .8
    elif change == "rank":
        row["rank"] = 2
    elif change == "duplicate":
        row["driver_id"] = "beta"
    elif change == "nan":
        row["expected_position"] = float("nan")
    else:
        row["p10"], row["p90"] = 2, 1
    with pytest.raises(ValidationError):
        Forecast.model_validate(payload)


def test_issued_forecast_cannot_be_generated_after_cutoff_or_trained_in_future():
    payload = forecast_payload()
    payload["kind"] = "issued"
    payload["generated_at"] = "2026-09-01T11:00:00Z"
    with pytest.raises(ValidationError, match="cutoff"):
        Forecast.model_validate(payload)
    payload["generated_at"] = "2026-09-01T10:00:00Z"
    payload["training_cutoff"] = "2026-09-02T00:00:00Z"
    with pytest.raises(ValidationError, match="training"):
        Forecast.model_validate(payload)


def test_naive_dates_and_unsafe_ids_are_rejected():
    payload = forecast_payload()
    payload["generated_at"] = "2026-09-01T10:00:00"
    with pytest.raises(ValidationError):
        Forecast.model_validate(payload)
    payload = forecast_payload()
    payload["id"] = "../overwrite"
    with pytest.raises(ValidationError):
        Forecast.model_validate(payload)


def test_event_cutoff_must_match_target_session_and_unique_entrants():
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)
    event = {"id": "demo", "season": 2026, "round": 1, "name": "Demo", "circuit": "Demo",
             "country": "Example", "timezone": "UTC", "entrants": [],
             "sessions": [{"id": "demo-r", "kind": "R", "start": now, "end": now + timedelta(hours=2)}],
             "targets": [{"target": "race", "session_id": "demo-r", "cutoff_at": now,
                          "state": "scheduled", "reason": None, "forecast_id": None}]}
    with pytest.raises(ValidationError, match="30 minutes"):
        Event.model_validate(event)


def test_demo_forecast_requires_global_disclosure():
    payload = {"generated_at": "2026-09-01T10:00:00Z", "demo_notice": None, "events": [],
               "forecasts": [forecast_payload()], "analyses": [],
               "evaluation": {"status": "pending", "summary": "Not evaluated", "seasons": [],
                              "comparison": [], "limitations": []}}
    with pytest.raises(ValidationError):
        SiteData.model_validate(payload)
