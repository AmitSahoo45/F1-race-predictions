"""Synthetic, offline rehearsals of weekend publication and approval gates."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from f1forecast.calendar import event_from_race, set_entrants
from f1forecast.contracts import SiteData
from f1forecast.operations import has_weekend_work, run_tick, target_rows
from f1forecast.publication import publish_site
from f1forecast.registry import approve_model, validated_patterns

Q_START = datetime(2026, 10, 3, 12, tzinfo=UTC)
R_START = datetime(2026, 10, 4, 12, tzinfo=UTC)


def _driver(identifier: str) -> dict:
    return {
        "id": identifier,
        "code": identifier.upper() * 3,
        "name": identifier.upper(),
        "team": "Test Team",
        "color": "#abcdef",
    }


def _event(*, q_issued: bool = False) -> dict:
    event = {
        "id": "2026-01",
        "season": 2026,
        "round": 1,
        "name": "Synthetic Grand Prix",
        "circuit": "test_circuit",
        "country": "Testland",
        "timezone": "UTC",
        "sessions": [
            {
                "id": "2026-01-FP1",
                "kind": "FP1",
                "start": Q_START - timedelta(days=1),
                "end": Q_START - timedelta(days=1, hours=-1),
            },
            {"id": "2026-01-Q", "kind": "Q", "start": Q_START, "end": Q_START + timedelta(hours=1)},
            {"id": "2026-01-R", "kind": "R", "start": R_START, "end": R_START + timedelta(hours=2)},
        ],
        "entrants": [_driver("a"), _driver("b")],
        "targets": [
            {
                "target": "qualifying",
                "session_id": "2026-01-Q",
                "cutoff_at": Q_START - timedelta(minutes=30),
                "state": "issued" if q_issued else "scheduled",
                "forecast_id": "2026-01-qualifying-issued" if q_issued else None,
                "entrant_ids": ["a", "b"],
                "roster_source": "Operator verified entry list",
                "roster_observed_at": Q_START - timedelta(hours=2),
            },
            {
                "target": "race",
                "session_id": "2026-01-R",
                "cutoff_at": R_START - timedelta(minutes=30),
                "state": "scheduled",
                "entrant_ids": ["a", "b"],
                "roster_source": "Operator verified entry list",
                "roster_observed_at": Q_START - timedelta(hours=2),
            },
        ],
    }
    return event


def _issued_q_forecast() -> dict:
    drivers = []
    for rank, (identifier, probabilities) in enumerate((("a", [0.7, 0.3]), ("b", [0.3, 0.7])), 1):
        drivers.append(
            {
                "driver_id": identifier,
                "rank": rank,
                "expected_position": probabilities[0] + 2 * probabilities[1],
                "p10": 1,
                "p90": 2,
                "win_probability": probabilities[0],
                "podium_probability": 1.0,
                "top10_probability": 1.0,
                "position_probabilities": probabilities,
                "explanations": [],
            }
        )
    return {
        "id": "2026-01-qualifying-issued",
        "event_id": "2026-01",
        "target": "qualifying",
        "kind": "issued",
        "generated_at": Q_START - timedelta(minutes=40),
        "cutoff_at": Q_START - timedelta(minutes=30),
        "training_cutoff": datetime(2026, 9, 1, tzinfo=UTC),
        "model_version": "synthetic-test-model",
        "model_kind": "baseline",
        "seed": 42,
        "sample_count": 20_000,
        "coverage": {
            "available_sessions": ["2026-01-FP1"],
            "missing_sessions": [],
            "summary": "Synthetic FP1 only.",
            "flags": [],
        },
        "input_snapshot_ids": ["synthetic-test-snapshot"],
        "drivers": drivers,
    }


def _site(*, q_issued: bool = False) -> SiteData:
    return SiteData.model_validate(
        {
            "generated_at": Q_START - timedelta(days=2),
            "demo_notice": None,
            "events": [_event(q_issued=q_issued)],
            "forecasts": [_issued_q_forecast()] if q_issued else [],
            "analyses": [],
            "evaluation": {
                "status": "pending",
                "summary": "Awaiting evaluation.",
                "seasons": [],
                "comparison": [],
                "limitations": [],
            },
        }
    )


def test_missed_cutoff_tick_publishes_unavailable_without_a_forecast_and_repeat_is_stable(
    tmp_path, monkeypatch
):
    site_dir = tmp_path / "site-data"
    archive_dir = tmp_path / "archive"
    registry = tmp_path / "registry.json"
    publish_site(_site(), site_dir)
    registry.write_text('{"schema_version":"1.0","models":[]}')
    monkeypatch.setattr(
        "f1forecast.providers.JolpicaProvider",
        lambda *_args, **_kwargs: pytest.fail("network provider was called"),
    )

    first = run_tick(
        site_dir, archive_dir, registry, now=Q_START - timedelta(minutes=29), ingest=False
    )
    persisted = SiteData.model_validate_json((site_dir / "site.json").read_text())
    q = next(t for t in persisted.events[0].targets if t.target == "qualifying")
    race = next(t for t in persisted.events[0].targets if t.target == "race")
    assert first["unavailable"] == ["2026-01/qualifying"]
    assert q.state == "unavailable" and "cutoff" in q.reason.lower()
    assert race.state == "scheduled"
    assert persisted.forecasts == []
    assert not list((site_dir / "forecasts").rglob("*.json"))

    saved_bytes = (site_dir / "site.json").read_bytes()
    saved_mtime = (site_dir / "site.json").stat().st_mtime_ns
    again = run_tick(
        site_dir, archive_dir, registry, now=Q_START - timedelta(minutes=20), ingest=False
    )
    assert again["issued"] == again["unavailable"] == []
    assert (site_dir / "site.json").read_bytes() == saved_bytes
    assert (site_dir / "site.json").stat().st_mtime_ns == saved_mtime


def test_race_roster_substitution_preserves_frozen_qualifying_roster_and_rows(tmp_path):
    site_dir = tmp_path / "site-data"
    publish_site(_site(q_issued=True), site_dir)
    before = SiteData.model_validate_json((site_dir / "site.json").read_text())
    frozen_forecast = before.forecasts[0].model_dump(mode="json")
    assert target_rows(before.events[0], "qualifying").driver_id.tolist() == ["a", "b"]

    set_entrants(
        site_dir,
        "2026-01",
        [_driver("b"), _driver("c")],
        target="race",
        observed_at=R_START - timedelta(hours=3),
        source="Verified substitution",
    )
    after = SiteData.model_validate_json((site_dir / "site.json").read_text())
    assert after.forecasts[0].model_dump(mode="json") == frozen_forecast
    assert target_rows(after.events[0], "qualifying").driver_id.tolist() == ["a", "b"]
    assert target_rows(after.events[0], "race").driver_id.tolist() == ["b", "c"]
    assert after.events[0].targets[0].entrant_ids == ["a", "b"]
    with pytest.raises(ValueError, match="frozen"):
        set_entrants(
            site_dir,
            "2026-01",
            [_driver("a"), _driver("c")],
            target="qualifying",
            observed_at=Q_START - timedelta(hours=3),
        )


def test_out_of_weekend_schedule_has_no_work():
    site = _site()
    assert not has_weekend_work(site, datetime(2026, 9, 1, tzinfo=UTC))
    assert not has_weekend_work(site, datetime(2026, 10, 10, tzinfo=UTC))
    assert has_weekend_work(site, Q_START - timedelta(minutes=45))


def test_result_reconciliation_remains_due_after_a_multi_day_feed_delay():
    site = _site(q_issued=True)
    assert has_weekend_work(site, Q_START + timedelta(days=3))
    assert not has_weekend_work(site, Q_START + timedelta(days=8))


def test_sprint_calendar_maps_actual_chronology_and_target_cutoffs():
    race = {
        "season": "2026",
        "round": "2",
        "raceName": "Synthetic Sprint Grand Prix",
        "Circuit": {"circuitId": "interlagos", "Location": {"country": "Brazil"}},
        "FirstPractice": {"date": "2026-11-06", "time": "10:00:00Z"},
        "SprintQualifying": {"date": "2026-11-06", "time": "14:00:00Z"},
        "Sprint": {"date": "2026-11-07", "time": "10:00:00Z"},
        "Qualifying": {"date": "2026-11-07", "time": "14:00:00Z"},
        "date": "2026-11-08",
        "time": "12:00:00Z",
    }
    event = event_from_race(race)
    assert [session.kind for session in event.sessions] == ["FP1", "SQ", "S", "Q", "R"]
    q = next(t for t in event.targets if t.target == "qualifying")
    r = next(t for t in event.targets if t.target == "race")
    assert event.sessions[2].end < q.cutoff_at < event.sessions[3].start
    assert event.sessions[3].end < r.cutoff_at < event.sessions[4].start
    assert q.cutoff_at == datetime(2026, 11, 7, 13, 30, tzinfo=UTC)


def _summary(loss: float, mae: float, events: int = 1) -> dict:
    return {
        "probability_loss": loss,
        "position_mae": mae,
        "win_brier": 0.1,
        "podium_brier": 0.1,
        "top_ten_brier": 0.1,
        "evaluated_events": events,
    }


def test_registry_pattern_gate_needs_scored_report_evidence_not_metadata_claims(tmp_path):
    model_path = tmp_path / "qualifying.cbm"
    model_path.write_bytes(b"synthetic model bytes for registry gate")
    metadata = {
        "target": "qualifying",
        "version": "synthetic-case-gate",
        "training_cutoff": "2025-12-01T00:00:00Z",
        "training_config": {"iterations": 10, "random_seed": 42},
        "input_dataset_sha256": "a" * 64,
        "baseline_name": "practice_pace",
        "baseline_temperature": 1.2,
        "temperature": 1.3,
        "calibration_oof_events": 2,
        "sha256": sha256(model_path.read_bytes()).hexdigest(),
        "validated_missing_patterns": [["missing-tyre-context"]],
    }
    report = {
        "evaluation_schema_version": "2",
        "events": [{"required_inputs_available": True, "unexpected_actual_entrants": []}],
        "target": "qualifying",
        "training_config": metadata["training_config"],
        "input_dataset_sha256": metadata["input_dataset_sha256"],
        "baseline_name": "practice_pace",
        "baseline_temperature": 1.2,
        "periods": {"locked_2025": {"baseline": _summary(0.8, 1.0)}},
        "missing_case_evidence": [
            {"pattern": ["missing-observed-conditions"], "baseline": _summary(0.9, 1.1)},
            {"pattern": ["missing-long-run-pace"], "baseline": _summary(0.0, 0.0, events=0)},
        ],
    }
    metadata_path = tmp_path / "qualifying.metadata.json"
    report_path = tmp_path / "qualifying.evaluation.json"
    history_path = tmp_path / "history.parquet"
    registry_path = tmp_path / "registry.json"
    metadata_path.write_text(json.dumps(metadata))
    report_path.write_text(json.dumps(report))
    history_path.write_bytes(b"synthetic history bytes for registry gate")
    record = approve_model(metadata_path, report_path, history_path, registry_path, kind="baseline")
    assert record.validated_missing_patterns == [["missing-observed-conditions"]]
    assert ["missing-tyre-context"] not in record.validated_missing_patterns


def test_catboost_missing_pattern_needs_its_own_paired_2025_promotion():
    report = {
        "missing_case_evidence": [
            {
                "pattern": ["missing-telemetry"],
                "learned": _summary(0.2, 0.9),
                "paired_locked_2025": {
                    "learned": _summary(0.2, 0.9),
                    "baseline": _summary(0.3, 1.0),
                },
            },
            {
                "pattern": ["missing-tyre-context"],
                "learned": _summary(0.2, 1.1),
                "paired_locked_2025": {
                    "learned": _summary(0.2, 1.1),
                    "baseline": _summary(0.3, 1.0),
                },
            },
            {
                "pattern": ["missing-long-run-pace"],
                "learned": _summary(0.1, 0.7),
                "paired_locked_2025": {
                    "learned": _summary(0.1, 0.7, events=0),
                    "baseline": _summary(0.3, 1.0, events=0),
                },
            },
            {
                "pattern": ["missing-feature-schema"],
                "learned": _summary(0.1, 0.7),
                "paired_locked_2025": {
                    "learned": _summary(0.1, 0.7),
                    "baseline": _summary(0.3, 1.0),
                },
            },
        ]
    }
    assert validated_patterns(report, "catboost") == [["missing-telemetry"]]
