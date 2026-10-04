from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from f1forecast.contracts import Event
from f1forecast.features import NUMERIC_FEATURES
from f1forecast.operations import (
    eligible_summary_versions,
    publication_window,
    target_rows,
    validate_inputs,
)
from f1forecast.registry import ModelRegistry, select_model


def example_event():
    start = datetime(2026, 10, 4, 12, tzinfo=UTC)
    return Event.model_validate(
        {
            "id": "2026-01",
            "season": 2026,
            "round": 1,
            "name": "Example",
            "circuit": "example",
            "country": "Example",
            "timezone": "UTC",
            "sessions": [
                {"id": "2026-01-R", "kind": "R", "start": start, "end": start + timedelta(hours=2)}
            ],
            "entrants": [
                {"id": "a", "code": "AAA", "name": "A", "team": "Team", "color": "#ffffff"},
                {"id": "b", "code": "BBB", "name": "B", "team": "Team", "color": "#ffffff"},
            ],
            "targets": [
                {
                    "target": "race",
                    "session_id": "2026-01-R",
                    "cutoff_at": start - timedelta(minutes=30),
                    "state": "scheduled",
                }
            ],
        }
    )


def test_forecast_window_never_backfills_and_does_not_issue_too_early():
    target = example_event().targets[0]
    assert publication_window(target, target.cutoff_at - timedelta(minutes=31)) == "early"
    assert publication_window(target, target.cutoff_at - timedelta(minutes=10)) == "ready"
    assert publication_window(target, target.cutoff_at + timedelta(seconds=1)) == "missed"


def test_target_entries_come_from_pre_session_manifest():
    frame = target_rows(example_event(), "race")
    assert frame.driver_id.tolist() == ["a", "b"]
    assert frame.target_session_id.unique().tolist() == ["2026-01-R"]


def test_target_team_identity_uses_canonical_alias_and_preserves_missing_team():
    event = example_event()
    event.entrants[0].team = "RB F1 Team"
    event.entrants[1].team = "None"
    frame = target_rows(event, "race")
    assert frame.team_id.tolist() == ["racing_bulls", None]
    assert event.entrants[0].team == "RB F1 Team"


def test_missing_required_qualifying_blocks_race_and_optional_cases_fail_closed():
    rows = pd.DataFrame(
        {
            **{name: [1.0, 1.0] for name in NUMERIC_FEATURES},
            "driver_id": ["a", "b"],
            "qualifying_position": [1, None],
            "practice_sessions": [3, 3],
            "practice_pace_gap_s": [0, 0.1],
            "long_run_pace_gap_s": [0, 0.2],
            "mean_speed": [210, 200],
            "mean_throttle": [70, 60],
            "brake_fraction": [0.2, 0.3],
            "recent_qualifying_count": [3, 0],
        }
    )
    with pytest.raises(ValueError, match="qualifying"):
        validate_inputs(rows, "race", [])
    rows["qualifying_position"] = [1, 2]
    with pytest.raises(ValueError, match="missing-data"):
        validate_inputs(rows, "race", [])
    assert validate_inputs(rows, "race", [["new-driver"]]) == ["new-driver"]


def test_missing_weather_and_qualifying_gap_require_evaluated_exact_pattern():
    rows = pd.DataFrame({name: [1.0, 2.0] for name in NUMERIC_FEATURES})
    rows.loc[0, "air_temp"] = float("nan")
    rows.loc[1, "qualifying_gap_s"] = float("nan")
    with pytest.raises(ValueError, match="missing-data"):
        validate_inputs(rows, "race", [])
    pattern = ["missing-observed-conditions", "missing-qualifying-gap"]
    assert validate_inputs(rows, "race", [pattern]) == pattern
    with pytest.raises(ValueError, match="missing-data"):
        validate_inputs(rows, "race", [["missing-observed-conditions"]])


def test_empty_registry_and_future_training_are_never_approved():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = ModelRegistry.model_validate({"models": []})
    assert select_model(registry, "race", now) is None


def test_late_or_partial_snapshot_cannot_replace_a_completed_pre_cutoff_version():
    rows = pd.DataFrame(
        [
            {
                "session_id": "2026-01-Q",
                "driver_id": "a",
                "session_complete": True,
                "available_at": "2026-10-03T15:00:00Z",
                "position": 2,
            },
            {
                "session_id": "2026-01-Q",
                "driver_id": "a",
                "session_complete": False,
                "available_at": "2026-10-04T10:00:00Z",
                "position": 3,
            },
            {
                "session_id": "2026-01-Q",
                "driver_id": "a",
                "session_complete": True,
                "available_at": "2026-10-04T12:00:00Z",
                "position": 1,
            },
        ]
    )
    eligible = eligible_summary_versions(rows, datetime(2026, 10, 4, 11, 30, tzinfo=UTC))
    assert eligible.position.tolist() == [2]


def test_latest_eligible_snapshot_replaces_the_whole_session_roster():
    rows = pd.DataFrame(
        [
            {
                "session_id": "2026-01-Q",
                "driver_id": driver,
                "source_snapshot_id": snapshot,
                "session_complete": complete,
                "available_at": available,
                "position": position,
            }
            for snapshot, available, complete, drivers in (
                ("old", "2026-10-03T15:00:00Z", True, ("a", "b")),
                ("new", "2026-10-04T10:00:00Z", True, ("a", "c")),
                ("partial", "2026-10-04T11:00:00Z", False, ("a", "e")),
                ("late", "2026-10-04T12:00:00Z", True, ("a", "d")),
            )
            for position, driver in enumerate(drivers, 1)
        ]
    )
    eligible = eligible_summary_versions(rows, datetime(2026, 10, 4, 11, 30, tzinfo=UTC))
    assert set(eligible.driver_id) == {"a", "c"}
    assert set(eligible.source_snapshot_id) == {"new"}
    before_correction = eligible_summary_versions(rows, datetime(2026, 10, 4, 9, 30, tzinfo=UTC))
    assert set(before_correction.driver_id) == {"a", "b"}
    assert set(before_correction.source_snapshot_id) == {"old"}
    after_late_arrival = eligible_summary_versions(rows, datetime(2026, 10, 4, 13, 0, tzinfo=UTC))
    assert set(after_late_arrival.driver_id) == {"a", "d"}
    assert set(after_late_arrival.source_snapshot_id) == {"late"}
