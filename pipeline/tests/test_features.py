from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from f1forecast.features import NUMERIC_FEATURES, build_feature_rows, missing_feature_groups


def _row(event: str, session: str, driver: str, start: str, end: str, **overrides):
    row = {
        "event_id": event,
        "season": int(event[:4]),
        "circuit_id": "bahrain",
        "driver_id": driver,
        "team_id": "red" if driver == "a" else "blue",
        "session_id": f"{event}-{session}",
        "session_type": session,
        "session_start": start,
        "session_end": end,
        "observed_at": end,
        "available_at": end,
        "provenance": "prospective",
        "position": None,
        "status": None,
        "pace_s": None,
        "long_run_pace_s": None,
        "consistency_s": None,
        "usable_laps": None,
        "tyre_age": None,
        "compound": None,
        "mean_speed": None,
        "mean_throttle": None,
        "brake_fraction": None,
        "air_temp": None,
        "track_temp": None,
        "qualifying_gap": None,
    }
    row.update(overrides)
    return row


def _target(target: str = "qualifying"):
    return pd.DataFrame(
        {
            "event_id": ["2025-02", "2025-02"],
            "season": [2025, 2025],
            "circuit_id": ["bahrain", "bahrain"],
            "driver_id": ["a", "b"],
            "team_id": ["red", "blue"],
            "target": [target, target],
            "target_session_id": [f"2025-02-{'Q' if target == 'qualifying' else 'R'}"] * 2,
            "cutoff": ["2025-04-12T11:30:00Z"] * 2,
        }
    )


def test_team_identity_aliases_match_history_without_merging_prior_brand():
    rows = pd.DataFrame([
        _row("2025-01", "Q", "a", "2025-03-01T09:00Z", "2025-03-01T10:00Z",
             team_id="racing_bulls", position=1),
        _row("2025-01", "Q", "b", "2025-03-01T09:00Z", "2025-03-01T10:00Z",
             team_id="blue", position=2),
        _row("2024-01", "Q", "a", "2024-03-01T09:00Z", "2024-03-01T10:00Z",
             team_id="rb", position=2),
        _row("2024-01", "Q", "b", "2024-03-01T09:00Z", "2024-03-01T10:00Z",
             team_id="blue", position=1),
    ])
    targets = _target()
    targets.loc[0, "team_id"] = "RB F1 Team"
    got = build_feature_rows(rows, targets).set_index("driver_id")
    assert got.loc["a", "team_id"] == "racing_bulls"
    assert got.loc["a", "recent_team_qualifying_rank"] == pytest.approx(0.0)
    # Both inputs retain their source values.
    assert targets.loc[0, "team_id"] == "RB F1 Team"
    assert rows.loc[2, "team_id"] == "rb"
    rows.loc[0, "team_id"] = "rb_f1_team"
    targets.loc[0, "team_id"] = "racing_bulls"
    assert build_feature_rows(rows, targets).iloc[0].recent_team_qualifying_rank == 0.0


@pytest.mark.parametrize("missing_team", [None, "None", " nan ", "NULL", ""])
def test_missing_team_identity_never_pools_unknown_teams_or_infers_affiliation(missing_team):
    rows = pd.DataFrame([
        _row("2025-01", "Q", "a", "2025-03-01T09:00Z", "2025-03-01T10:00Z",
             team_id=missing_team, position=1),
        _row("2025-01", "Q", "b", "2025-03-01T09:00Z", "2025-03-01T10:00Z",
             team_id="blue", position=2),
        _row("2024-01", "R", "a", "2024-03-01T09:00Z", "2024-03-01T10:00Z",
             team_id="red", position=1),
        _row("2025-02", "Q", "a", "2025-04-12T12:00Z", "2025-04-12T13:00Z",
             team_id="red", position=1),
    ])
    targets = _target()
    targets.loc[0, "team_id"] = missing_team
    got = build_feature_rows(rows, targets).set_index("driver_id")
    assert pd.isna(got.loc["a", "team_id"])
    assert pd.isna(got.loc["a", "recent_team_qualifying_rank"])
    assert pd.isna(got.loc["a", "recent_team_race_rank"])
    assert "missing-team-form" in missing_feature_groups(got, "qualifying")


def test_prospective_features_exclude_late_feed_and_target_result():
    rows = [
        _row("2024-01", "Q", "a", "2024-03-01T09:00Z", "2024-03-01T10:00Z", position=1),
        _row("2024-01", "Q", "b", "2024-03-01T09:00Z", "2024-03-01T10:00Z", position=2),
        _row(
            "2025-02",
            "FP1",
            "a",
            "2025-04-12T09:00Z",
            "2025-04-12T10:00Z",
            pace_s=90,
            usable_laps=8,
        ),
        _row(
            "2025-02",
            "FP1",
            "b",
            "2025-04-12T09:00Z",
            "2025-04-12T10:00Z",
            pace_s=91,
            usable_laps=9,
        ),
        _row(
            "2025-02",
            "FP2",
            "a",
            "2025-04-12T10:00Z",
            "2025-04-12T11:00Z",
            pace_s=75,
            available_at="2025-04-12T11:31Z",
        ),
        _row("2025-02", "Q", "a", "2025-04-12T12:00Z", "2025-04-12T13:00Z", position=1, pace_s=70),
    ]
    got = build_feature_rows(pd.DataFrame(rows), _target(), mode="prospective")
    a = got.set_index("driver_id").loc["a"]
    assert a["practice_pace_gap_s"] == pytest.approx(-0.5)
    assert a["practice_sessions"] == 1
    assert a["recent_qualifying_rank"] == pytest.approx(0)
    assert a["source_latest_at"] == pd.Timestamp("2025-04-12T10:00Z")


def test_reconstruction_requires_explicit_mode_and_uses_session_end_for_old_download():
    rows = pd.DataFrame(
        [
            _row(
                "2025-02",
                "FP1",
                "a",
                "2025-04-12T09:00Z",
                "2025-04-12T10:00Z",
                pace_s=90,
                observed_at="2025-04-12T10:00Z",
                available_at="2026-01-01T00:00Z",
                provenance="retrospective",
            ),
            _row(
                "2025-02",
                "FP1",
                "b",
                "2025-04-12T09:00Z",
                "2025-04-12T10:00Z",
                pace_s=91,
                observed_at="2025-04-12T10:00Z",
                available_at="2026-01-01T00:00Z",
                provenance="retrospective",
            ),
        ]
    )
    live = build_feature_rows(rows, _target(), mode="prospective")
    historical = build_feature_rows(rows, _target(), mode="reconstructed")
    assert live["practice_sessions"].eq(0).all()
    assert historical["practice_sessions"].eq(1).all()
    assert historical["feature_provenance"].eq("reconstructed").all()
    assert historical["source_latest_at"].eq(pd.Timestamp("2025-04-12T10:00Z")).all()


def test_race_uses_prior_qualifying_but_never_sprint_or_future_sessions():
    rows = [
        _row(
            "2025-02",
            "Q",
            "a",
            "2025-04-12T09:00Z",
            "2025-04-12T10:00Z",
            position=2,
            qualifying_gap=0.2,
        ),
        _row(
            "2025-02",
            "Q",
            "b",
            "2025-04-12T09:00Z",
            "2025-04-12T10:00Z",
            position=1,
            qualifying_gap=0,
        ),
        _row("2025-02", "S", "a", "2025-04-12T10:00Z", "2025-04-12T11:00Z", position=1),
        _row("2025-02", "R", "a", "2025-04-12T12:00Z", "2025-04-12T14:00Z", position=1),
    ]
    got = build_feature_rows(pd.DataFrame(rows), _target("race"), mode="prospective")
    a = got.set_index("driver_id").loc["a"]
    assert a["qualifying_position"] == 2
    assert a["qualifying_gap_s"] == pytest.approx(0.2)
    assert a["practice_sessions"] == 0
    assert a["source_latest_at"] == pd.Timestamp("2025-04-12T10:00Z")


def test_rookie_and_substitute_keep_missing_history_with_zero_counts():
    got = build_feature_rows(pd.DataFrame(), _target(), mode="prospective")
    assert got["recent_qualifying_count"].eq(0).all()
    assert got["circuit_history_count"].eq(0).all()
    assert got["recent_qualifying_rank"].isna().all()
    assert got["practice_pace_gap_s"].isna().all()


def test_missing_pattern_covers_optional_groups_and_excludes_qualifying_only_structural_fields():
    row = {name: 1.0 for name in NUMERIC_FEATURES}
    row.update(
        {
            "recent_qualifying_rank": np.nan,
            "recent_qualifying_count": 0,
            "long_run_pace_gap_s": np.nan,
            "tyre_age": np.nan,
            "air_temp": np.nan,
            "qualifying_position": np.nan,
            "qualifying_gap_s": np.nan,
        }
    )
    qualifying = missing_feature_groups(pd.DataFrame([row]), "qualifying")
    assert "new-driver" in qualifying
    assert "missing-long-run-pace" in qualifying
    assert "missing-tyre-context" in qualifying
    assert "missing-observed-conditions" in qualifying
    assert "missing-qualifying-result" not in qualifying
    assert "missing-qualifying-gap" not in qualifying
    absent_structural = pd.DataFrame([row]).drop(
        columns=["qualifying_position", "qualifying_gap_s"]
    )
    assert "missing-feature-schema" not in missing_feature_groups(absent_structural, "qualifying")
    race = missing_feature_groups(pd.DataFrame([row]), "race")
    assert "missing-qualifying-result" in race
    assert "missing-qualifying-gap" in race


def test_each_nonstructural_numeric_feature_has_a_missing_pattern():
    full = {name: 1.0 for name in NUMERIC_FEATURES}
    for name in NUMERIC_FEATURES:
        if name in {"qualifying_position", "qualifying_gap_s"}:
            continue
        row = dict(full)
        row[name] = np.nan
        assert missing_feature_groups(pd.DataFrame([row]), "qualifying"), name


def test_rookie_does_not_hide_missing_history_for_an_established_driver():
    full = {name: 1.0 for name in NUMERIC_FEATURES}
    rookie = dict(full, recent_qualifying_count=0, recent_qualifying_rank=np.nan)
    established = dict(full, recent_qualifying_count=3, recent_qualifying_rank=np.nan)
    pattern = missing_feature_groups(pd.DataFrame([rookie, established]), "qualifying")
    assert "new-driver" in pattern
    assert "missing-recent-qualifying-form" in pattern


def test_incomplete_sessions_never_supply_practice_form_or_race_qualifying():
    rows = pd.DataFrame(
        [
            _row(
                "2024-01",
                "Q",
                "a",
                "2024-03-01T09:00Z",
                "2024-03-01T10:00Z",
                position=1,
                session_complete=False,
                source_snapshot_id="old-partial-q",
            ),
            _row(
                "2024-01",
                "Q",
                "b",
                "2024-03-01T09:00Z",
                "2024-03-01T10:00Z",
                position=2,
                session_complete=False,
                source_snapshot_id="old-partial-q",
            ),
            _row(
                "2025-02",
                "FP1",
                "a",
                "2025-04-12T08:00Z",
                "2025-04-12T09:00Z",
                pace_s=90,
                session_complete=True,
                source_snapshot_id="complete-fp1",
            ),
            _row(
                "2025-02",
                "FP1",
                "b",
                "2025-04-12T08:00Z",
                "2025-04-12T09:00Z",
                pace_s=91,
                session_complete=True,
                source_snapshot_id="complete-fp1",
            ),
            _row(
                "2025-02",
                "FP2",
                "a",
                "2025-04-12T09:00Z",
                "2025-04-12T10:00Z",
                pace_s=70,
                session_complete=False,
                source_snapshot_id="partial-fp2",
            ),
            _row(
                "2025-02",
                "FP2",
                "b",
                "2025-04-12T09:00Z",
                "2025-04-12T10:00Z",
                pace_s=71,
                session_complete=False,
                source_snapshot_id="partial-fp2",
            ),
            _row(
                "2025-02",
                "Q",
                "a",
                "2025-04-12T10:00Z",
                "2025-04-12T11:00Z",
                position=2,
                qualifying_gap=0.2,
                session_complete=False,
                source_snapshot_id="partial-q",
            ),
            _row(
                "2025-02",
                "Q",
                "b",
                "2025-04-12T10:00Z",
                "2025-04-12T11:00Z",
                position=1,
                qualifying_gap=0,
                session_complete=False,
                source_snapshot_id="partial-q",
            ),
        ]
    )
    features = build_feature_rows(rows, _target("race"), mode="prospective")
    assert features["recent_qualifying_count"].eq(0).all()
    assert features["practice_sessions"].eq(1).all()
    assert features["qualifying_position"].isna().all()
    assert all(value == ("2025-02-FP1",) for value in features["used_session_ids"])
    assert all(value == ("complete-fp1",) for value in features["used_snapshot_ids"])
    rows.loc[rows["session_id"].eq("2025-02-Q"), "session_complete"] = True
    completed = build_feature_rows(rows, _target("race"), mode="prospective")
    assert completed.set_index("driver_id").loc["a", "qualifying_position"] == 2
    assert "partial-q" in completed.iloc[0]["used_snapshot_ids"]
