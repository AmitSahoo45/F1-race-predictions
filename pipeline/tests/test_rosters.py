"""A provisional roster must use completed, eligible observations and real identities."""

import pandas as pd
from f1forecast.rosters import infer_roster
from test_operations import example_event


def test_provisional_roster_ignores_partial_later_sessions_and_keeps_observed_names(
    tmp_path, monkeypatch
):
    event = example_event()
    target = next(t for t in event.targets if t.target == "race")
    rows = pd.DataFrame(
        [
            {
                "event_id": event.id,
                "session_id": event.id + "-Q",
                "session_type": "Q",
                "session_end": "2026-10-03T14:00:00Z",
                "available_at": "2026-10-03T15:00:00Z",
                "session_complete": True,
                "driver_id": "hulkenberg",
                "driver_code": "HUL",
                "team_id": "test_team",
            },
            {
                "event_id": event.id,
                "session_id": event.id + "-Q-partial",
                "session_type": "Q",
                "session_end": "2026-10-04T10:00:00Z",
                "available_at": "2026-10-04T10:10:00Z",
                "session_complete": False,
                "driver_id": "wrong",
                "driver_code": "BAD",
                "team_id": "test_team",
            },
        ]
    )
    observed = rows.iloc[:1]
    results = pd.DataFrame(
        [
            {
                "Abbreviation": "HUL",
                "FullName": "Nico Hülkenberg",
                "TeamName": "Test Team",
                "TeamColor": "abcdef",
            }
        ]
    )
    monkeypatch.setattr(
        "f1forecast.rosters.load_event_snapshots",
        lambda *_a, **_k: [
            (
                {"session_id": event.id + "-Q", "session_start": "2026-10-03T13:00:00Z"},
                observed,
                {"results": results},
            )
        ],
    )
    roster, observed_at = infer_roster(event, target, rows, tmp_path)
    assert [(d.id, d.code, d.name, d.team) for d in roster] == [
        ("hulkenberg", "HUL", "Nico Hülkenberg", "Test Team")
    ]
    assert observed_at == pd.Timestamp("2026-10-03T15:00:00Z")


def test_unfinalized_or_post_cutoff_roster_is_unavailable(tmp_path):
    event = example_event()
    target = next(t for t in event.targets if t.target == "race")
    rows = pd.DataFrame(
        [
            {
                "event_id": event.id,
                "session_id": event.id + "-Q",
                "session_type": "Q",
                "session_end": "2026-10-03T14:00:00Z",
                "available_at": "2026-10-04T12:30:00Z",
                "session_complete": True,
            }
        ]
    )
    assert infer_roster(event, target, rows, tmp_path) == ([], None)
