import numpy as np
import pandas as pd
import pytest
from f1forecast.features import NUMERIC_FEATURES, build_feature_rows, missing_feature_groups
from f1forecast.modeling import FEATURE_GROUPS


def _practice(session, end, laps, **fractions):
    return {
        "event_id": "2024-01", "season": 2024, "circuit_id": "test",
        "driver_id": "a", "team_id": "team", "session_id": f"2024-01-{session}",
        "session_type": session, "session_end": end, "observed_at": end,
        "available_at": end, "provenance": "prospective", "usable_laps": laps,
        **{f"compound_{name}_fraction": fractions.get(name, 0.0)
           for name in ("soft", "medium", "hard", "intermediate", "wet")},
    }


def _target():
    return pd.DataFrame([{
        "event_id": "2024-01", "season": 2024, "circuit_id": "test", "driver_id": "a",
        "team_id": "team", "target": "qualifying", "target_session_id": "2024-01-Q",
        "cutoff": "2024-03-02T11:30Z",
    }])


def test_compound_model_inputs_are_lap_weighted_and_cutoff_safe():
    source = pd.DataFrame([
        _practice("FP1", "2024-03-01T10:00Z", 2, soft=1.0),
        _practice("FP2", "2024-03-01T14:00Z", 6, hard=1.0),
        _practice("Q", "2024-03-02T13:00Z", 20, medium=1.0),
        _practice("FP3", "2024-03-02T12:00Z", 20, wet=1.0),
    ])
    row = build_feature_rows(source, _target()).iloc[0]
    assert row["compound_soft_fraction"] == pytest.approx(0.25)
    assert row["compound_hard_fraction"] == pytest.approx(0.75)
    assert row["compound_medium_fraction"] == 0
    assert row["compound_intermediate_fraction"] == 0
    assert row["compound_wet_fraction"] == 0
    for name in ("soft", "medium", "hard", "intermediate", "wet"):
        column = f"compound_{name}_fraction"
        assert column in NUMERIC_FEATURES
        assert FEATURE_GROUPS[column] == "tyre context"


def test_unknown_compound_coverage_cannot_be_reported_as_known_zero():
    source = pd.DataFrame([
        _practice("FP1", "2024-03-01T10:00Z", 2, soft=1.0),
        _practice("FP2", "2024-03-01T14:00Z", 6,
                  soft=np.nan, medium=np.nan, hard=np.nan, intermediate=np.nan, wet=np.nan),
    ])
    row = build_feature_rows(source, _target())
    columns = [f"compound_{name}_fraction"
               for name in ("soft", "medium", "hard", "intermediate", "wet")]
    assert row[columns].isna().all().all()
    assert "missing-tyre-context" in missing_feature_groups(row, "qualifying")
    legacy = build_feature_rows(source.drop(columns=columns), _target())
    assert legacy[columns].isna().all().all()


def test_zero_usable_laps_do_not_dilute_known_compound_mix():
    source = pd.DataFrame([
        _practice("FP1", "2024-03-01T10:00Z", 2, soft=1.0),
        _practice("FP2", "2024-03-01T14:00Z", 0,
                  soft=np.nan, medium=np.nan, hard=np.nan, intermediate=np.nan, wet=np.nan),
    ])
    row = build_feature_rows(source, _target()).iloc[0]
    assert row["compound_soft_fraction"] == 1
    assert row["compound_hard_fraction"] == 0
