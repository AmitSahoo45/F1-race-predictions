"""Lap-derived model inputs retain actual source stint boundaries."""

import math

import pandas as pd
import pytest


def _summary(laps):
    from f1forecast.lap_summaries import summarize_laps

    return summarize_laps(pd.DataFrame(laps))


def test_separate_adjacent_stints_do_not_form_a_long_run():
    result = _summary(
        {
            "LapNumber": [1, 2, 3, 4],
            "Stint": [1, 1, 2, 2],
            "lap_seconds": [90, 90, 110, 110],
            "usable": [True] * 4,
            "Compound": ["MEDIUM"] * 4,
        }
    )
    assert math.isnan(result["long_run_pace_s"])
    assert result["consistency_s"] == 0.0


def test_long_runs_require_consecutive_usable_laps_inside_a_stint():
    result = _summary(
        {
            "LapNumber": [7, 1, 2, 3, 4, 5, 6],
            "Stint": [2, 1, 1, 1, 1, 2, 2],
            "lap_seconds": [100, 90, 91, 300, 92, 102, 101],
            "usable": [True, True, True, False, True, True, True],
            "Compound": ["MEDIUM"] * 7,
        }
    )
    assert result["long_run_pace_s"] == 101.0
    assert result["consistency_s"] == 1.0


def test_consistency_pools_within_stint_degrees_of_freedom():
    result = _summary(
        {
            "LapNumber": [1, 2, 3, 4, 5],
            "Stint": [1, 1, 2, 2, 2],
            "lap_seconds": [89, 91, 118, 120, 122],
            "usable": [True] * 5,
        }
    )
    assert result["consistency_s"] == pytest.approx(math.sqrt(10 / 3))


@pytest.mark.parametrize("stints", [None, [None] * 3, [1, None, 1], [1, 1.5, 1]])
def test_missing_or_invalid_source_stints_leave_stint_metrics_missing(stints):
    laps = {
        "LapNumber": [1, 2, 3],
        "lap_seconds": [89, 90, 91],
        "usable": [True] * 3,
    }
    if stints is not None:
        laps["Stint"] = stints
    result = _summary(laps)
    assert math.isnan(result["long_run_pace_s"])
    assert math.isnan(result["consistency_s"])


def test_compound_fractions_count_only_usable_laps_and_preserve_known_absence():
    result = _summary(
        {
            "LapNumber": [1, 2, 3, 4],
            "Stint": [1, 1, 2, 2],
            "lap_seconds": [90, 91, 92, 300],
            "usable": [True, True, True, False],
            "Compound": ["SOFT", "SOFT", "MEDIUM", None],
        }
    )
    assert result["compound_soft_fraction"] == pytest.approx(2 / 3)
    assert result["compound_medium_fraction"] == pytest.approx(1 / 3)
    assert result["compound_hard_fraction"] == 0.0
    assert result["compound_intermediate_fraction"] == 0.0
    assert result["compound_wet_fraction"] == 0.0


@pytest.mark.parametrize("compounds", [["SOFT", None], ["SOFT", "UNKNOWN"], None])
def test_unknown_compound_leaves_entire_compound_vector_missing(compounds):
    laps = {"lap_seconds": [90, 91], "usable": [True, True]}
    if compounds is not None:
        laps["Compound"] = compounds
    result = _summary(laps)
    assert all(math.isnan(value) for key, value in result.items() if key.startswith("compound_"))


def test_no_usable_laps_do_not_assert_zero_compound_exposure():
    result = _summary({"lap_seconds": [90], "usable": [False], "Compound": ["SOFT"]})
    assert all(math.isnan(value) for value in result.values())


def test_normalization_uses_actual_stints_and_emits_compound_inputs():
    from f1forecast.ingestion import normalize_session
    from test_ingestion import NOW, raw_session

    raw = raw_session()
    raw.laps["Stint"] = [1, 2]
    summary, _, _ = normalize_session(raw, NOW)
    assert pd.isna(summary.iloc[0]["consistency_s"])
    assert summary.iloc[0]["compound_medium_fraction"] == 1.0
