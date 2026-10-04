"""Observable tyre context and pace statistics within actual source stints."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

LAP_SUMMARY_SEMANTICS = "source-stints-compound-fractions-v1"
COMPOUNDS = ("soft", "medium", "hard", "intermediate", "wet")
COMPOUND_FEATURES = tuple(f"compound_{name}_fraction" for name in COMPOUNDS)
LAP_SUMMARY_FIELDS = ("long_run_pace_s", "consistency_s", *COMPOUND_FEATURES)


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    column = frame[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"duplicate lap column: {name}")
    return column


def summarize_laps(laps: pd.DataFrame) -> dict[str, float]:
    """Summarize one driver's normalized laps without inventing stint boundaries.

    Long-run pace pools laps from consecutive usable runs of at least three
    inside a source stint. Consistency is the pooled within-stint sample standard
    deviation, so different stint pace levels do not increase it. Unknown stint
    IDs on usable laps leave both metrics missing. Compound fractions describe
    relative compound labels, not absolute tyre specifications across weekends.
    """
    result = {name: float("nan") for name in LAP_SUMMARY_FIELDS}
    if laps.empty or not {"usable", "lap_seconds"}.issubset(laps.columns):
        return result
    usable = laps.loc[_series(laps, "usable").eq(True)].copy()
    if usable.empty:
        return result
    seconds = cast(pd.Series, pd.to_numeric(_series(usable, "lap_seconds"), errors="coerce"))
    if not np.isfinite(seconds.to_numpy(dtype="float64")).all():
        return result
    usable["lap_seconds"] = seconds
    if "Compound" in usable:
        compounds = _series(usable, "Compound").astype("string").str.strip().str.lower()
        if bool(compounds.isin(COMPOUNDS).all()):
            for compound, feature in zip(COMPOUNDS, COMPOUND_FEATURES, strict=True):
                result[feature] = float(compounds.eq(compound).mean())
    if not {"Stint", "LapNumber"}.issubset(usable.columns):
        return result
    stints = cast(pd.Series, pd.to_numeric(_series(usable, "Stint"), errors="coerce"))
    numbers = cast(pd.Series, pd.to_numeric(_series(usable, "LapNumber"), errors="coerce"))
    for values in (stints, numbers):
        if not bool((values.notna() & values.gt(0) & values.mod(1).eq(0)).all()):
            return result
    if bool(numbers.duplicated().any()):
        return result
    usable["Stint"], usable["LapNumber"] = stints, numbers
    squared_deviations, degrees_of_freedom = 0.0, 0
    long_run_laps: list[float] = []
    for _, stint in usable.groupby("Stint", sort=False):
        stint = cast(pd.DataFrame, stint).sort_values("LapNumber")
        times = _series(stint, "lap_seconds")
        if len(times) >= 2:
            squared_deviations += float(((times - times.mean()) ** 2).sum())
            degrees_of_freedom += len(times) - 1
        runs = _series(stint, "LapNumber").diff().ne(1).cumsum()
        for _, run in stint.groupby(runs, sort=False):
            run = cast(pd.DataFrame, run)
            if len(run) >= 3:
                long_run_laps.extend(_series(run, "lap_seconds").astype(float).tolist())
    if degrees_of_freedom:
        result["consistency_s"] = math.sqrt(squared_deviations / degrees_of_freedom)
    if long_run_laps:
        result["long_run_pace_s"] = float(np.median(long_run_laps))
    return result
