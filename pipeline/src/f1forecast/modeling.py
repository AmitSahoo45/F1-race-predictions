"""Event-grouped CatBoost PairLogit ranking and score explanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRanker, EFstrType, Pool

from f1forecast.features import NUMERIC_FEATURES

FEATURE_GROUPS: dict[str, str] = {
    "recent_qualifying_rank": "recent qualifying form",
    "recent_qualifying_count": "recent qualifying form",
    "recent_race_rank": "recent race form",
    "recent_race_count": "recent race form",
    "recent_team_qualifying_rank": "team form",
    "recent_team_race_rank": "team form",
    "circuit_qualifying_rank": "circuit history",
    "circuit_race_rank": "circuit history",
    "circuit_history_count": "circuit history",
    "practice_pace_gap_s": "practice pace",
    "long_run_pace_gap_s": "long-run pace",
    "consistency_s": "stint consistency",
    "usable_laps": "practice coverage",
    "tyre_age": "tyre context",
    "mean_speed": "public telemetry summary",
    "mean_throttle": "public telemetry summary",
    "brake_fraction": "public telemetry summary",
    "air_temp": "observed conditions",
    "track_temp": "observed conditions",
    "practice_sessions": "practice coverage",
    "qualifying_position": "qualifying result",
    "qualifying_gap_s": "qualifying result",
}


def fit_ranker(
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = NUMERIC_FEATURES,
    iterations: int = 300,
    random_seed: int = 42,
) -> CatBoostRanker:
    """Fit one model target; rows must contain complete event result orders."""
    features = list(feature_columns)
    if rows.empty or not features or not {"event_id", "position"}.issubset(rows.columns):
        raise ValueError("training requires event groups, features, and positions")
    if missing := set(features) - set(rows.columns):
        raise ValueError(f"training features missing {sorted(missing)}")
    ordered = rows.sort_values("event_id", kind="stable").copy()
    if np.asarray(ordered["position"].isna(), dtype=bool).any():
        raise ValueError("missing classification cannot train full-order ranker")
    if np.asarray(ordered.groupby("event_id").size().lt(2), dtype=bool).any():
        raise ValueError("ranking groups need at least two classified drivers")
    for _, group in ordered.groupby("event_id"):
        if set(group["position"].astype(int)) != set(range(1, len(group) + 1)):
            raise ValueError("training requires complete unique event positions")
    labels = (
        ordered.groupby("event_id")["position"].transform("max").to_numpy(dtype=float)
        + 1
        - ordered["position"].to_numpy(dtype=float)
    )
    pool = Pool(
        ordered[features].astype(float),
        label=labels,
        group_id=ordered["event_id"].astype(str).to_numpy(),
    )
    model = CatBoostRanker(
        loss_function="PairLogit",
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )
    model.fit(pool)
    return model


def predict_scores(
    model: CatBoostRanker, rows: pd.DataFrame, *, feature_columns: Sequence[str] = NUMERIC_FEATURES
) -> np.ndarray:
    return np.asarray(model.predict(rows[list(feature_columns)].astype(float)), dtype=float)


def score_explanations(
    model: CatBoostRanker,
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = NUMERIC_FEATURES,
    reference_medians: Mapping[str, float] | None = None,
    top_k: int = 3,
) -> tuple[str, list[list[dict[str, float | str]]]]:
    """Return grouped contributions to the score, never to sampled probabilities.

    CatBoost PairLogit SHAP support is checked at runtime. If unavailable,
    feature groups are permuted among the current field. That fallback is
    model score sensitivity, not SHAP or a probability explanation.
    """
    features = list(feature_columns)
    data = rows[features].astype(float)
    group_names = [FEATURE_GROUPS.get(name, name.replace("_", " ")) for name in features]
    try:
        pool = Pool(data, group_id=np.zeros(len(rows), dtype=int))
        shap = np.asarray(model.get_feature_importance(data=pool, type=EFstrType.ShapValues))
        values = shap[:, :-1]
        if values.shape != data.shape:
            raise ValueError("unexpected SHAP shape")
        method = "SHAP score contribution"
    except (CatBoostError, ValueError, RuntimeError, TypeError):
        if len(rows) < 2:
            raise ValueError("grouped permutation requires at least two drivers")
        base = predict_scores(model, rows, feature_columns=features)
        values = np.zeros(data.shape)
        rng = np.random.default_rng(42)
        for group in set(group_names):
            indices = [index for index, name in enumerate(group_names) if name == group]
            changed = data.copy()
            permutation = rng.permutation(len(data))
            changed.iloc[:, indices] = data.iloc[permutation, indices].to_numpy()
            values[:, indices[0]] = base - np.asarray(model.predict(changed), dtype=float)
        method = "grouped permutation score sensitivity"
    explanations: list[list[dict[str, float | str]]] = []
    for contributions in values:
        totals: dict[str, float] = {}
        for group, contribution in zip(group_names, contributions, strict=True):
            totals[group] = totals.get(group, 0.0) + float(contribution)
        strongest = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        explanations.append(
            [{"group": group, "score_contribution": value} for group, value in strongest]
        )
    return method, explanations
