"""Plackett-Luce probabilities from one fixed-seed set of complete orders."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class OrderSamples:
    driver_ids: tuple[str, ...]
    scores: np.ndarray
    position_probabilities: np.ndarray
    expected_positions: np.ndarray
    win_probabilities: np.ndarray
    podium_probabilities: np.ndarray
    top_ten_probabilities: np.ndarray
    orders: np.ndarray
    seed: int
    temperature: float


def sample_orders(
    driver_ids: list[str] | tuple[str, ...],
    scores: list[float] | np.ndarray,
    *,
    temperature: float = 1.0,
    seed: int = 42,
    n_samples: int = 20_000,
) -> OrderSamples:
    """Sample complete PL orders via the equivalent Gumbel random utility model."""
    ids = tuple(str(value) for value in driver_ids)
    logits = np.asarray(scores, dtype=float)
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(logits):
        raise ValueError("driver IDs and scores must be nonempty, unique, and aligned")
    if not np.isfinite(logits).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("scores must be finite and temperature positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    utility = logits / temperature
    utility -= utility.max()
    rng = np.random.default_rng(seed)
    gumbels = rng.gumbel(size=(n_samples, len(ids)))
    orders = np.argsort(-(utility + gumbels), axis=1)
    positions = np.empty_like(orders)
    positions[np.arange(n_samples)[:, None], orders] = np.arange(1, len(ids) + 1)
    counts = np.stack(
        [np.bincount(positions[:, index], minlength=len(ids) + 1)[1:] for index in range(len(ids))]
    )
    probabilities = counts / n_samples
    return OrderSamples(
        driver_ids=ids,
        scores=logits.copy(),
        position_probabilities=probabilities,
        expected_positions=probabilities @ np.arange(1, len(ids) + 1),
        win_probabilities=probabilities[:, 0],
        podium_probabilities=probabilities[:, : min(3, len(ids))].sum(axis=1),
        top_ten_probabilities=probabilities[:, : min(10, len(ids))].sum(axis=1),
        orders=orders,
        seed=seed,
        temperature=float(temperature),
    )


def normalized_order_nll(
    scores: list[float] | np.ndarray, positions: list[int] | np.ndarray, temperature: float
) -> float:
    """Exact full-order PL loss divided by uniform-order loss, log(n!)."""
    values = np.asarray(scores, dtype=float)
    places = np.asarray(positions, dtype=float)
    n = len(values)
    if n < 2 or len(places) != n or not np.isfinite(values).all() or not np.isfinite(places).all():
        raise ValueError("complete scores and positions required")
    if set(places) != set(range(1, n + 1)):
        raise ValueError("positions must form a complete order")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    ordered = values[np.argsort(places)] / temperature
    # All suffix denominators in one reverse pass; calibration evaluates this
    # thousands of times across chronological folds.
    suffix_logsumexp = np.logaddexp.accumulate(ordered[::-1])[::-1]
    loss = np.sum(suffix_logsumexp[:-1] - ordered[:-1])
    return float(loss / lgamma(n + 1))


def fit_temperature(
    oof_predictions: pd.DataFrame,
    *,
    training_cutoff: str | pd.Timestamp,
    bounds: tuple[float, float] = (0.25, 4.0),
) -> float:
    """Fit bounded temperature only on complete earlier out-of-fold events."""
    required = {
        "event_id",
        "event_end",
        "score",
        "position",
        "prediction_provenance",
        "feature_provenance",
    }
    if missing := required - set(oof_predictions.columns):
        raise ValueError(f"OOF predictions missing {sorted(missing)}")
    if oof_predictions.empty:
        raise ValueError("OOF predictions required")
    if (
        not np.asarray(
            oof_predictions["prediction_provenance"].eq("chronological_oof"), dtype=bool
        ).all()
        or not np.asarray(
            oof_predictions["feature_provenance"].isin(("prospective", "reconstructed")), dtype=bool
        ).all()
    ):
        raise ValueError("calibration provenance must be chronological OOF with known features")
    cutoff = pd.Timestamp(training_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("training cutoff must be timezone aware")
    ends = pd.to_datetime(oof_predictions["event_end"], utc=True, errors="coerce")
    if ends.isna().any() or ends.ge(cutoff).any():
        raise ValueError("calibration events must finish before training cutoff")
    groups = []
    for _, group in oof_predictions.groupby("event_id"):
        scores = group["score"].to_numpy(dtype=float)
        positions = group["position"].to_numpy(dtype=float)
        # Invalid or partial events cannot provide a full-order likelihood.
        if len(scores) >= 2 and set(positions) == set(range(1, len(scores) + 1)):
            groups.append((scores, positions))
    if not groups:
        raise ValueError("no complete OOF events for temperature calibration")
    result = minimize_scalar(
        lambda value: float(
            np.mean([normalized_order_nll(scores, places, value) for scores, places in groups])
        ),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError("temperature fitting failed")
    return float(result.x)
