"""Full-order event metrics, calibration bins, and event bootstrap intervals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from f1forecast.probabilities import OrderSamples, normalized_order_nll


@dataclass(frozen=True)
class EventEvaluation:
    full_order: bool
    excluded_drivers: dict[str, str]
    normalized_nll: float | None
    position_mae: float | None
    win_brier: float | None
    podium_brier: float | None
    top_ten_brier: float | None
    calibration_rows: list[dict[str, float | int | str]]

    @property
    def probability_loss(self) -> float | None:
        return self.normalized_nll


def evaluate_event(
    samples: OrderSamples,
    actual_positions: Mapping[str, int | None],
    *,
    statuses: Mapping[str, str] | None = None,
) -> EventEvaluation:
    """Score complete orders; explicitly exclude DNS/DSQ/unclassified rows."""
    statuses = statuses or {}
    positions = [actual_positions.get(driver) for driver in samples.driver_ids]
    invalid_status = ("DNS", "DID NOT START", "DNQ", "DID NOT QUALIFY", "DSQ", "DISQUAL")
    excluded = {
        driver: statuses.get(driver, "unclassified")
        for driver, position in zip(samples.driver_ids, positions, strict=True)
        if (
            position is None
            or not isinstance(position, (int, np.integer))
            or position < 1
            or any(word in statuses.get(driver, "").upper() for word in invalid_status)
        )
    }
    n = len(samples.driver_ids)
    complete = not excluded and set(positions) == set(range(1, n + 1))
    if not complete:
        if not excluded:
            excluded = {
                driver: "incomplete or duplicate classification" for driver in samples.driver_ids
            }
        return EventEvaluation(False, excluded, None, None, None, None, None, [])
    true_positions = np.asarray(positions, dtype=int)

    def brier(probabilities: np.ndarray, threshold: int) -> float:
        return float(np.mean((probabilities - (true_positions <= threshold)) ** 2))

    calibration = []
    for outcome, probabilities, threshold in (
        ("win", samples.win_probabilities, 1),
        ("podium", samples.podium_probabilities, min(3, n)),
        ("top_ten", samples.top_ten_probabilities, min(10, n)),
    ):
        for driver, probability, actual in zip(
            samples.driver_ids, probabilities, true_positions, strict=True
        ):
            calibration.append(
                {
                    "driver_id": driver,
                    "outcome": outcome,
                    "probability": float(probability),
                    "observed": int(actual <= threshold),
                }
            )
    return EventEvaluation(
        full_order=True,
        excluded_drivers={},
        normalized_nll=normalized_order_nll(samples.scores, true_positions, samples.temperature),
        position_mae=float(np.mean(np.abs(samples.expected_positions - true_positions))),
        win_brier=brier(samples.win_probabilities, 1),
        podium_brier=brier(samples.podium_probabilities, min(3, n)),
        top_ten_brier=brier(samples.top_ten_probabilities, min(10, n)),
        calibration_rows=calibration,
    )


def aggregate_evaluations(
    events: Sequence[EventEvaluation], *, seed: int = 42, bootstrap_samples: int = 2_000
) -> dict[str, object]:
    """Average equally across complete events and bootstrap whole events."""
    usable = [event for event in events if event.full_order]
    metrics = ("normalized_nll", "position_mae", "win_brier", "podium_brier", "top_ten_brier")
    result: dict[str, object] = {
        "event_count": len(events),
        "evaluated_events": len(usable),
        "excluded_events": len(events) - len(usable),
    }
    if not usable:
        return result
    rng = np.random.default_rng(seed)
    for metric in metrics:
        values = np.asarray([getattr(event, metric) for event in usable], dtype=float)
        result[metric] = float(values.mean())
        if bootstrap_samples > 0:
            indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            distribution = values[indices].mean(axis=1)
            result[f"{metric}_95pct_ci"] = [
                float(x) for x in np.quantile(distribution, [0.025, 0.975])
            ]
    result["probability_loss"] = result["normalized_nll"]
    return result


def calibration_bins(
    events: Sequence[EventEvaluation], *, bins: int = 10
) -> list[dict[str, object]]:
    """Observed rate by predicted-probability bin for each outcome."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    rows = [row for event in events for row in event.calibration_rows]
    output: list[dict[str, object]] = []
    for outcome in ("win", "podium", "top_ten"):
        group = [row for row in rows if row["outcome"] == outcome]
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            selected = [
                row
                for row in group
                if lower <= float(row["probability"]) < upper
                or index == bins - 1
                and float(row["probability"]) == 1
            ]
            if selected:
                output.append(
                    {
                        "outcome": outcome,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "count": len(selected),
                        "mean_probability": float(
                            np.mean([float(row["probability"]) for row in selected])
                        ),
                        "observed_rate": float(
                            np.mean([float(row["observed"]) for row in selected])
                        ),
                    }
                )
    return output


def promotion_decision(learned: Mapping[str, float], baseline: Mapping[str, float]) -> bool:
    """Strict held-out probability gain with no position-error regression."""
    return (
        learned["probability_loss"] < baseline["probability_loss"]
        and learned["position_mae"] <= baseline["position_mae"]
    )
