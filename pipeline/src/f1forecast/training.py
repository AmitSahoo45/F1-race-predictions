"""Chronological ranking fits, calibration, backtests, and model artifacts."""

# Pyright's current pandas unions make ordinary DataFrame/Series operations
# ambiguous here; runtime schema checks and walk-forward tests cover these paths.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from f1forecast.evaluation import (
    EventEvaluation,
    aggregate_evaluations,
    calibration_bins,
    evaluate_event,
    promotion_decision,
)
from f1forecast.features import NUMERIC_FEATURES, completed_session_rows, missing_feature_groups
from f1forecast.modeling import fit_ranker, predict_scores
from f1forecast.probabilities import OrderSamples, fit_temperature, sample_orders


@dataclass(frozen=True)
class TrainedArtifact:
    model_path: Path
    metadata_path: Path
    temperature: float
    version: str
    model: CatBoostRanker


def _canonical_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(UTC)
        return stamp.isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        if pd.isna(value):
            return None
        return pd.Timedelta(value).isoformat()
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        if np.isnat(value):
            return None
        return _canonical_value(pd.Timestamp(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number if math.isfinite(number) else str(number)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"unsupported dataset value type: {type(value).__name__}")


def input_dataset_sha256(rows: pd.DataFrame) -> str:
    """Hash every supplied cell, independent of DataFrame row and column order."""
    identity = ("event_id", "target", "driver_id")
    if missing := set(identity) - set(rows.columns):
        raise ValueError(f"dataset identity columns missing: {sorted(missing)}")
    if not rows.columns.is_unique or not all(isinstance(name, str) for name in rows.columns):
        raise ValueError("dataset columns must be unique strings")
    columns = sorted(rows.columns)
    records = []
    for values in rows[columns].itertuples(index=False, name=None):
        record = {
            name: _canonical_value(value) for name, value in zip(columns, values, strict=True)
        }
        serialized = json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        records.append((*(str(record[name]) for name in identity), serialized))
    records.sort()
    payload = json.dumps(
        {"columns": columns, "rows": [record[-1] for record in records]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def validate_feature_cutoffs(rows: pd.DataFrame) -> None:
    """Reject any feature snapshot whose source or provenance crosses its cutoff."""
    needed = {"cutoff", "source_latest_at", "feature_provenance"}
    if missing := needed - set(rows.columns):
        raise ValueError(f"feature rows missing {sorted(missing)}")
    freeze = _time(rows["cutoff"])
    source = _time(rows["source_latest_at"])
    if freeze.isna().any():
        raise ValueError("invalid feature cutoff")
    if (source.notna() & source.gt(freeze)).any():
        raise ValueError("feature source occurs after its forecast cutoff")
    if not rows["feature_provenance"].isin(("prospective", "reconstructed")).all():
        raise ValueError("unknown feature provenance")


def validate_training_rows(rows: pd.DataFrame, training_cutoff: str | pd.Timestamp) -> None:
    """Reject feature, label, and provenance leaks before any model fit."""
    needed = {
        "event_id",
        "cutoff",
        "source_latest_at",
        "target_session_end",
        "weekend_complete_at",
        "feature_provenance",
    }
    if missing := needed - set(rows.columns):
        raise ValueError(f"training rows missing {sorted(missing)}")
    end = pd.Timestamp(training_cutoff)
    if end.tzinfo is None:
        raise ValueError("training cutoff must be timezone aware")
    validate_feature_cutoffs(rows)
    freeze = _time(rows["cutoff"])
    target_end = _time(rows["target_session_end"])
    weekend_end = _time(rows["weekend_complete_at"])
    if freeze.isna().any() or target_end.isna().any():
        raise ValueError("training requires valid target cutoff and end timestamps")
    if target_end.ge(end).any() or freeze.ge(end).any():
        raise ValueError("training requires complete target sessions before training cutoff")
    if weekend_end.isna().any() or weekend_end.ge(end).any() or weekend_end.lt(target_end).any():
        raise ValueError("training requires recorded completed weekend before training cutoff")
    if "position" in rows and rows["position"].isna().any():
        raise ValueError("training labels contain missing classifications")


def attach_results(feature_rows: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Attach finalized labels separately; never use them to form feature entrants."""
    needed = {
        "event_id",
        "session_id",
        "session_type",
        "driver_id",
        "session_end",
        "position",
        "status",
    }
    if missing := needed - set(summaries.columns):
        raise ValueError(f"result summaries missing {sorted(missing)}")
    summaries = completed_session_rows(summaries)
    race_rows = summaries.loc[
        summaries["session_type"].eq("R")
        & pd.to_numeric(summaries["position"], errors="coerce").notna()
    ]
    weekend_end = race_rows.groupby("event_id")["session_end"].max().rename("weekend_complete_at")
    labels = summaries[["session_id", "driver_id", "session_end", "position", "status"]].rename(
        columns={
            "session_id": "target_session_id",
            "session_end": "target_session_end",
        }
    )
    if labels.duplicated(["target_session_id", "driver_id"]).any():
        raise ValueError("duplicate target classifications")
    result = feature_rows.drop(
        columns=["position", "status", "target_session_end", "weekend_complete_at"], errors="ignore"
    ).merge(labels, on=["target_session_id", "driver_id"], how="left", validate="one_to_one")
    return result.join(weekend_end, on="event_id")


def _complete_events(rows: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for event, group in rows.groupby("event_id", sort=False):
        positions = pd.to_numeric(group["position"], errors="coerce")
        weekend_known = (
            "weekend_complete_at" in group and _time(group["weekend_complete_at"]).notna().all()
        )
        statuses = (
            group["status"].astype(str).str.upper()
            if "status" in group
            else pd.Series("", index=group.index)
        )
        invalid = statuses.str.contains(r"DNS|DID NOT START|DNQ|DID NOT QUALIFY|DSQ|DISQUAL")
        if (
            weekend_known
            and len(group) >= 2
            and not invalid.any()
            and positions.notna().all()
            and set(positions) == set(range(1, len(group) + 1))
        ):
            keep.append(event)
    return rows.loc[rows["event_id"].isin(keep)].copy()


def baseline_scores(rows: pd.DataFrame, target: str, name: str) -> pd.Series:
    """Field-relative simple baseline scores; larger values imply better rank."""
    column = {
        "recent_qualifying": "recent_qualifying_rank",
        "practice_pace": "practice_pace_gap_s",
        "qualifying_result": "qualifying_position",
    }
    if name == "blended":
        return cast(pd.Series, rows["baseline_score"].astype(float).fillna(0.5))
    allowed = (
        ("recent_qualifying", "practice_pace") if target == "qualifying" else ("qualifying_result",)
    )
    if name not in allowed:
        raise ValueError(f"unknown baseline {name!r} for {target}")
    values = pd.to_numeric(rows[column[name]], errors="coerce")
    return (1 - values.rank(pct=True, ascending=True)).fillna(0.5)


def _baseline_names(target: str) -> tuple[str, ...]:
    return (
        ("recent_qualifying", "practice_pace", "blended")
        if target == "qualifying"
        else ("qualifying_result", "blended")
    )


def _baseline_oof(rows: pd.DataFrame, target: str, name: str) -> pd.DataFrame:
    records = []
    for _, group in _event_groups(rows.loc[rows["season"].between(2022, 2024)], target):
        if _complete_events(group).empty:
            continue
        frame = group[["event_id", "driver_id", "position"]].copy()
        frame["event_end"] = group["weekend_complete_at"].iloc[0]
        frame["score"] = baseline_scores(group, target, name).to_numpy()
        frame["prediction_provenance"] = "chronological_oof"
        frame["feature_provenance"] = group["feature_provenance"].to_numpy()
        records.append(frame)
    if not records:
        return pd.DataFrame(
            columns=[
                "event_id",
                "driver_id",
                "position",
                "event_end",
                "score",
                "prediction_provenance",
                "feature_provenance",
            ]
        )
    return pd.concat(records, ignore_index=True)


def _select_baseline(
    rows: pd.DataFrame, target: str, *, n_samples: int, seed: int
) -> tuple[str, float, dict[str, dict[str, float | None]]]:
    """Choose on development only, with each development event calibrated on prior ones."""
    candidates: dict[str, dict[str, float | None]] = {}
    cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
    for name in _baseline_names(target):
        oof = _baseline_oof(rows, target, name)
        results = []
        for _, group in _event_groups(rows.loc[rows["season"].between(2022, 2024)], target):
            temperature, _ = _temp_from_prior_oof(oof, group["cutoff"].iloc[0])
            sample = sample_orders(
                group["driver_id"].tolist(),
                baseline_scores(group, target, name).to_numpy(),
                temperature=temperature,
                seed=seed,
                n_samples=n_samples,
            )
            actual = {
                str(row["driver_id"]): None if pd.isna(row["position"]) else int(row["position"])
                for _, row in group.iterrows()
            }
            statuses = {
                str(row["driver_id"]): str(row.get("status", "")) for _, row in group.iterrows()
            }
            results.append(_evaluate_completed_weekend(sample, actual, statuses, group))
        summary = aggregate_evaluations(results, seed=seed, bootstrap_samples=0)
        candidates[name] = {
            "probability_loss": summary.get("probability_loss"),
            "position_mae": summary.get("position_mae"),
        }
    valid = [
        (name, metrics)
        for name, metrics in candidates.items()
        if metrics["probability_loss"] is not None
    ]
    if valid:
        winner = min(
            valid,
            key=lambda item: (float(item[1]["probability_loss"]), float(item[1]["position_mae"])),
        )[0]
    else:
        winner = "recent_qualifying" if target == "qualifying" else "qualifying_result"
    all_oof = _baseline_oof(rows, target, winner)
    temp, _ = _temp_from_prior_oof(all_oof, cutoff)
    return winner, temp, candidates


def _event_groups(rows: pd.DataFrame, target: str) -> list[tuple[str, pd.DataFrame]]:
    if target not in {"qualifying", "race"}:
        raise ValueError("target must be qualifying or race")
    selected = rows.loc[rows["target"].eq(target)].copy()
    selected["cutoff"] = _time(selected["cutoff"])
    selected["target_session_end"] = _time(selected["target_session_end"])
    selected["weekend_complete_at"] = _time(selected["weekend_complete_at"])
    selected["source_latest_at"] = _time(selected["source_latest_at"])
    if selected.empty:
        raise ValueError(f"no rows for {target}")
    groups = list(selected.groupby("event_id", sort=False))
    groups.sort(key=lambda item: item[1]["cutoff"].min())
    for _, group in groups:
        if (
            group["cutoff"].nunique() != 1
            or group["target_session_end"].nunique(dropna=False) != 1
            or group["weekend_complete_at"].nunique(dropna=False) != 1
        ):
            raise ValueError("event target rows must share cutoff and session end")
    return groups


def _oof_for_calibration(
    rows: pd.DataFrame, target: str, *, iterations: int, min_train_events: int, seed: int
) -> pd.DataFrame:
    groups = _event_groups(rows.loc[rows["season"].between(2022, 2024)], target)
    oof = []
    for event, held in groups:
        earlier = rows.loc[
            (rows["target"].eq(target))
            & (_time(rows["weekend_complete_at"]).lt(held["cutoff"].min()))
            & rows["event_id"].ne(event)
            & rows["season"].between(2022, 2024)
        ]
        earlier = _complete_events(earlier)
        if earlier["event_id"].nunique() < min_train_events or not _complete_events(held).shape[0]:
            continue
        validate_training_rows(earlier, held["cutoff"].min())
        model = fit_ranker(earlier, iterations=iterations, random_seed=seed)
        prediction = held[["event_id", "driver_id", "position"]].copy()
        prediction["event_end"] = held["weekend_complete_at"].iloc[0]
        prediction["score"] = predict_scores(model, held)
        prediction["prediction_provenance"] = "chronological_oof"
        prediction["feature_provenance"] = held["feature_provenance"].to_numpy()
        oof.append(prediction)
    if not oof:
        return pd.DataFrame(
            columns=[
                "event_id",
                "driver_id",
                "position",
                "event_end",
                "score",
                "prediction_provenance",
                "feature_provenance",
            ]
        )
    return pd.concat(oof, ignore_index=True)


def _temp_from_prior_oof(oof: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[float, int]:
    eligible = oof.loc[_time(oof["event_end"]).lt(cutoff)]
    if eligible.empty:
        return 1.0, 0
    return fit_temperature(eligible, training_cutoff=cutoff), int(eligible["event_id"].nunique())


def _forecast_distribution(samples: OrderSamples) -> dict[str, Any]:
    """Keep the sampled position histogram, without the large raw-order array."""
    return {
        "driver_ids": list(samples.driver_ids),
        "scores": samples.scores.tolist(),
        "temperature": samples.temperature,
        "position_probabilities": samples.position_probabilities.tolist(),
        "expected_positions": samples.expected_positions.tolist(),
        "win_probabilities": samples.win_probabilities.tolist(),
        "podium_probabilities": samples.podium_probabilities.tolist(),
        "top_ten_probabilities": samples.top_ten_probabilities.tolist(),
        "sample_count": len(samples.orders),
        "sample_seed": samples.seed,
    }


def _fold_version(
    training_hash: str,
    target: str,
    cutoff: pd.Timestamp,
    training_events: list[str],
    iterations: int,
    min_train_events: int,
    seed: int,
) -> str:
    recipe = {
        "training_dataset_sha256": training_hash,
        "target": target,
        "cutoff": cutoff.isoformat(),
        "training_events": training_events,
        "iterations": iterations,
        "min_train_events": min_train_events,
        "seed": seed,
    }
    digest = sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{target}-fold-{digest[:12]}"


def _used_ids(held: pd.DataFrame, column: str) -> list[str]:
    if column not in held:
        return []
    return sorted(
        {
            str(item)
            for used in held[column]
            if isinstance(used, (tuple, list))
            for item in used
            if item is not None and not pd.isna(item)
        }
    )


def _evaluate_completed_weekend(
    samples: OrderSamples,
    actual: dict[str, int | None],
    statuses: dict[str, str],
    held: pd.DataFrame,
) -> EventEvaluation:
    weekend = _time(held["weekend_complete_at"])
    target_end = _time(held["target_session_end"])
    if weekend.isna().any() or target_end.isna().any() or not weekend.ge(target_end).all():
        return EventEvaluation(
            False,
            {driver: "Grand Prix weekend incomplete" for driver in samples.driver_ids},
            None,
            None,
            None,
            None,
            None,
            [],
        )
    return evaluate_event(samples, actual, statuses=statuses)


def walk_forward_backtest(
    rows: pd.DataFrame,
    target: str,
    *,
    min_train_events: int = 3,
    iterations: int = 300,
    seed: int = 42,
    n_samples: int = 20_000,
) -> dict[str, Any]:
    """Retrain after prior completed events and evaluate by whole weekend.

    2022-24 are development, 2025 is locked test, and 2026 is separate.
    Temperature for 2025/26 is learned only from earlier development OOF
    predictions; held-out results never choose a temperature.
    """
    dataset_hash = input_dataset_sha256(rows)
    validate_feature_cutoffs(rows)
    groups = _event_groups(rows, target)
    # Build development OOF as the chronological walk proceeds. A fold is fit
    # once, then its scores calibrate only later folds.
    oof = pd.DataFrame(
        columns=[
            "event_id",
            "driver_id",
            "position",
            "event_end",
            "score",
            "prediction_provenance",
            "feature_provenance",
        ]
    )
    baseline_name, fixed_baseline_temp, baseline_candidates = _select_baseline(
        rows, target, n_samples=n_samples, seed=seed
    )
    baseline_oof = _baseline_oof(rows, target, baseline_name)
    records: list[dict[str, Any]] = []
    by_period: dict[str, dict[str, list[EventEvaluation]]] = {
        name: {"learned": [], "baseline": []}
        for name in ("development_2022_2024", "locked_2025", "separate_2026")
    }
    paired_locked: dict[str, list[EventEvaluation]] = {"learned": [], "baseline": []}
    case_results: dict[tuple[str, ...], dict[str, Any]] = {}
    for event, held in groups:
        season = int(held["season"].iloc[0])
        period = (
            "development_2022_2024"
            if season <= 2024
            else "locked_2025"
            if season == 2025
            else "separate_2026"
        )
        if season < 2022 or season > 2026:
            continue
        cutoff = held["cutoff"].iloc[0]
        earlier = rows.loc[
            rows["target"].eq(target)
            & _time(rows["weekend_complete_at"]).lt(cutoff)
            & rows["event_id"].ne(event)
        ]
        earlier = _complete_events(earlier)
        actual = {
            str(row["driver_id"]): None if pd.isna(row["position"]) else int(row["position"])
            for _, row in held.iterrows()
        }
        status = {str(row["driver_id"]): str(row.get("status", "")) for _, row in held.iterrows()}
        missing_pattern = missing_feature_groups(held, target)
        training_events = sorted(str(value) for value in earlier["event_id"].unique())
        latest_complete = _time(earlier["weekend_complete_at"]).max()
        learned_forecast = None
        model_version = None
        training_hash = None
        if earlier["event_id"].nunique() >= min_train_events:
            validate_training_rows(earlier, cutoff)
            model = fit_ranker(earlier, iterations=iterations, random_seed=seed)
            scores = predict_scores(model, held)
            temp, calibration_events = _temp_from_prior_oof(oof, cutoff)
            learned_samples = sample_orders(
                held["driver_id"].tolist(),
                scores,
                temperature=temp,
                seed=seed,
                n_samples=n_samples,
            )
            learned_eval = _evaluate_completed_weekend(learned_samples, actual, status, held)
            learned_forecast = _forecast_distribution(learned_samples)
            training_hash = input_dataset_sha256(earlier)
            model_version = _fold_version(
                training_hash, target, cutoff, training_events, iterations, min_train_events, seed
            )
            if period == "development_2022_2024" and learned_eval.full_order:
                prediction = held[["event_id", "driver_id", "position"]].copy()
                prediction["event_end"] = held["weekend_complete_at"].iloc[0]
                prediction["score"] = scores
                prediction["prediction_provenance"] = "chronological_oof"
                prediction["feature_provenance"] = held["feature_provenance"].to_numpy()
                oof = prediction if oof.empty else pd.concat([oof, prediction], ignore_index=True)
        else:
            learned_eval = None
            calibration_events = 0
        baseline_temp = (
            _temp_from_prior_oof(baseline_oof, cutoff)[0]
            if period == "development_2022_2024"
            else fixed_baseline_temp
        )
        baseline_samples = sample_orders(
            held["driver_id"].tolist(),
            baseline_scores(held, target, baseline_name).to_numpy(),
            temperature=baseline_temp,
            seed=seed,
            n_samples=n_samples,
        )
        baseline_eval = _evaluate_completed_weekend(baseline_samples, actual, status, held)
        by_period[period]["baseline"].append(baseline_eval)
        case = case_results.setdefault(
            tuple(missing_pattern),
            {
                "baseline": [],
                "learned": [],
                "paired_baseline": [],
                "paired_learned": [],
                "baseline_event_ids": [],
                "learned_event_ids": [],
                "paired_event_ids": [],
            },
        )
        case["baseline"].append(baseline_eval)
        if baseline_eval.full_order:
            case["baseline_event_ids"].append(event)
        if learned_eval is not None:
            by_period[period]["learned"].append(learned_eval)
            case["learned"].append(learned_eval)
            if learned_eval.full_order:
                case["learned_event_ids"].append(event)
            if period == "locked_2025" and learned_eval.full_order and baseline_eval.full_order:
                paired_locked["learned"].append(learned_eval)
                paired_locked["baseline"].append(baseline_eval)
                case["paired_learned"].append(learned_eval)
                case["paired_baseline"].append(baseline_eval)
                case["paired_event_ids"].append(event)
        records.append(
            {
                "event_id": event,
                "season": season,
                "period": period,
                "target": target,
                "cutoff": cutoff.isoformat(),
                "target_session_end": held["target_session_end"].iloc[0].isoformat()
                if pd.notna(held["target_session_end"].iloc[0])
                else None,
                "training_cutoff": (
                    latest_complete.isoformat() if pd.notna(latest_complete) else None
                ),
                "training_latest_completed_at": (
                    latest_complete.isoformat() if pd.notna(latest_complete) else None
                ),
                "training_events": int(earlier["event_id"].nunique()),
                "training_event_ids": training_events,
                "model_version": model_version,
                "training_dataset_sha256": training_hash,
                "source_snapshot_ids": _used_ids(held, "used_snapshot_ids"),
                "used_session_ids": _used_ids(held, "used_session_ids"),
                "calibration_oof_events": calibration_events,
                "baseline_name": baseline_name,
                "baseline_temperature": baseline_temp,
                "baseline_forecast": _forecast_distribution(baseline_samples),
                "learned_forecast": learned_forecast,
                "missing_pattern": missing_pattern,
                "full_order": baseline_eval.full_order,
                "excluded_drivers": baseline_eval.excluded_drivers,
                "baseline": _event_metrics(baseline_eval),
                "learned": _event_metrics(learned_eval) if learned_eval is not None else None,
            }
        )
    summaries = {}
    for period, models in by_period.items():
        summaries[period] = {}
        for name, event_results in models.items():
            summary = aggregate_evaluations(event_results, seed=seed)
            summary["calibration_bins"] = calibration_bins(event_results)
            summaries[period][name] = summary
    learned = aggregate_evaluations(paired_locked["learned"], seed=seed)
    baseline = aggregate_evaluations(paired_locked["baseline"], seed=seed)
    promotion = (
        cast(int, learned["evaluated_events"]) > 0
        and cast(int, baseline["evaluated_events"]) > 0
        and promotion_decision(learned, baseline)
    )
    missing_case_evidence = []
    for pattern, case in sorted(case_results.items()):
        case_baseline = aggregate_evaluations(case["baseline"], seed=seed)
        case_learned = aggregate_evaluations(case["learned"], seed=seed)
        case_paired_baseline = aggregate_evaluations(case["paired_baseline"], seed=seed)
        case_paired_learned = aggregate_evaluations(case["paired_learned"], seed=seed)
        case_gate = (
            cast(int, case_paired_learned["evaluated_events"]) > 0
            and cast(int, case_paired_baseline["evaluated_events"]) > 0
            and promotion_decision(case_paired_learned, case_paired_baseline)
        )
        missing_case_evidence.append(
            {
                "pattern": list(pattern),
                "baseline": case_baseline,
                "learned": case_learned,
                "baseline_evaluated_event_count": cast(int, case_baseline["evaluated_events"]),
                "learned_evaluated_event_count": cast(int, case_learned["evaluated_events"]),
                "baseline_event_ids": case["baseline_event_ids"],
                "learned_event_ids": case["learned_event_ids"],
                "paired_locked_2025": {
                    "baseline": case_paired_baseline,
                    "learned": case_paired_learned,
                    "event_ids": case["paired_event_ids"],
                    "promotion_passed": case_gate,
                },
                "paired_2025_promotion_passed": case_gate,
            }
        )
    return {
        "target": target,
        "input_dataset_sha256": dataset_hash,
        "events": records,
        "periods": summaries,
        "promotion_passed": promotion,
        "calibration_source": "2022-2024 chronological OOF",
        "training_config": {
            "iterations": iterations,
            "random_seed": seed,
            "ranking_objective": "PairLogit",
            "feature_columns": list(NUMERIC_FEATURES),
            "min_train_events": min_train_events,
        },
        "baseline_name": baseline_name,
        "baseline_temperature": fixed_baseline_temp,
        "baseline_development_candidates": baseline_candidates,
        "baseline_selection_provenance": "posthoc development 2022-2024",
        "promotion_paired_2025": {"learned": learned, "baseline": baseline},
        "missing_case_evidence": missing_case_evidence,
        "prediction_provenance": "historical reconstruction",
    }


def _event_metrics(event: EventEvaluation) -> dict[str, float | None]:
    return {
        name: getattr(event, name)
        for name in ("normalized_nll", "position_mae", "win_brier", "podium_brier", "top_ten_brier")
    }


def fit_final_model(
    rows: pd.DataFrame,
    target: str,
    *,
    training_cutoff: str | pd.Timestamp,
    output_dir: str | Path,
    iterations: int = 300,
    min_train_events: int = 3,
    seed: int = 42,
) -> TrainedArtifact:
    """Save a target ranker plus auditable training and calibration metadata."""
    dataset_hash = input_dataset_sha256(rows)
    validate_feature_cutoffs(rows)
    cutoff = pd.Timestamp(training_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("training cutoff must be timezone aware")
    chosen = rows.loc[
        rows["target"].eq(target) & _time(rows["weekend_complete_at"]).lt(cutoff)
    ].copy()
    chosen = _complete_events(chosen)
    if chosen["event_id"].nunique() < min_train_events:
        raise ValueError("insufficient complete prior events")
    validate_training_rows(chosen, cutoff)
    oof = _oof_for_calibration(
        chosen, target, iterations=iterations, min_train_events=min_train_events, seed=seed
    )
    temp, oof_events = _temp_from_prior_oof(oof, cutoff)
    baseline_name, baseline_temp, baseline_candidates = _select_baseline(
        chosen, target, n_samples=20_000, seed=seed
    )
    model = fit_ranker(chosen, iterations=iterations, random_seed=seed)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{target}.cbm"
    model.save_model(str(model_path))
    digest = sha256(model_path.read_bytes()).hexdigest()
    version = f"{target}-{digest[:12]}"
    medians = {
        name: float(value) if pd.notna(value) else None
        for name, value in chosen[list(NUMERIC_FEATURES)].median(numeric_only=True).items()
    }
    metadata = {
        "version": version,
        "target": target,
        "training_cutoff": cutoff.isoformat(),
        "input_dataset_sha256": dataset_hash,
        "training_event_ids": sorted(chosen["event_id"].unique().tolist()),
        "training_row_count": len(chosen),
        "feature_provenance": sorted(chosen["feature_provenance"].unique().tolist()),
        "feature_columns": list(NUMERIC_FEATURES),
        "reference_medians": medians,
        "ranking_objective": "PairLogit",
        "iterations": iterations,
        "random_seed": seed,
        "training_config": {
            "iterations": iterations,
            "random_seed": seed,
            "ranking_objective": "PairLogit",
            "feature_columns": list(NUMERIC_FEATURES),
            "min_train_events": min_train_events,
        },
        "temperature": temp,
        "temperature_bounds": [0.25, 4.0],
        "calibration_source": "2022-2024 chronological OOF"
        if oof_events
        else "uncalibrated default",
        "calibration_oof_events": oof_events,
        "baseline_name": baseline_name,
        "baseline_temperature": baseline_temp,
        "baseline_development_candidates": baseline_candidates,
        "sha256": digest,
    }
    metadata_path = directory / f"{target}.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return TrainedArtifact(model_path, metadata_path, temp, version, model)
