"""Cutoff-aware, one-driver-per-target features from session summaries."""

# Pyright's current pandas unions make ordinary DataFrame/Series operations
# ambiguous here; runtime schema checks and feature tests cover these paths.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .identity import canonical_team_id
from .lap_summaries import COMPOUND_FEATURES

SESSION_COLUMNS = (
    "event_id",
    "season",
    "circuit_id",
    "driver_id",
    "team_id",
    "session_id",
    "session_type",
    "session_end",
    "observed_at",
    "available_at",
    "provenance",
)
TARGET_COLUMNS = (
    "event_id",
    "season",
    "circuit_id",
    "driver_id",
    "team_id",
    "target",
    "target_session_id",
    "cutoff",
)
NUMERIC_FEATURES = (
    "recent_qualifying_rank",
    "recent_qualifying_count",
    "recent_race_rank",
    "recent_race_count",
    "recent_team_qualifying_rank",
    "recent_team_race_rank",
    "circuit_qualifying_rank",
    "circuit_race_rank",
    "circuit_history_count",
    "practice_pace_gap_s",
    "long_run_pace_gap_s",
    "consistency_s",
    "usable_laps",
    "tyre_age",
    *COMPOUND_FEATURES,
    "mean_speed",
    "mean_throttle",
    "brake_fraction",
    "air_temp",
    "track_temp",
    "practice_sessions",
    "qualifying_position",
    "qualifying_gap_s",
)

MISSING_GROUP_BY_FEATURE = {
    "recent_qualifying_rank": "missing-recent-qualifying-form",
    "recent_qualifying_count": "missing-history-count",
    "recent_race_rank": "missing-recent-race-form",
    "recent_race_count": "missing-history-count",
    "recent_team_qualifying_rank": "missing-team-form",
    "recent_team_race_rank": "missing-team-form",
    "circuit_qualifying_rank": "missing-circuit-history",
    "circuit_race_rank": "missing-circuit-history",
    "circuit_history_count": "missing-history-count",
    "practice_pace_gap_s": "missing-practice-pace",
    "long_run_pace_gap_s": "missing-long-run-pace",
    "consistency_s": "missing-stint-consistency",
    "usable_laps": "missing-practice-coverage",
    "tyre_age": "missing-tyre-context",
    **dict.fromkeys(COMPOUND_FEATURES, "missing-tyre-context"),
    "mean_speed": "missing-telemetry",
    "mean_throttle": "missing-telemetry",
    "brake_fraction": "missing-telemetry",
    "air_temp": "missing-observed-conditions",
    "track_temp": "missing-observed-conditions",
    "practice_sessions": "missing-practice-coverage",
    "qualifying_position": "missing-qualifying-result",
    "qualifying_gap_s": "missing-qualifying-gap",
}


def completed_session_rows(summaries: pd.DataFrame) -> pd.DataFrame:
    """Use finalized source sessions when the archive provides completion proof.

    Caller-supplied frames without ``session_complete`` retain the historical
    API contract: their caller is responsible for asserting completion.
    """
    if "session_complete" not in summaries:
        return summaries
    complete = summaries["session_complete"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    return summaries.loc[complete].copy()


def missing_feature_groups(rows: pd.DataFrame, target: str) -> list[str]:
    """Return a stable pattern for any missing model input in this target field."""
    if target not in {"qualifying", "race"}:
        raise ValueError("target must be qualifying or race")
    if rows.empty:
        raise ValueError("at least one entrant is required")
    flags: set[str] = set()
    structural_for_qualifying = {"qualifying_position", "qualifying_gap_s"}
    required_features = (
        set(NUMERIC_FEATURES) - structural_for_qualifying
        if target == "qualifying"
        else set(NUMERIC_FEATURES)
    )
    absent_columns = required_features - set(rows.columns)
    if absent_columns:
        flags.add("missing-feature-schema")
    dependent_counts = {
        "recent_qualifying_rank": "recent_qualifying_count",
        "recent_race_rank": "recent_race_count",
        "circuit_qualifying_rank": "circuit_history_count",
        "circuit_race_rank": "circuit_history_count",
    }
    for name in NUMERIC_FEATURES:
        if target == "qualifying" and name in structural_for_qualifying:
            continue
        if name not in rows:
            flags.add(MISSING_GROUP_BY_FEATURE[name])
            continue
        values = pd.to_numeric(rows[name], errors="coerce")
        if values.isna().any():
            count_name = dependent_counts.get(name)
            missing = values.isna()
            if count_name in rows:
                count = pd.to_numeric(rows[count_name], errors="coerce")
                missing &= count.ne(0)
            if missing.any():
                flags.add(MISSING_GROUP_BY_FEATURE[name])
    for count_name, zero_flag in (
        ("recent_qualifying_count", "new-driver"),
        ("recent_race_count", "new-race-history"),
        ("circuit_history_count", "no-circuit-history"),
        ("practice_sessions", "missing-practice"),
    ):
        if count_name in rows and pd.to_numeric(rows[count_name], errors="coerce").eq(0).any():
            flags.add(zero_flag)
    return sorted(flags)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _mean(values: Iterable[object]) -> float:
    data = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return float(data.mean()) if data.notna().any() else float("nan")


def _recent_rank(
    rows: pd.DataFrame,
    session_type: str,
    driver: str | None = None,
    team: str | None = None,
    circuit: str | None = None,
) -> tuple[float, int]:
    group = rows.loc[rows["session_type"].eq(session_type)]
    if circuit is not None:
        group = group.loc[group["circuit_id"].eq(circuit)]
    if driver is not None:
        group = group.loc[group["driver_id"].eq(driver)]
    if team is not None:
        group = group.loc[group["team_id"].eq(team)]
    group = group.sort_values("session_end").tail(5)
    return _mean(group["rank_fraction"]), len(group)


def _practice_features(eligible: pd.DataFrame, driver: str) -> dict[str, float | int]:
    practices = eligible.loc[eligible["session_type"].isin(("FP1", "FP2", "FP3"))].copy()
    result: dict[str, float | int] = {"practice_sessions": 0}
    if practices.empty:
        return result
    for source, target in (
        ("pace_s", "practice_pace_gap_s"),
        ("long_run_pace_s", "long_run_pace_gap_s"),
    ):
        practices[target] = pd.to_numeric(practices[source], errors="coerce")
        practices[target] -= practices.groupby("session_id")[target].transform("median")
    own = practices.loc[practices["driver_id"].eq(driver)]
    result["practice_sessions"] = int(own["session_id"].nunique())
    for name in (
        "practice_pace_gap_s",
        "long_run_pace_gap_s",
        "consistency_s",
        "tyre_age",
        "mean_speed",
        "mean_throttle",
        "brake_fraction",
        "air_temp",
        "track_temp",
    ):
        result[name] = _mean(own[name])
    result["usable_laps"] = float(
        pd.to_numeric(own["usable_laps"], errors="coerce").sum(min_count=1)
    )
    # Compound mix describes all usable practice laps, not the average of
    # differently sized sessions. Unknown source compounds stay unavailable.
    weights = pd.to_numeric(own["usable_laps"], errors="coerce")
    contributing = own.loc[weights.gt(0), list(COMPOUND_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    compound_known = (
        weights.notna().all() and not contributing.empty and contributing.notna().all().all()
    )
    for name in COMPOUND_FEATURES:
        result[name] = (
            float(contributing[name].mul(weights).sum() / weights.sum())
            if compound_known else float("nan")
        )
    return result


def _baseline_score(frame: pd.DataFrame, target: str) -> pd.Series:
    def rank_as_good(column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        return 1 - values.rank(pct=True, ascending=True)

    if target == "qualifying":
        parts = (("recent_qualifying_rank", 0.6), ("practice_pace_gap_s", 0.4))
    else:
        parts = (("qualifying_position", 0.7), ("recent_race_rank", 0.3))
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for feature, weight in parts:
        value = rank_as_good(feature)
        good = value.notna()
        numerator += value.fillna(0) * weight
        denominator += good.astype(float) * weight
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.5)


def build_feature_rows(
    summaries: pd.DataFrame, targets: pd.DataFrame, *, mode: str = "prospective"
) -> pd.DataFrame:
    """Create cutoff-safe rows. Reconstruction infers availability at session end.

    ``targets`` supplies the pre-session entrants; do not derive them from eventual
    classification rows. In reconstruction mode the resulting rows must remain
    labelled reconstructed, because archived feeds may contain later corrections.
    """
    if mode not in {"prospective", "reconstructed"}:
        raise ValueError("mode must be prospective or reconstructed")
    missing_targets = set(TARGET_COLUMNS) - set(targets.columns)
    if missing_targets:
        raise ValueError(f"targets missing columns: {sorted(missing_targets)}")
    if not summaries.empty:
        missing = set(SESSION_COLUMNS) - set(summaries.columns)
        if missing:
            raise ValueError(f"summaries missing columns: {sorted(missing)}")
    src = completed_session_rows(summaries).copy()
    for name in SESSION_COLUMNS:
        if name not in src:
            src[name] = pd.Series(dtype="object")
    src["team_id"] = [
        canonical_team_id(team, int(season))
        for team, season in zip(src["team_id"], src["season"], strict=True)
    ]
    for name in ("session_end", "observed_at", "available_at"):
        if name not in src:
            src[name] = pd.Series(dtype="datetime64[ns, UTC]")
        src[name] = _utc(src[name])
    for name in (
        "position",
        "pace_s",
        "long_run_pace_s",
        "consistency_s",
        "usable_laps",
        "tyre_age",
        *COMPOUND_FEATURES,
        "mean_speed",
        "mean_throttle",
        "brake_fraction",
        "air_temp",
        "track_temp",
        "qualifying_gap",
    ):
        if name not in src:
            src[name] = np.nan
    if "throttle" in src and src["mean_throttle"].isna().all():
        src["mean_throttle"] = src["throttle"]
    src["position"] = pd.to_numeric(src["position"], errors="coerce")
    target_rows = targets.copy()
    target_rows["team_id"] = [
        canonical_team_id(team, int(season))
        for team, season in zip(target_rows["team_id"], target_rows["season"], strict=True)
    ]
    target_rows["cutoff"] = _utc(target_rows["cutoff"])
    if target_rows["cutoff"].isna().any():
        raise ValueError("invalid target cutoff")
    if target_rows.duplicated(["event_id", "target", "driver_id"]).any():
        raise ValueError("duplicate target entrant")
    if not target_rows["target"].isin(("qualifying", "race")).all():
        raise ValueError("target must be qualifying or race")
    output: list[dict[str, object]] = []
    group_keys = ["event_id", "target", "target_session_id", "cutoff"]
    for (event_id, target, session_id, cutoff), entrants in target_rows.groupby(
        group_keys, sort=False
    ):
        candidate = src.loc[src["session_end"].le(cutoff) & src["session_id"].ne(session_id)]
        if mode == "prospective":
            candidate = candidate.loc[
                candidate["observed_at"].le(cutoff) & candidate["available_at"].le(cutoff)
            ]
        else:
            candidate = candidate.loc[candidate["provenance"].eq("retrospective")]
        past = candidate.loc[
            candidate["event_id"].ne(event_id) & candidate["session_type"].isin(("Q", "R"))
        ].copy()
        past["field_size"] = past.groupby("session_id")["position"].transform("max")
        past["rank_fraction"] = (past["position"] - 1) / (past["field_size"] - 1).clip(lower=1)
        past = past.loc[past["position"].gt(0) & past["rank_fraction"].notna()]
        current = candidate.loc[
            candidate["event_id"].eq(event_id)
            & candidate["session_type"].isin(("FP1", "FP2", "FP3", "Q"))
        ]
        if target == "qualifying":
            current = current.loc[current["session_type"].ne("Q")]
        used = pd.concat((past, current), ignore_index=True)
        latest = used["session_end"].max() if not used.empty else pd.NaT
        for item in entrants.to_dict("records"):
            driver = str(item["driver_id"])
            team = item["team_id"]
            circuit = str(item["circuit_id"])
            q_rank, q_count = _recent_rank(past, "Q", driver=driver)
            r_rank, r_count = _recent_rank(past, "R", driver=driver)
            team_q = _recent_rank(past, "Q", team=team)[0] if team is not None else np.nan
            team_r = _recent_rank(past, "R", team=team)[0] if team is not None else np.nan
            circuit_q, circuit_q_count = _recent_rank(past, "Q", driver=driver, circuit=circuit)
            circuit_r, circuit_r_count = _recent_rank(past, "R", driver=driver, circuit=circuit)
            own_q = (
                current.loc[current["session_type"].eq("Q") & current["driver_id"].eq(driver)]
                .sort_values("session_end")
                .tail(1)
            )
            record: dict[str, object] = dict(item)
            record.update(
                {
                    "recent_qualifying_rank": q_rank,
                    "recent_qualifying_count": q_count,
                    "recent_race_rank": r_rank,
                    "recent_race_count": r_count,
                    "recent_team_qualifying_rank": team_q,
                    "recent_team_race_rank": team_r,
                    "circuit_qualifying_rank": circuit_q,
                    "circuit_race_rank": circuit_r,
                    "circuit_history_count": circuit_q_count + circuit_r_count,
                    "qualifying_position": _mean(own_q["position"]),
                    "qualifying_gap_s": _mean(own_q["qualifying_gap"]),
                    "source_latest_at": latest,
                    "feature_provenance": mode,
                    "used_session_ids": tuple(sorted(used["session_id"].dropna().unique())),
                    "used_snapshot_ids": tuple(sorted(used["source_snapshot_id"].dropna().unique()))
                    if "source_snapshot_id" in used
                    else (),
                }
            )
            record.update(_practice_features(current, driver))
            output.append(record)
    result = pd.DataFrame(output)
    if result.empty:
        return result
    for name in NUMERIC_FEATURES:
        if name not in result:
            result[name] = np.nan
    result["baseline_score"] = np.nan
    for (_, target), group in result.groupby(["event_id", "target"]):
        result.loc[group.index, "baseline_score"] = _baseline_score(group, target)
    return result
