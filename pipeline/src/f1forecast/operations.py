"""Weekend inference. Provider delays and missing data fail closed."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRanker

from .archive import load_summaries
from .contracts import Event, Forecast, SiteData, TargetState
from .features import build_feature_rows, completed_session_rows, missing_feature_groups
from .identity import canonical_team_id
from .modeling import predict_scores, score_explanations
from .probabilities import sample_orders
from .publication import publish_forecast, publish_site, publish_telemetry, verify_archive
from .registry import (
    ModelRecord,
    ModelRegistry,
    covers_case,
    fallback_model,
    select_model,
    verified_path,
)

RESULT_RETRY_WINDOW = timedelta(days=7)


class UnvalidatedCase(ValueError):
    """Optional inputs are missing in a pattern the release was not evaluated on."""


def eligible_summary_versions(summaries: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Select whole eligible snapshots so revisions cannot leave phantom entrants."""
    if summaries.empty:
        return summaries
    eligible = completed_session_rows(summaries).copy()
    eligible["available_at"] = pd.to_datetime(eligible.available_at, utc=True)
    eligible = eligible.loc[eligible.available_at.le(cutoff)]
    if "source_snapshot_id" in eligible:
        identified = eligible.loc[
            eligible["source_snapshot_id"].map(lambda value: isinstance(value, str) and bool(value))
        ]
        latest = (
            identified.sort_values(["available_at", "source_snapshot_id"])
            .drop_duplicates("session_id", keep="last")[["session_id", "source_snapshot_id"]]
        )
        selected = identified.merge(
            latest, on=["session_id", "source_snapshot_id"], how="inner", validate="many_to_one"
        )
        # Legacy caller frames have no immutable snapshot identifier. Retain
        # their row-based behavior only for sessions with no identified version.
        legacy = eligible.loc[~eligible["session_id"].isin(latest["session_id"])]
        eligible = pd.concat([selected, legacy], ignore_index=True)
    return eligible.sort_values("available_at").drop_duplicates(
        ["session_id", "driver_id"], keep="last"
    )


def result_reconciliation_due(
    site: SiteData, event: Event, target: TargetState, now: datetime
) -> bool:
    if target.state != "issued" or not target.forecast_id:
        return False
    session = next(s for s in event.sessions if s.id == target.session_id)
    if not session.end <= now <= session.end + RESULT_RETRY_WINDOW:
        return False
    existing = next((a for a in site.analyses if a.event_id == event.id), None)
    return existing is None or not any(a.target == target.target for a in existing.actuals)


def publication_window(target: TargetState, now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("clock must be timezone-aware")
    if now > target.cutoff_at:
        return "missed"
    return "ready" if now >= target.cutoff_at - timedelta(minutes=30) else "early"


def has_weekend_work(site: SiteData, now: datetime) -> bool:
    for event in site.events:
        if not event.id.startswith(f"{event.season}-"):
            continue
        for target in event.targets:
            if result_reconciliation_due(site, event, target, now):
                return True
            session = next(s for s in event.sessions if s.id == target.session_id)
            if target.cutoff_at - timedelta(hours=1) <= now <= session.end + timedelta(hours=24):
                return True
    return False


def target_rows(event: Event, target: str) -> pd.DataFrame:
    state = next(t for t in event.targets if t.target == target)
    entrants = [d for d in event.entrants if not state.entrant_ids or d.id in state.entrant_ids]
    return pd.DataFrame(
        [
            {
                "event_id": event.id,
                "season": event.season,
                "circuit_id": event.circuit,
                "driver_id": d.id,
                "team_id": canonical_team_id(d.team, event.season),
                "target": target,
                "target_session_id": state.session_id,
                "cutoff": state.cutoff_at,
            }
            for d in entrants
        ]
    )


def validate_inputs(
    rows: pd.DataFrame, target: str, validated_patterns: list[list[str]]
) -> list[str]:
    if len(rows) < 2:
        raise ValueError("required entrants are unavailable")
    if target == "race" and (
        "qualifying_position" not in rows
        or bool(np.asarray(rows["qualifying_position"].isna()).any())
    ):
        raise ValueError("required qualifying positions are unavailable for one or more entrants")
    if target == "qualifying" and (
        "practice_pace_gap_s" not in rows or rows["practice_pace_gap_s"].notna().sum() < 2
    ):
        raise ValueError("required practice pace is unavailable")
    flags = missing_feature_groups(rows, target)
    if "missing-feature-schema" in flags:
        raise ValueError("required feature schema is incomplete")
    if not covers_case(validated_patterns, flags):
        raise UnvalidatedCase("unvalidated missing-data case: " + ", ".join(flags))
    return flags


def make_forecast(
    event: Event,
    target: TargetState,
    rows: pd.DataFrame,
    record: ModelRecord,
    registry_root: Path,
    *,
    kind: str = "issued",
    now: datetime | None = None,
    fallback_for: ModelRecord | None = None,
) -> Forecast:
    now = now or datetime.now(UTC)
    metadata = json.loads(
        verified_path(registry_root, record.metadata_path, record.hashes).read_text(
            encoding="utf-8"
        )
    )
    flags = validate_inputs(rows, target.target, record.validated_missing_patterns)
    if target.roster_source and target.roster_source.endswith("provisional roster"):
        flags.append("roster-from-latest-session")
    explanations = {}
    if record.kind == "catboost":
        model = CatBoostRanker()
        model.load_model(str(verified_path(registry_root, record.model_path, record.hashes)))
        scores = predict_scores(model, rows)
        _, contributions = score_explanations(
            model, rows, reference_medians=metadata.get("reference_medians")
        )
        explanations = {
            driver: [
                {"group": group["group"], "contribution": group["score_contribution"]}
                for group in groups
            ]
            for driver, groups in zip(rows.driver_id, contributions, strict=True)
        }
    else:
        from .training import baseline_scores

        scores = baseline_scores(rows, target.target, record.baseline_name).to_numpy(dtype=float)
    samples = sample_orders(rows.driver_id.tolist(), scores, temperature=record.temperature)
    predictions = []
    for rank, i in enumerate(np.argsort(samples.expected_positions, kind="stable"), 1):
        probs = samples.position_probabilities[i]
        driver_id = rows.driver_id.iloc[i]
        groups = explanations.get(driver_id, [])
        if isinstance(groups, dict):
            groups = groups.get("groups", [])
        predictions.append(
            {
                "driver_id": driver_id,
                "rank": rank,
                "expected_position": float(samples.expected_positions[i]),
                "p10": int(np.searchsorted(np.cumsum(probs), 0.1)) + 1,
                "p90": int(np.searchsorted(np.cumsum(probs), 0.9)) + 1,
                "win_probability": float(probs[0]),
                "podium_probability": float(sum(probs[:3])),
                "top10_probability": float(sum(probs[:10])),
                "position_probabilities": probs.tolist(),
                "explanations": groups,
            }
        )
    available = sorted(
        {
            session
            for sessions in rows.used_session_ids
            for session in sessions
            if session.startswith(event.id + "-")
        }
    )
    snapshot_ids = (
        sorted({str(snapshot) for snapshots in rows["used_snapshot_ids"] for snapshot in snapshots})
        if "used_snapshot_ids" in rows
        else []
    )
    eligible = [
        s.id for s in event.sessions if s.end <= target.cutoff_at and s.id != target.session_id
    ]
    missing = sorted(set(eligible) - set(available))
    summary = (
        "Uses "
        + (", ".join(s.rsplit("-", 1)[-1] for s in available) or "historical form")
        + ("; coverage notes: " + ", ".join(flags) if flags else ".")
    )
    if fallback_for is not None:
        summary += (
            f". Baseline used: CatBoost release {fallback_for.version} has not been validated"
            " for this missing-data case."
        )
    return Forecast.model_validate(
        {
            "id": f"{event.id}-{target.target}-{kind}",
            "event_id": event.id,
            "target": target.target,
            "kind": kind,
            "generated_at": now,
            "cutoff_at": target.cutoff_at,
            "training_cutoff": record.training_cutoff,
            "model_version": record.version,
            "input_snapshot_ids": ["history-sha256:" + record.hashes[record.history_path]]
            + snapshot_ids,
            "model_kind": record.kind,
            "seed": 42,
            "sample_count": 20000,
            "coverage": {
                "available_sessions": available,
                "missing_sessions": missing,
                "summary": summary,
                "flags": flags,
            },
            "drivers": predictions,
        }
    )


def forecast_for_case(
    event: Event,
    target: TargetState,
    rows: pd.DataFrame,
    registry: ModelRegistry,
    record: ModelRecord,
    registry_root: Path,
    *,
    now: datetime,
) -> Forecast:
    """Issue the selected release, or its paired baseline for a case only that validated.

    Only an unvalidated optional-data case falls back; missing required inputs and
    integrity failures stay unavailable whichever model is selected.
    """
    try:
        return make_forecast(event, target, rows, record, registry_root, now=now)
    except UnvalidatedCase:
        fallback = fallback_model(registry, record, now)
        if fallback is None:
            raise
        return make_forecast(
            event, target, rows, fallback, registry_root, now=now, fallback_for=record
        )


def run_tick(
    site_dir: str | Path,
    archive_dir: str | Path,
    registry_path: str | Path,
    *,
    now: datetime | None = None,
    ingest: bool = True,
) -> dict:
    root, registry_path = Path(site_dir), Path(registry_path)
    site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
    registry = ModelRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
    clock = now or datetime.now(UTC)
    verify_archive(root / "forecasts")
    report = {"checked_at": clock.isoformat(), "issued": [], "unavailable": [], "errors": []}
    from .ingestion import ingest_session
    from .providers import JolpicaProvider

    archive_rows = load_summaries(archive_dir)
    for event in site.events:
        # Synthetic example events are not provider identities or real calendars.
        if not event.id.startswith(f"{event.season}-"):
            continue
        for target in event.targets:
            if target.forecast_id or target.state == "unavailable":
                continue
            window = publication_window(target, clock)
            if window == "early":
                continue
            if window == "missed":
                target.state, target.reason = (
                    "unavailable",
                    "No forecast was issued before the cutoff.",
                )
                report["unavailable"].append(f"{event.id}/{target.target}")
                continue
            record = select_model(registry, target.target, clock)
            if record is None:
                target.reason = "Awaiting an approved, historically evaluated model."
                continue
            try:
                prior = pd.read_parquet(
                    verified_path(registry_path.parent, record.history_path, record.hashes)
                )
                if ingest:
                    provider = JolpicaProvider(Path(archive_dir) / "jolpica-cache")
                    for session in event.sessions:
                        if (
                            session.end <= clock
                            and session.start < target.cutoff_at
                            and session.kind in ("FP1", "FP2", "FP3", "Q")
                        ):
                            ingest_session(
                                event.season,
                                event.round,
                                session.kind,
                                archive_dir,
                                jolpica_provider=provider,
                            )
                    archive_rows = load_summaries(archive_dir)
                summaries = pd.concat([prior, archive_rows], ignore_index=True)
                summaries = eligible_summary_versions(summaries, target.cutoff_at)
                if not target.entrant_ids:
                    from .rosters import infer_roster

                    roster, observed_at = infer_roster(event, target, summaries, archive_dir)
                    if roster and observed_at is not None:
                        known = {d.id: d for d in event.entrants}
                        known.update({d.id: d for d in roster})
                        event.entrants = list(known.values())
                        target.entrant_ids = [d.id for d in roster]
                        target.roster_source = (
                            "Latest completed pre-target session; provisional roster"
                        )
                        target.roster_observed_at = observed_at.to_pydatetime()
                if not target.entrant_ids:
                    raise ValueError("Required pre-target entrant list is unavailable.")
                rows = build_feature_rows(summaries, target_rows(event, target.target))
                # Re-read the real clock after network IO; slow fetches may miss the cutoff.
                final_clock = now or datetime.now(UTC)
                forecast = forecast_for_case(
                    event, target, rows, registry, record, registry_path.parent, now=final_clock
                )
                publish_forecast(forecast, root / "forecasts", now=final_clock)
                site.forecasts.append(forecast)
                target.forecast_id, target.state, target.reason = forecast.id, "issued", None
                report["issued"].append(forecast.id)
            except (
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                httpx.HTTPError,
                requests.RequestException,
            ) as exc:
                target.reason = str(exc)
                report["errors"].append(
                    {"event": event.id, "target": target.target, "reason": str(exc)}
                )
                if (now or datetime.now(UTC)) > target.cutoff_at:
                    target.state = "unavailable"
        # Reconcile results separately; never change an issued forecast to add actuals.
        for target in event.targets:
            session = next(s for s in event.sessions if s.id == target.session_id)
            if not result_reconciliation_due(site, event, target, clock):
                continue
            try:
                if ingest:
                    ingest_session(
                        event.season, event.round, session.kind, archive_dir, refresh=True
                    )
                from .analysis_export import export_analysis

                analysis, telemetry = export_analysis(event.id, archive_dir)
                publish_telemetry(telemetry, root)
                site.analyses = [a for a in site.analyses if a.event_id != event.id] + [analysis]
            except (
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                httpx.HTTPError,
                requests.RequestException,
            ) as exc:
                report["errors"].append(
                    {
                        "event": event.id,
                        "target": target.target,
                        "reason": "Result reconciliation: " + str(exc),
                    }
                )
    site.generated_at = clock
    publish_site(site, root)
    return report
