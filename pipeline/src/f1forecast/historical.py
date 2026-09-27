"""Publish observed weekends and saved chronological folds without refitting history."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .contracts import Event, Forecast, SiteData
from .publication import canonical_json, publish_forecast, publish_site, verify_archive
from .rosters import drivers_from_snapshots as _drivers


def forecast_from_record(
    event: Event, record: dict, report: dict, *, model_choice: str, now: datetime | None = None
) -> Forecast:
    """Use the exact saved fold distributions, never the final all-history model."""
    if record["event_id"] != event.id:
        raise ValueError("backtest event does not match manifest event")
    block = record.get(f"{model_choice}_forecast")
    if model_choice not in {"baseline", "learned"} or not block:
        raise ValueError("requested chronological fold is unavailable")
    if not set(block["driver_ids"]).issubset({d.id for d in event.entrants}):
        raise ValueError("fold driver identities are absent from the event roster")
    target = next(t for t in event.targets if t.target == report["target"])
    cutoff = datetime.fromisoformat(record["cutoff"])
    if target.cutoff_at != cutoff:
        raise ValueError("historical cutoff differs from target manifest")
    recipe = {
        "dataset": report["input_dataset_sha256"],
        "target": report["target"],
        "config": report["training_config"],
        "baseline": report["baseline_name"],
        "choice": model_choice,
    }
    digest = hashlib.sha256(canonical_json(recipe)).hexdigest()[:12]
    predictions = []
    for rank, i in enumerate(np.argsort(block["expected_positions"], kind="stable"), 1):
        probabilities = block["position_probabilities"][i]
        cdf = np.cumsum(probabilities)
        predictions.append(
            {
                "driver_id": block["driver_ids"][i],
                "rank": rank,
                "expected_position": block["expected_positions"][i],
                "p10": int(np.searchsorted(cdf, 0.1)) + 1,
                "p90": int(np.searchsorted(cdf, 0.9)) + 1,
                "win_probability": block["win_probabilities"][i],
                "podium_probability": block["podium_probabilities"][i],
                "top10_probability": block["top_ten_probabilities"][i],
                "position_probabilities": probabilities,
                "explanations": [],
            }
        )
    used = sorted({s for s in record.get("used_session_ids", []) if s.startswith(event.id + "-")})
    eligible = {s.id for s in event.sessions if s.end <= cutoff and s.id != target.session_id}
    return Forecast.model_validate(
        {
            "id": f"{event.id}-{report['target']}-reconstructed-{digest}",
            "event_id": event.id,
            "target": report["target"],
            "kind": "reconstructed",
            "generated_at": now or datetime.now(UTC),
            "cutoff_at": cutoff,
            "training_cutoff": record["training_cutoff"],
            "model_version": record["model_version"]
            + ("-baseline" if model_choice == "baseline" else ""),
            "model_kind": "baseline" if model_choice == "baseline" else "catboost",
            "sample_count": block.get("sample_count", 20_000),
            "seed": block.get("sample_seed", 42),
            "input_snapshot_ids": sorted(record.get("source_snapshot_ids", [])),
            "coverage": {
                "available_sessions": used,
                "missing_sessions": sorted(eligible - set(used)),
                "summary": "Historical reconstruction from eligible earlier sessions. Source archives were retrieved after the event; this was not issued before the session.",
                "flags": ["retrospective-archive", *record.get("missing_pattern", [])],
            },
            "drivers": predictions,
        }
    )


def publish_history(
    report_paths: list[str],
    archive_dir: str | Path,
    site_dir: str | Path,
    *,
    years: list[int],
    now: datetime | None = None,
    use_catalog: bool = True,
) -> SiteData:
    """Replace example data with real schedules, observed analysis and held-out folds."""
    from .analysis_export import export_analysis
    from .archive import load_event_snapshots
    from .calendar import event_from_race
    from .providers import JolpicaProvider
    from .reporting import publish_evaluation

    root = Path(site_dir)
    clock = now or datetime.now(UTC)
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    for report in reports:
        if not report.get("input_dataset_sha256") or not report.get("events"):
            raise ValueError("historical publication requires reproducible backtest records")
    previous = (
        SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
        if (root / "site.json").exists()
        else None
    )
    verify_archive(root / "forecasts")
    provider = JolpicaProvider(Path(archive_dir) / "jolpica-cache")
    events, forecasts, analyses = [], [], []
    for year in sorted(set(years)):
        for race in provider.fetch_races(year):
            event = event_from_race(race)
            snapshots = load_event_snapshots(archive_dir, event.id, use_catalog=use_catalog)
            event.entrants = _drivers(snapshots)
            old = next((e for e in previous.events if e.id == event.id), None) if previous else None
            if snapshots:
                analyses.append(export_analysis(event.id, archive_dir, use_catalog=use_catalog))
            for target in event.targets:
                frozen = (
                    next(
                        (
                            t
                            for t in old.targets
                            if t.target == target.target and t.state == "issued"
                        ),
                        None,
                    )
                    if old
                    else None
                )
                if frozen and previous and old:
                    forecast = next(f for f in previous.forecasts if f.id == frozen.forecast_id)
                    target.cutoff_at = frozen.cutoff_at
                    target.state, target.forecast_id = "issued", forecast.id
                    target.entrant_ids = frozen.entrant_ids
                    original = next(s for s in old.sessions if s.id == frozen.session_id)
                    event.sessions = [
                        original if s.id == original.id else s for s in event.sessions
                    ]
                    known = {d.id: d for d in old.entrants}
                    known.update({d.id: d for d in event.entrants})
                    event.entrants = list(known.values())
                    forecasts.append(forecast)
                    continue
                report = next((r for r in reports if r["target"] == target.target), None)
                record = (
                    next((r for r in report["events"] if r["event_id"] == event.id), None)
                    if report
                    else None
                )
                # Development-selected models are shown only on the held-out periods.
                if year >= 2025 and record and report and record.get("training_cutoff"):
                    choice = (
                        "learned"
                        if report.get("promotion_passed") and record.get("learned_forecast")
                        else "baseline"
                    )
                    historical_cutoff = datetime.fromisoformat(record["cutoff"])
                    changed_schedule = historical_cutoff != target.cutoff_at
                    if changed_schedule:
                        manifest = next((m for m, _, _ in snapshots
                                         if m["session_id"] == target.session_id), None)
                        if manifest is None or datetime.fromisoformat(manifest["session_start"]) != historical_cutoff + timedelta(minutes=30):
                            raise ValueError("historical cutoff conflicts with schedule without matching archived session evidence")
                        target.cutoff_at = historical_cutoff
                        event.sessions = [
                            s.model_copy(update={
                                "start": datetime.fromisoformat(manifest["session_start"]),
                                "end": datetime.fromisoformat(manifest["session_end"]),
                            }) if s.id == target.session_id else s
                            for s in event.sessions
                        ]
                    forecast = forecast_from_record(
                        event, record, report, model_choice=choice, now=clock
                    )
                    if changed_schedule:
                        forecast.coverage.flags.append("target-time-from-archived-session")
                        forecast.coverage.summary += " Target timing follows the archived session and differs from the current calendar."
                    existing = root / "forecasts" / "reconstructed" / f"{forecast.id}.json"
                    if existing.exists():
                        prior = Forecast.model_validate_json(existing.read_text(encoding="utf-8"))
                        candidate = forecast.model_copy(update={"generated_at": prior.generated_at})
                        if prior != candidate:
                            raise ValueError(
                                "saved historical forecast differs from this evaluation"
                            )
                        forecast = prior
                    publish_forecast(forecast, root / "forecasts", now=clock)
                    forecasts.append(forecast)
                    target.forecast_id, target.state = forecast.id, "reconstructed"
                    target.entrant_ids = [d.driver_id for d in forecast.drivers]
                    target.roster_source = (
                        "Historical roster from the eligible pre-target source session"
                    )
                elif target.cutoff_at < clock:
                    target.state = "unavailable"
                    target.reason = (
                        "Development period; no held-out forecast is published."
                        if year < 2025
                        else "No eligible chronological forecast is available for this session."
                    )
            events.append(Event.model_validate(event.model_dump()))
    evaluation = (
        previous.evaluation
        if previous and previous.evaluation.status == "evaluated"
        else {
            "status": "pending",
            "summary": "Real historical data ingestion and chronological evaluation are in progress. No model accuracy claim is published yet.",
            "seasons": [],
            "comparison": [],
            "limitations": [
                "Archived data can include later corrections; backtests are historical reconstructions.",
                "Public telemetry omits fuel loads, setups and many team sensor channels.",
                "Prospective forecasts require an approved model and eligible data before the target cutoff.",
            ],
        }
    )
    site = SiteData.model_validate(
        {
            "generated_at": clock,
            "demo_notice": None,
            "events": events,
            "forecasts": forecasts,
            "analyses": analyses,
            "evaluation": evaluation,
        }
    )
    publish_site(site, root)
    if reports:
        publish_evaluation(report_paths, root)
        site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
    return site
