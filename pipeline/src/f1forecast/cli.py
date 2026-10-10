"""Local training and bounded scheduled inference entry points."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import requests


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="f1forecast")
    commands = root.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Export Pydantic JSON Schemas")
    schema.add_argument("--output", default="schemas")
    demo = commands.add_parser("demo", help="Create explicitly synthetic UI fixtures")
    demo.add_argument("--output", default="site-data")
    validate = commands.add_parser("validate", help="Check contracts and immutable forecasts")
    validate.add_argument("--site-data", default="site-data")
    validate.add_argument("--base-ref", help="Also compare archive hashes against a Git base ref")
    validate.add_argument(
        "--all-history",
        action="store_true",
        help="Check every committed ledger in a full Git checkout",
    )
    ingest = commands.add_parser("ingest", help="Archive one provider session")
    ingest.add_argument("year", type=int)
    ingest.add_argument("event", type=int)
    ingest.add_argument("session", choices=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"])
    ingest.add_argument("--archive", default="data/archive")
    ingest.add_argument("--refresh", action="store_true")
    backfill = commands.add_parser("backfill", help="Resumable, serial historical ingestion")
    backfill.add_argument("--start-year", type=int, default=2022)
    backfill.add_argument("--end-year", type=int, default=datetime.now(UTC).year)
    backfill.add_argument("--archive", default="data/archive")
    backfill.add_argument("--sessions", nargs="+", default=["FP1", "FP2", "FP3", "Q", "R"])
    prepare = commands.add_parser(
        "prepare", help="Build cutoff-safe historical features and labels"
    )
    prepare.add_argument("--archive", default="data/archive")
    prepare.add_argument("--output", default="data/training.parquet")
    prepare.add_argument(
        "--snapshot-files",
        action="store_true",
        help="Read immutable Parquet snapshots without opening the catalog",
    )
    for command in ("train", "evaluate"):
        model = commands.add_parser(command)
        model.add_argument("target", choices=["qualifying", "race"])
        model.add_argument("--data", default="data/training.parquet")
        model.add_argument("--output", default="models/runs")
        model.add_argument("--iterations", type=int, default=300)
        model.add_argument("--min-train-events", type=int, default=3)
        if command == "train":
            model.add_argument("--training-cutoff", required=True, help="UTC ISO timestamp")
    approve = commands.add_parser("approve", help="Register an evaluated model for inference")
    approve.add_argument("--metadata", required=True)
    approve.add_argument("--report", required=True)
    approve.add_argument("--history", required=True, help="Derived session summary Parquet")
    approve.add_argument("--registry", default="models/registry.json")
    approve.add_argument("--kind", choices=["baseline", "catboost"], default="baseline")
    history = commands.add_parser("export-history", help="Package derived summaries for inference")
    history.add_argument("--archive", default="data/archive")
    history.add_argument("--output", default="data/history.parquet")
    history.add_argument(
        "--snapshot-files",
        action="store_true",
        help="Read immutable snapshots without opening the catalog",
    )
    calendar = commands.add_parser("sync-calendar")
    calendar.add_argument("year", type=int)
    calendar.add_argument("--site-data", default="site-data")
    calendar.add_argument("--cache", default="data/calendar-cache")
    entrants = commands.add_parser("set-entrants", help="Supply a verified pre-session entry list")
    entrants.add_argument("event_id")
    entrants.add_argument("--file", required=True)
    entrants.add_argument("--site-data", default="site-data")
    entrants.add_argument("--target", choices=["qualifying", "race"], required=True)
    entrants.add_argument(
        "--source", required=True, help="Source URL or entry-list evidence reference"
    )
    tick = commands.add_parser("tick", help="Bounded weekend inference; never train in CI")
    tick.add_argument("--site-data", default="site-data")
    tick.add_argument("--archive", default="data/archive")
    tick.add_argument("--registry", default="models/registry.json")
    tick.add_argument("--refresh-calendar", action="store_true")
    tick.add_argument("--offline", action="store_true")
    analysis = commands.add_parser("export-analysis")
    analysis.add_argument("event_id")
    analysis.add_argument("--archive", default="data/archive")
    analysis.add_argument("--site-data", default="site-data")
    evaluation = commands.add_parser("publish-evaluation")
    evaluation.add_argument("--reports", nargs="+", required=True)
    evaluation.add_argument("--site-data", default="site-data")
    historical = commands.add_parser(
        "publish-history", help="Publish real observed weekends and saved held-out forecasts"
    )
    historical.add_argument("--reports", nargs="*", default=[])
    historical.add_argument(
        "--years", nargs="+", type=int, default=list(range(2022, datetime.now(UTC).year + 1))
    )
    historical.add_argument("--archive", default="data/archive")
    historical.add_argument("--site-data", default="site-data")
    historical.add_argument(
        "--snapshot-files",
        action="store_true",
        help="Read immutable snapshots while a backfill owns the catalog",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"f1forecast: {exc}", file=sys.stderr)
        return 1


def dispatch(args: argparse.Namespace) -> int:
    command = args.command
    if command == "schema":
        from .schema import export_schemas

        export_schemas(args.output)
    elif command == "demo":
        from .demo import create_demo

        create_demo(args.output)
    elif command == "validate":
        from .contracts import SiteData
        from .publication import verify_archive, verify_archive_history, verify_telemetry

        root = Path(args.site_data)
        site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
        previous = None
        if args.base_ref:
            result = subprocess.run(
                ["git", "show", f"{args.base_ref}:{root.as_posix()}/forecasts/hashes.json"],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                previous = json.loads(result.stdout)
            elif (
                "does not exist" not in result.stderr
                and "exists on disk, but not" not in result.stderr
            ):
                raise ValueError("could not verify Git base archive: " + result.stderr)
        count = verify_archive(root / "forecasts", previous)
        if args.all_history:
            verify_archive_history(root / "forecasts")
        # The bundled copy must exactly match the immutable archive, too.
        for forecast in site.forecasts:
            path = root / "forecasts" / forecast.kind / f"{forecast.id}.json"
            if not path.exists() or json.loads(
                path.read_text(encoding="utf-8")
            ) != forecast.model_dump(mode="json"):
                raise ValueError("bundled forecast differs from its immutable archived artifact")
        traces = verify_telemetry(site.analyses, root)
        print(
            f"Validated {len(site.events)} events, {len(site.forecasts)} bundled forecasts, "
            f"{count} immutable archive entries and {traces} telemetry files."
        )
    elif command == "ingest":
        from .ingestion import ingest_session
        from .providers import JolpicaProvider

        snapshot = ingest_session(
            args.year,
            args.event,
            args.session,
            args.archive,
            refresh=args.refresh,
            jolpica_provider=JolpicaProvider(Path(args.archive) / "jolpica-cache"),
        )
        print(json.dumps(snapshot.manifest, default=str, indent=2))
    elif command == "backfill":
        from .calendar import event_from_race
        from .ingestion import ingest_session
        from .providers import JolpicaProvider

        provider = JolpicaProvider(Path(args.archive) / "jolpica-cache")
        failures = []
        now = datetime.now(UTC)
        for year in range(args.start_year, args.end_year + 1):
            for race in provider.fetch_races(year):
                event = event_from_race(race)
                for session in event.sessions:
                    if session.kind not in args.sessions or session.end >= now:
                        continue
                    try:
                        snapshot = ingest_session(
                            year, event.round, session.kind, args.archive, jolpica_provider=provider
                        )
                        print(f"Archived {snapshot.manifest['session_id']}", flush=True)
                    except (
                        ValueError,
                        RuntimeError,
                        OSError,
                        LookupError,
                        httpx.HTTPError,
                        requests.RequestException,
                    ) as exc:
                        failures.append({"session": session.id, "error": str(exc)})
                        print(f"Unavailable {session.id}: {exc}", file=sys.stderr, flush=True)
        report = Path(args.archive) / "backfill-failures.json"
        report.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        return 1 if failures else 0
    elif command == "prepare":
        from .datasets import prepare_training_data

        data = prepare_training_data(args.archive, args.output, use_catalog=not args.snapshot_files)
        print(f"Prepared {len(data)} driver-target rows across {data.event_id.nunique()} events.")
    elif command in ("train", "evaluate"):
        from .training import fit_final_model, walk_forward_backtest

        data = pd.read_parquet(args.data)
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        common = {"iterations": args.iterations, "min_train_events": args.min_train_events}
        if command == "train":
            artifact = fit_final_model(
                data,
                args.target,
                training_cutoff=args.training_cutoff,
                output_dir=destination,
                **common,
            )
            print(f"Trained {artifact.version}; not approved until evaluation passes.")
        else:
            report = walk_forward_backtest(data, args.target, **common)
            path = destination / f"{args.target}.evaluation.json"
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Evaluation: {path}; promotion passed: {report['promotion_passed']}")
    elif command == "approve":
        from .registry import approve_model

        record = approve_model(
            args.metadata, args.report, args.history, args.registry, kind=args.kind
        )
        print(f"Registered {record.version} with {record.validation_events} held-out events.")
    elif command == "export-history":
        from .archive import load_summaries

        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        load_summaries(args.archive, use_catalog=not args.snapshot_files).to_parquet(
            destination, index=False
        )
    elif command == "sync-calendar":
        from .calendar import sync_calendar

        site = sync_calendar(args.year, args.site_data, args.cache)
        print(f"Synced {len(site.events)} real events; synthetic examples removed from the site.")
    elif command == "set-entrants":
        from .calendar import set_entrants

        set_entrants(
            args.site_data,
            args.event_id,
            json.loads(Path(args.file).read_text(encoding="utf-8")),
            target=args.target,
            source=args.source,
        )
    elif command == "tick":
        from .calendar import sync_calendar
        from .contracts import SiteData
        from .operations import has_weekend_work, run_tick
        site = SiteData.model_validate_json(
            (Path(args.site_data) / "site.json").read_text(encoding="utf-8")
        )
        clock = datetime.now(UTC)
        real_current_events = [
            e for e in site.events if e.season == clock.year and e.id.startswith(f"{clock.year}-")
        ]
        if real_current_events and not has_weekend_work(site, clock):
            print(
                json.dumps(
                    {"status": "idle", "reason": "Outside weekend work windows; no files changed."}
                )
            )
            return 0
        if args.refresh_calendar and not args.offline:
            sync_calendar(
                datetime.now(UTC).year,
                args.site_data,
                Path(args.archive) / "jolpica-cache",
                refresh=True,
            )
        print(
            json.dumps(
                run_tick(args.site_data, args.archive, args.registry, ingest=not args.offline),
                indent=2,
            )
        )
    elif command == "export-analysis":
        from .analysis_export import export_analysis
        from .contracts import SiteData
        from .publication import publish_site, publish_telemetry

        root = Path(args.site_data)
        site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
        analysis, telemetry = export_analysis(args.event_id, args.archive)
        publish_telemetry(telemetry, root)
        site.analyses = [a for a in site.analyses if a.event_id != args.event_id] + [analysis]
        publish_site(site, root)
    elif command == "publish-evaluation":
        from .reporting import publish_evaluation

        publish_evaluation(args.reports, args.site_data)
    elif command == "publish-history":
        from .historical import publish_history

        site = publish_history(
            args.reports,
            args.archive,
            args.site_data,
            years=args.years,
            use_catalog=not args.snapshot_files,
        )
        print(
            f"Published {len(site.events)} real events, {len(site.forecasts)} forecasts and {len(site.analyses)} observed analyses."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
