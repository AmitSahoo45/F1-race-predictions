"""Resume immutable lap-summary enrichment as the sole archive writer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .archive import index_published_snapshot, write_snapshot
from .bulk_backfill import _save_report
from .bulk_provider import COMMITS, TracingInsightsProvider, restore_source_stints
from .coverage import _latest_manifests
from .job_control import JobControl, WorkStopped
from .lap_summaries import LAP_SUMMARY_FIELDS, LAP_SUMMARY_SEMANTICS, summarize_laps
from .providers import FastF1Provider, JolpicaProvider, SourceValidationError


class PinnedReader(Protocol):
    def _read_json(self, url: str, commit: str) -> tuple[Any, dict[str, str | int]]: ...


def _restore_stints(
    laps: pd.DataFrame, manifest: dict, reader: PinnedReader, control: JobControl
) -> tuple[pd.DataFrame, dict]:
    source = next(
        (
            row
            for row in manifest.get("source_files", [])
            if str(row.get("url", "")).endswith("/session_laptimes.json")
        ),
        None,
    )
    if source is None:
        raise SourceValidationError("missing pinned lap source evidence")
    year = int(manifest.get("season", manifest["session_id"].split("-")[0]))
    commit = COMMITS[year]
    if source.get("commit") != commit or f"/{commit}/" not in source["url"]:
        raise SourceValidationError("lap source is not the reviewed pinned revision")
    payload, evidence = reader._read_json(source["url"], commit)
    control.check()
    if evidence.get("sha256") != source.get("sha256"):
        raise SourceValidationError("pinned lap JSON content hash changed")
    restored = restore_source_stints(laps, payload)
    return (restored if restored is not None else laps), {
        "stint_source": "restored_pinned_source" if restored is not None else "absent_in_source",
        "source_file": evidence,
    }


def run_enrichment(
    archive_dir: Path | str,
    report_path: Path | str,
    *,
    control: JobControl,
    reader: PinnedReader | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Recompute lap inputs in new snapshots; preserve every original snapshot.

    Only hash-identical pinned raw files may supply missing bulk stint IDs.
    Summary availability timestamps remain unchanged: these are new derivations
    of already archived evidence, not a claim of earlier source availability.
    The caller must ensure that no other archive writer is running.
    """
    archive, report_file = Path(archive_dir), Path(report_path)
    report = (
        json.loads(report_file.read_text(encoding="utf-8"))
        if report_file.exists()
        else {
            "started_at": now().isoformat(),
            "sessions": [],
        }
    )
    records = {row["session_id"]: row for row in report["sessions"]}
    report.update(status="running", lap_summary_semantics=LAP_SUMMARY_SEMANTICS)
    report.pop("reason", None)

    def checkpoint() -> None:
        report["sessions"] = sorted(records.values(), key=lambda row: row["session_id"])
        report["counts"] = dict(Counter(row["outcome"] for row in records.values()))
        report["updated_at"] = now().isoformat()
        _save_report(report_file, report)

    try:
        control.check()
        latest, _ = _latest_manifests(archive)
        for session_id, manifest in sorted(latest.items()):
            control.check()
            old_id = manifest["snapshot_id"]
            if manifest.get("lap_summary_semantics") == LAP_SUMMARY_SEMANTICS:
                # Files are published before the catalog transaction. A previous
                # interruption must not leave default training readers stale.
                index_published_snapshot(archive, old_id)
                if records.get(session_id, {}).get("snapshot_id") != old_id:
                    records[session_id] = {
                        "session_id": session_id,
                        "outcome": "already_current",
                        "snapshot_id": old_id,
                    }
                continue
            try:
                directory = archive / "snapshots" / old_id
                summary = pd.read_parquet(directory / "summary.parquet")
                tables = {
                    path.stem: pd.read_parquet(path)
                    for path in directory.glob("*.parquet")
                    if path.stem != "summary"
                }
                laps = tables["laps"]
                evidence = {"stint_source": "archived_source"}
                missing_stints = "Stint" not in laps or bool(laps["Stint"].isna().any())
                if (
                    str(manifest.get("source", "")).startswith("TracingInsights/")
                    and missing_stints
                ):
                    if reader is None:
                        reader = TracingInsightsProvider(
                            JolpicaProvider(archive / "jolpica-cache", sleep=control.sleep),
                            practice_roster=FastF1Provider(
                                archive / "fastf1-cache", sleep=control.sleep
                            ),
                            sleep=control.sleep,
                        )
                    laps, evidence = _restore_stints(laps, manifest, reader, control)
                    tables["laps"] = laps
                if "driver_code" not in summary or "Driver" not in laps:
                    raise SourceValidationError("lap summary requires archived driver codes")
                metrics = pd.DataFrame(
                    [
                        summarize_laps(laps.loc[laps["Driver"].eq(code)])
                        for code in summary["driver_code"]
                    ],
                    index=summary.index,
                    columns=pd.Index(LAP_SUMMARY_FIELDS),
                )
                summary = summary.copy()
                summary[list(LAP_SUMMARY_FIELDS)] = metrics
                stamp = now()
                if stamp.tzinfo is None:
                    raise ValueError("enrichment time must be timezone-aware")
                stamp = max(
                    stamp,
                    datetime.fromisoformat(manifest["retrieved_at"]) + timedelta(microseconds=1),
                )
                updated = {
                    **manifest,
                    "retrieved_at": stamp.isoformat(),
                    "lap_summary_semantics": LAP_SUMMARY_SEMANTICS,
                }
                updated.pop("snapshot_id", None)
                updated["enrichments"] = [
                    *manifest.get("enrichments", []),
                    {
                        "operation": "recompute_lap_summaries",
                        "lap_summary_semantics": LAP_SUMMARY_SEMANTICS,
                        "parent_snapshot_id": old_id,
                        "retrieved_at": stamp.isoformat(),
                        "summary_fields_updated": list(LAP_SUMMARY_FIELDS),
                        "summary_availability_preserved": True,
                        **evidence,
                    },
                ]
                stint_rows = int(laps["Stint"].notna().sum()) if "Stint" in laps else 0
                updated["coverage"] = {
                    **manifest.get("coverage", {}),
                    "stint_rows": stint_rows,
                    "missing_stint_rows": len(laps) - stint_rows,
                }
                control.check()
                result = write_snapshot(archive, updated, summary, tables)
                records[session_id] = {
                    "session_id": session_id,
                    "outcome": "enriched",
                    "parent_snapshot_id": old_id,
                    "snapshot_id": result["snapshot_id"],
                    **evidence,
                }
            except WorkStopped:
                raise
            except Exception as exc:  # noqa: BLE001 - checkpoint failed sessions for safe retry
                records[session_id] = {
                    "session_id": session_id,
                    "outcome": "failed",
                    "snapshot_id": old_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            checkpoint()
        report["status"] = (
            "completed_with_failures"
            if any(row["outcome"] == "failed" for row in records.values())
            else "completed"
        )
    except WorkStopped as exc:
        report["status"], report["reason"] = "stopped", str(exc)
    finally:
        checkpoint()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/archive"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/rehearsal/lap-summary-enrichment-progress.json")
    )
    parser.add_argument(
        "--stop-at",
        type=datetime.fromisoformat,
        required=True,
        help="Explicit timezone-aware future checkpoint deadline",
    )
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stop_at.tzinfo is None or args.stop_at <= datetime.now(UTC):
        raise ValueError("enrichment deadline must be timezone-aware and in the future")
    report = run_enrichment(
        args.archive,
        args.report,
        control=JobControl(stop_at=args.stop_at, stop_file=args.stop_file),
    )
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 1 if report["status"] == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
