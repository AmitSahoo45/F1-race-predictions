"""Resume a serial historical import from pinned public JSON with direct fallback.

Only this process writes the archive catalog. A completed existing session,
especially a primary FastF1 snapshot, is always retained. Failures are logged
and never converted into fabricated session rows.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .bulk_provider import (
    COMMITS,
    MissingBulkSourceError,
    PracticeRosterUnavailableError,
    TracingInsightsProvider,
)
from .calendar import event_from_race
from .ingestion import ingest_session
from .providers import FastF1Provider, JolpicaProvider

LOG = logging.getLogger(__name__)


def _completed_manifest(archive: Path, session_id: str) -> dict | None:
    """Look at immutable manifests without opening DuckDB or loading Parquet."""
    candidates = []
    for path in (archive / "snapshots").glob(f"{session_id}_*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("session_id") == session_id
            and manifest.get("coverage", {}).get("session_complete") is True
        ):
            candidates.append(manifest)
    if not candidates:
        return None
    # Prefer an original FastF1 observation even if later bulk snapshots exist.
    return max(
        candidates, key=lambda row: (row.get("source") == "FastF1", row.get("retrieved_at", ""))
    )


def _save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".staging")
    staging.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    staging.replace(path)


def run_bulk_backfill(
    archive_dir: Path | str,
    *,
    years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026),
    jolpica: JolpicaProvider | None = None,
    bulk: TracingInsightsProvider | None = None,
    direct: FastF1Provider | None = None,
    now: datetime | None = None,
    report_path: Path | str | None = None,
    rounds: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Import past sessions once; fall back only when a required JSON file is absent.

    The caller must ensure no other archive writer is active. The report is
    replaced after every session so interruption leaves reviewable progress.
    """
    archive = Path(archive_dir)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        raise ValueError("backfill reference time must be UTC-aware")
    jolpica = jolpica or JolpicaProvider(archive / "jolpica-cache")
    direct = direct or FastF1Provider(archive / "fastf1-cache")
    bulk = bulk or TracingInsightsProvider(jolpica, practice_roster=direct, now=lambda: reference)
    report_file = Path(report_path) if report_path else archive / "bulk-backfill-report.json"
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "reference_time": reference.isoformat(),
        "pinned_revisions": {str(year): COMMITS[year] for year in years},
        "counts": {"skipped": 0, "bulk": 0, "direct_fallback": 0, "failed": 0},
        "sessions": [],
    }
    _save_report(report_file, report)
    for year in years:
        if year not in COMMITS:
            raise ValueError(f"year {year} has no reviewed pinned bulk source")
        for race in jolpica.fetch_races(year):
            round_number = int(race["round"])
            if rounds is not None and (year, round_number) not in rounds:
                continue
            calendar = event_from_race(race)
            gp = next((session for session in calendar.sessions if session.kind == "R"), None)
            if gp is None or reference < gp.end + timedelta(hours=24):
                continue
            kinds = {session.kind for session in calendar.sessions}
            for kind in ("FP1", "FP2", "FP3", "Q", "R"):
                if kind not in kinds:
                    continue
                session_id = f"{year}-{round_number:02d}-{kind}"
                previous = _completed_manifest(archive, session_id)
                if previous is not None:
                    entry = {
                        "session_id": session_id,
                        "outcome": "skipped",
                        "source": previous.get("source"),
                        "snapshot_id": previous.get("snapshot_id"),
                    }
                else:
                    try:
                        snapshot = ingest_session(
                            year,
                            round_number,
                            kind,
                            archive,
                            fastf1_provider=bulk,
                            provenance="retrospective",
                            refresh=True,
                        )
                        entry = {
                            "session_id": session_id,
                            "outcome": "bulk",
                            "source": snapshot.manifest["source"],
                            "snapshot_id": snapshot.manifest.get("snapshot_id"),
                            "coverage": snapshot.manifest.get("coverage"),
                        }
                    except (MissingBulkSourceError, PracticeRosterUnavailableError) as exc:
                        LOG.warning(
                            "%s pinned JSON absent; direct FastF1 fallback: %s", session_id, exc
                        )
                        try:
                            snapshot = ingest_session(
                                year,
                                round_number,
                                kind,
                                archive,
                                fastf1_provider=direct,
                                jolpica_provider=jolpica,
                                provenance="retrospective",
                                refresh=True,
                            )
                            entry = {
                                "session_id": session_id,
                                "outcome": "direct_fallback",
                                "reason": str(exc),
                                "source": snapshot.manifest["source"],
                                "snapshot_id": snapshot.manifest.get("snapshot_id"),
                                "coverage": snapshot.manifest.get("coverage"),
                            }
                        except Exception as fallback_error:  # noqa: BLE001 - record one failed session
                            entry = {
                                "session_id": session_id,
                                "outcome": "failed",
                                "error": f"direct fallback: {type(fallback_error).__name__}: "
                                f"{fallback_error}",
                                "bulk_error": str(exc),
                            }
                    except Exception as exc:  # noqa: BLE001 - continue resumable import
                        entry = {
                            "session_id": session_id,
                            "outcome": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                report["counts"][entry["outcome"]] += 1
                report["sessions"].append(entry)
                report["updated_at"] = datetime.now(UTC).isoformat()
                _save_report(report_file, report)
                LOG.info("%s: %s %s", session_id, entry["outcome"], entry.get("error", ""))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/archive"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_bulk_backfill(
        args.archive,
        years=tuple(args.years),
        report_path=args.report,
    )
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
