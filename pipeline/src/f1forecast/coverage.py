"""Filesystem-only audit of archived sessions against the public calendar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .calendar import event_from_race
from .providers import JolpicaProvider


class CalendarProvider(Protocol):
    def fetch_races(self, year: int) -> list[dict]: ...


def _latest_manifests(root: Path) -> tuple[dict[str, dict], int]:
    """Read only atomically published snapshot directories, never DuckDB."""
    latest: dict[str, tuple[datetime, str, dict]] = {}
    versions = 0
    for file in (root / "snapshots").glob("*/manifest.json"):
        if file.parent.name.startswith("."):
            continue
        manifest = json.loads(file.read_text(encoding="utf-8"))
        if manifest.get("snapshot_id") != file.parent.name:
            continue
        versions += 1
        session_id = str(manifest["session_id"])
        retrieved = datetime.fromisoformat(manifest["retrieved_at"])
        current = latest.get(session_id)
        if current is None or (retrieved, file.parent.name) > (current[0], current[1]):
            latest[session_id] = (retrieved, file.parent.name, manifest)
    return {session_id: row[2] for session_id, row in latest.items()}, versions


def _bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except FileNotFoundError:
            # A concurrent writer may atomically rename its staging directory.
            continue
    return total


def build_coverage_report(
    archive_dir: Path | str,
    years: Iterable[int],
    *,
    now: datetime | None = None,
    calendar_provider: CalendarProvider | None = None,
) -> dict:
    """Count only calendar sessions whose scheduled end has passed."""
    root = Path(archive_dir)
    cutoff = now or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise ValueError("coverage cutoff must be timezone aware")
    years_list = sorted(set(years))
    provider = calendar_provider or JolpicaProvider(root / "jolpica-cache")
    latest, versions = _latest_manifests(root)
    requested: dict[int, set[str]] = {}
    for year in years_list:
        requested[year] = {
            session.id
            for race in provider.fetch_races(year)
            for session in event_from_race(race).sessions
            if session.end <= cutoff
        }

    all_requested = set().union(*requested.values()) if requested else set()
    present = all_requested & latest.keys()
    missing = sorted(all_requested - latest.keys())
    incomplete = sorted(
        session_id
        for session_id in present
        if not latest[session_id].get("coverage", {}).get("session_complete", False)
    )
    finalized = {
        session_id
        for session_id in present
        if latest[session_id].get("coverage", {}).get("session_complete", False)
    }
    fully_covered = {
        session_id
        for session_id in finalized
        if latest[session_id].get("coverage", {}).get("complete", False)
    }
    source_counts = Counter(str(latest[session_id].get("source", "unknown")) for session_id in present)
    per_year: dict[str, dict] = {}
    for year, expected in requested.items():
        archived = expected & latest.keys()
        source_year = Counter(str(latest[session_id].get("source", "unknown")) for session_id in archived)
        per_year[str(year)] = {
            "requested": len(expected),
            "archived": len(archived),
            "finalized": len(archived & finalized),
            "fully_covered": len(archived & fully_covered),
            "missing": len(expected - latest.keys()),
            "incomplete": len(archived - finalized),
            "primary_sessions": source_year.get("FastF1", 0),
            "bulk_sessions": sum(
                count for source, count in source_year.items() if source.startswith("TracingInsights/")
            ),
            "source_counts": dict(sorted(source_year.items())),
        }

    coverages = [latest[session_id].get("coverage", {}) for session_id in present]
    source_file_inventory = [
        {"session_id": session_id, **file}
        for session_id in present
        for file in latest[session_id].get("source_files", [])
    ]
    retrieved = [datetime.fromisoformat(latest[session_id]["retrieved_at"]) for session_id in present]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "calendar_source": "Jolpica race schedule",
        "years": years_list,
        "requested_sessions": len(all_requested),
        "archived_sessions": len(present),
        "finalized_sessions": len(finalized),
        "fully_covered_sessions": len(fully_covered),
        "missing_session_ids": missing,
        "incomplete_session_ids": incomplete,
        "orphan_archived_session_ids": sorted(
            session_id
            for session_id in latest.keys() - all_requested
            if any(session_id.startswith(f"{year}-") for year in years_list)
        ),
        "snapshot_versions": versions,
        "source_counts": dict(sorted(source_counts.items())),
        "extraction_versions": dict(sorted(Counter(
            str(latest[session_id]["extraction_version"])
            for session_id in present
            if latest[session_id].get("extraction_version")
        ).items())),
        "provenance_modes": dict(sorted(Counter(
            str(latest[session_id].get("provenance", "unknown")) for session_id in present
        ).items())),
        "completion_evidence_counts": dict(sorted(Counter(
            str(latest[session_id].get("coverage", {}).get("completion_evidence"))
            for session_id in finalized
            if latest[session_id].get("coverage", {}).get("completion_evidence")
        ).items())),
        "per_year": per_year,
        "missing_telemetry_driver_sessions": sum(bool(c.get("missing_telemetry_drivers")) for c in coverages),
        "missing_lap_driver_sessions": sum(bool(c.get("missing_lap_drivers")) for c in coverages),
        "unmatched_identity_driver_sessions": sum(bool(c.get("unmatched_identity_drivers")) for c in coverages),
        "missing_weather_sessions": sum(not c.get("weather_rows", 0) for c in coverages),
        "missing_circuit_sessions": sum(not c.get("circuit_points", 0) for c in coverages),
        "missing_result_sessions": sum(bool(c.get("missing_results")) for c in coverages),
        "source_file_records": len(source_file_inventory),
        "pinned_source_file_records": sum(
            len(str(file.get("sha256", ""))) == 64 for file in source_file_inventory
        ),
        "pinned_commits": sorted({
            str(file["commit"]) for file in source_file_inventory if file.get("commit")
        }),
        "source_file_inventory": sorted(
            source_file_inventory, key=lambda file: (str(file["session_id"]), str(file.get("url", "")))
        ),
        "retrieval_window_utc": {
            "earliest": min(retrieved).isoformat() if retrieved else None,
            "latest": max(retrieved).isoformat() if retrieved else None,
        },
        "archive_bytes": _bytes_under(root),
        "snapshot_bytes": _bytes_under(root / "snapshots"),
        "fastf1_cache_bytes": _bytes_under(root / "fastf1-cache"),
        "jolpica_cache_bytes": _bytes_under(root / "jolpica-cache"),
        "availability_note": (
            "Retrospective archive retrieval is not historical availability. "
            "Any pre-cutoff reconstruction uses inferred availability; prospective forecasts "
            "require observations captured before their cutoff."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable F1 archive coverage")
    parser.add_argument("--archive", default="data/archive")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--output", default="data/rehearsal/coverage-report.json")
    args = parser.parse_args()
    report = build_coverage_report(args.archive, range(args.start_year, args.end_year + 1))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}: {report['archived_sessions']}/{report['requested_sessions']} sessions")


if __name__ == "__main__":
    main()
