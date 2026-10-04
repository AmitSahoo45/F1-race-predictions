"""Prepare reproducible historical feature/label tables from the local archive."""

import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .archive import load_summaries
from .features import NUMERIC_FEATURES, build_feature_rows, completed_session_rows
from .lap_summaries import LAP_SUMMARY_SEMANTICS
from .training import (
    _event_groups,
    attach_results,
    input_dataset_sha256,
    validate_feature_cutoffs,
)


def training_rows_from_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    """Build labels and cutoff-safe features without reading or changing the source frame."""
    if summaries.empty:
        raise ValueError("archive is empty; ingest historical sessions first")
    finalized = completed_session_rows(summaries)
    entries = []
    for _, results in finalized.loc[finalized.session_type.isin(["Q", "R"])].groupby("session_id"):
        session = results.iloc[0]
        target = "qualifying" if session.session_type == "Q" else "race"
        cutoff = pd.Timestamp(session.session_start) - timedelta(minutes=30)
        # Historical roster proxy is the last eligible earlier session, never the target result.
        eligible = finalized.loc[
            finalized.event_id.eq(session.event_id)
            & pd.to_datetime(finalized.session_end, utc=True).le(cutoff)
            & finalized.session_type.isin(
                ["FP1", "FP2", "FP3"] if target == "qualifying" else ["Q"]
            )
        ]
        if eligible.empty:
            continue
        last_id = eligible.sort_values("session_end").iloc[-1].session_id
        for entrant in eligible.loc[eligible.session_id.eq(last_id)].itertuples():
            entries.append(
                {
                    "event_id": session.event_id,
                    "season": int(session.season),
                    "circuit_id": session.circuit_id,
                    "driver_id": entrant.driver_id,
                    "team_id": entrant.team_id,
                    "target": target,
                    "target_session_id": session.session_id,
                    "cutoff": cutoff,
                }
            )
    if not entries:
        raise ValueError("no target has a prior eligible session roster; ingest practice, Q and R")
    features = build_feature_rows(summaries, pd.DataFrame(entries), mode="reconstructed")
    return attach_results(features, summaries)


def prepare_training_data(
    archive_dir: str | Path, destination: str | Path, *, use_catalog: bool = True
) -> pd.DataFrame:
    summaries = load_summaries(archive_dir, use_catalog=use_catalog)
    labeled = training_rows_from_summaries(summaries)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(destination, index=False)
    return labeled


def _check_freeze_destination(destination: Path) -> None:
    if destination.is_symlink() or (
        destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise FileExistsError(f"freeze destination contains existing data: {destination}")


def freeze_training_data(
    archive_dir: str | Path, destination_dir: str | Path, *, use_catalog: bool = False
) -> dict:
    """Atomically publish validated history and training from one immutable source read.

    Feature semantics describe the current code recipe. Referenced source snapshot
    manifests retain the evidence for whether lap summaries were enriched.
    """
    destination = Path(destination_dir).absolute()
    _check_freeze_destination(destination)
    destination = destination.resolve()
    summaries = load_summaries(archive_dir, use_catalog=use_catalog)
    training = training_rows_from_summaries(summaries)
    validate_feature_cutoffs(training)
    groups = {target: len(_event_groups(training, target)) for target in ("qualifying", "race")}
    if "source_snapshot_id" not in summaries or not all(
        isinstance(value, str) and bool(value) for value in summaries["source_snapshot_id"]
    ):
        raise ValueError("freeze requires source snapshot identities for every history row")
    snapshot_ids = sorted(set(summaries["source_snapshot_id"].tolist()))
    dataset_hash = input_dataset_sha256(training)
    manifest = {
        "schema_version": "1",
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_snapshot_ids": snapshot_ids,
        "history_rows": len(summaries),
        "history_sessions": int(summaries["session_id"].nunique()),
        "history_events": int(summaries["event_id"].nunique()),
        "training_rows": len(training),
        "training_events": int(training["event_id"].nunique()),
        "target_events": groups,
        "input_dataset_sha256": dataset_hash,
        "feature_columns": list(NUMERIC_FEATURES),
        "feature_semantics": {
            "availability": "reconstructed; earlier completed sessions with inferred availability",
            "forecast_cutoff_minutes": 30,
            "roster": "whole latest eligible pre-target session",
            "team_identity": "season-aware aliases; no inferred missing affiliations",
            "stints": LAP_SUMMARY_SEMANTICS,
            "compound_mix": "usable-lap-weighted relative compound fractions; unknown stays missing",
            "scope": "current feature recipe; source manifests retain enrichment evidence",
        },
        "validation": {"feature_cutoffs": "passed", "event_groups": groups},
        "files": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent,
    )).resolve()
    # Only this newly allocated sibling may be recursively cleaned on failure.
    if staging.parent != destination.parent or not staging.name.endswith(".staging"):
        raise ValueError("freeze staging directory escaped the destination parent")
    try:
        history_path, training_path = staging / "history.parquet", staging / "training.parquet"
        summaries.to_parquet(history_path, index=False)
        training.to_parquet(training_path, index=False)
        restored_history = pd.read_parquet(history_path)
        restored_training = pd.read_parquet(training_path)
        pd.testing.assert_frame_equal(
            summaries, restored_history, check_dtype=False, check_exact=True,
        )
        if input_dataset_sha256(restored_training) != dataset_hash:
            raise ValueError("saved training dataset fingerprint differs from prepared rows")
        validate_feature_cutoffs(restored_training)
        for target in groups:
            _event_groups(restored_training, target)
        manifest["files"] = {
            path.name: {"sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path in (history_path, training_path)
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("saved freeze manifest differs from verified metadata")
        _check_freeze_destination(destination)
        if destination.exists():
            destination.rmdir()  # Only an empty caller-supplied directory can reach this point.
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
