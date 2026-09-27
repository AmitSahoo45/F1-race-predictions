"""Immutable Parquet session snapshots indexed by a local DuckDB catalog."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd


def _connect(archive_dir: Path | str) -> duckdb.DuckDBPyConnection:
    root = Path(archive_dir)
    root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(root / "catalog.duckdb"))
    connection.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id VARCHAR PRIMARY KEY, session_id VARCHAR NOT NULL,
        retrieved_at TIMESTAMPTZ NOT NULL, relative_path VARCHAR NOT NULL,
        source VARCHAR NOT NULL, coverage_json VARCHAR NOT NULL
    )""")
    return connection


def latest_snapshot(
    archive_dir: Path | str, session_id: str
) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]] | None:
    root = Path(archive_dir)
    if not (root / "catalog.duckdb").exists():
        return None
    with _connect(root) as conn:
        row = conn.execute(
            "SELECT relative_path FROM snapshots WHERE session_id = ? ORDER BY retrieved_at DESC LIMIT 1",
            [session_id],
        ).fetchone()
    if row is None:
        return None
    path = root / row[0]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_parquet(path / "summary.parquet")
    tables = {
        file.stem: pd.read_parquet(file)
        for file in path.glob("*.parquet")
        if file.stem != "summary"
    }
    return manifest, summary, tables


def write_snapshot(
    archive_dir: Path | str,
    manifest: dict,
    summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict:
    root = Path(archive_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = f"{manifest['session_id']}|{manifest['retrieved_at']}|{manifest['source']}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    snapshot_id = f"{manifest['session_id']}_{digest}"
    final = root / "snapshots" / snapshot_id
    result = {**manifest, "snapshot_id": snapshot_id, "schema_version": "1"}
    if final.exists():
        stored = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        if stored != result:
            raise ValueError("snapshot identity already exists with different metadata")
        return stored
    staging = root / "snapshots" / f".{snapshot_id}.staging"
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        summary.to_parquet(staging / "summary.parquet", index=False)
        for name, table in tables.items():
            if not name.isidentifier() or name == "summary":
                raise ValueError(f"invalid table name: {name}")
            table.to_parquet(staging / f"{name}.parquet", index=False)
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        staging.replace(final)
        with _connect(root) as conn:
            conn.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
                [
                    snapshot_id,
                    manifest["session_id"],
                    manifest["retrieved_at"],
                    str(final.relative_to(root)),
                    manifest["source"],
                    json.dumps(manifest["coverage"], sort_keys=True),
                ],
            )
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_summaries(archive_dir: Path | str) -> pd.DataFrame:
    """Load the latest immutable snapshot per session, with timestamps intact."""
    root = Path(archive_dir)
    if not (root / "catalog.duckdb").exists():
        return pd.DataFrame()
    with _connect(root) as conn:
        rows = conn.execute("""SELECT relative_path FROM (
            SELECT relative_path, ROW_NUMBER() OVER
              (PARTITION BY session_id ORDER BY retrieved_at DESC) AS rn
            FROM snapshots) WHERE rn = 1""").fetchall()
    if not rows:
        return pd.DataFrame()
    frames = []
    for row in rows:
        path = root / row[0]
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        frames.append(
            pd.read_parquet(path / "summary.parquet").assign(
                source_snapshot_id=manifest["snapshot_id"],
                session_complete=bool(manifest.get("coverage", {}).get("session_complete", False)),
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_event_snapshots(
    archive_dir: Path | str, event_id: str, *, use_catalog: bool = True
) -> list[tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]]:
    """Return latest immutable sessions; filesystem mode is safe during catalog writes."""
    root = Path(archive_dir)
    if not use_catalog:
        latest: dict[str, tuple[pd.Timestamp, Path]] = {}
        for manifest_file in (root / "snapshots").glob(f"{event_id}-*/manifest.json"):
            path = manifest_file.parent
            if path.name.startswith("."):
                continue
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            session_id = str(manifest.get("session_id", ""))
            if not session_id.startswith(f"{event_id}-") or manifest.get("snapshot_id") != path.name:
                continue
            retrieved = pd.Timestamp(manifest["retrieved_at"])
            if not isinstance(retrieved, pd.Timestamp):
                continue
            previous = latest.get(session_id)
            if previous is None or (retrieved, path.name) > (previous[0], previous[1].name):
                latest[session_id] = (retrieved, path)
        return [
            (
                json.loads((path / "manifest.json").read_text(encoding="utf-8")),
                pd.read_parquet(path / "summary.parquet"),
                {
                    file.stem: pd.read_parquet(file)
                    for file in path.glob("*.parquet")
                    if file.stem != "summary"
                },
            )
            for _, path in (latest[session_id] for session_id in sorted(latest))
        ]
    if not (root / "catalog.duckdb").exists():
        return []
    with _connect(root) as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM snapshots WHERE session_id LIKE ? ORDER BY session_id",
            [f"{event_id}-%"],
        ).fetchall()
    return [snapshot for row in rows if (snapshot := latest_snapshot(root, row[0])) is not None]
