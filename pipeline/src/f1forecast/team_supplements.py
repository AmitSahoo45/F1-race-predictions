"""Apply reviewed same-event team evidence without modifying archived observations."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .identity import canonical_team_id
from .ingestion import SessionSnapshot


def _timestamp(value: object, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(str(value))
    if not isinstance(stamp, pd.Timestamp) or stamp.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def apply_team_supplement(
    manifest: dict,
    summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    supplement: dict,
    evidence_path: Path | str,
) -> SessionSnapshot:
    """Validate reviewed entry-list evidence and prepare an immutable replacement.

    Mapping review is explicit: this function verifies the pinned PDF bytes and
    their supplied identity/time metadata; it does not automatically parse PDFs.
    No files are written, including on failure. The caller owns archive writes.
    """
    for key in ("event_id", "season", "session_id", "circuit_id"):
        if supplement.get(key) != manifest.get(key):
            raise ValueError(f"team supplement {key} mismatch")
        if key not in summary or not bool(summary[key].eq(manifest[key]).all()):
            raise ValueError(f"summary {key} mismatch")
    if manifest.get("session_type") != "FP1" or not manifest.get("snapshot_id"):
        raise ValueError("entry-list supplement requires an archived FP1 session")
    source = supplement["source"]
    url = urlparse(str(source["url"]))
    if url.scheme != "https" or url.hostname != "www.fia.com":
        raise ValueError("team evidence must identify an official FIA HTTPS source")
    evidence = Path(evidence_path).read_bytes()
    if not evidence.startswith(b"%PDF") or sha256(evidence).hexdigest() != source["sha256"]:
        raise ValueError("team evidence PDF hash mismatch")
    published = _timestamp(source["published_at"], "source publication")
    retrieved = _timestamp(source["retrieved_at"], "source retrieval")
    session_start = _timestamp(manifest["session_start"], "session start")
    if published > session_start or retrieved < published:
        raise ValueError("team evidence publication/retrieval chronology invalid")
    if not supplement.get("source_event_name"):
        raise ValueError("team evidence requires the reviewed source event name")
    if "driver_code" not in summary or bool(summary["driver_code"].isna().any()):
        raise ValueError("summary driver codes missing")
    if summary["driver_code"].duplicated().any():
        raise ValueError("duplicate summary driver code")
    drivers = supplement["drivers"]
    codes = set(summary["driver_code"].astype(str))
    if set(drivers) != codes:
        raise ValueError("team supplement driver codes mismatch")
    numbers: dict[str, set[str]] = {}
    for name, code_column in (("laps", "Driver"), ("results", "Abbreviation")):
        frame = tables.get(name, pd.DataFrame())
        if {code_column, "DriverNumber"}.issubset(frame.columns):
            for code, number in frame[[code_column, "DriverNumber"]].itertuples(index=False):
                if pd.notna(code) and pd.notna(number):
                    numbers.setdefault(str(code), set()).add(str(number))
    season = int(manifest["season"])
    replacements: dict[str, str] = {}
    for code in sorted(codes):
        entrant = drivers[code]
        if numbers.get(code) != {str(entrant["number"])}:
            raise ValueError(f"team supplement driver number mismatch: {code}")
        team = canonical_team_id(entrant["team_id"], season)
        if team is None or team != entrant["team_id"]:
            raise ValueError(f"team supplement requires canonical team identity: {code}")
        existing = summary.loc[summary["driver_code"].eq(code), "team_id"].iloc[0]
        existing_canonical = canonical_team_id(existing, season)
        if existing_canonical is not None and existing_canonical != team:
            raise ValueError(f"team supplement conflicts with observed team: {code}")
        if existing_canonical is None:
            replacements[code] = team

    updated = summary.copy(deep=True)
    for code, team in replacements.items():
        mask = updated["driver_code"].eq(code)
        updated.loc[mask, "team_id"] = team
        if "available_at" in updated:
            for index in updated.index[mask]:
                updated.loc[index, "available_at"] = max(
                    _timestamp(updated.loc[index, "available_at"], "previous availability"),
                    retrieved,
                )
    new_manifest = deepcopy(manifest)
    new_manifest["parent_snapshot_id"] = new_manifest.pop("snapshot_id")
    new_manifest["retrieved_at"] = max(
        datetime.now(UTC), retrieved.to_pydatetime(),
        _timestamp(manifest["retrieved_at"], "parent retrieval").to_pydatetime()
        + timedelta(microseconds=1),
    ).isoformat()
    new_manifest["source_files"] = [*new_manifest.get("source_files", []), deepcopy(source)]
    new_manifest["sources"] = {
        **new_manifest.get("sources", {}), "team_identity": "FIA same-event pre-FP1 entry list",
    }
    new_manifest["team_identity_supplement"] = {
        **deepcopy(supplement), "filled_driver_codes": sorted(replacements),
        "parent_snapshot_id": manifest["snapshot_id"],
    }
    coverage = new_manifest.setdefault("coverage", {})
    missing = updated["team_id"].map(lambda value: canonical_team_id(value, season)).isna()
    coverage["missing_team_drivers"] = sorted(updated.loc[missing, "driver_code"].astype(str))
    telemetry_satisfied = not coverage.get("missing_telemetry_drivers") or bool(
        coverage.get("telemetry_omission_reason")
    )
    coverage["complete"] = bool(
        coverage.get("session_complete") and telemetry_satisfied
        and not coverage.get("missing_results") and not coverage["missing_team_drivers"]
    )
    return SessionSnapshot(
        updated, {name: table.copy(deep=True) for name, table in tables.items()},
        new_manifest["sources"], new_manifest,
    )
