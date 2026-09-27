"""Approved, integrity-checked model releases used by scheduled inference."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .contracts import Contract, Identifier, Metrics, Target, UTCDate
from .publication import atomic_write, canonical_json


def validated_patterns(report: dict, kind: str) -> list[list[str]]:
    """Approve only patterns with scored evidence; learned cases must pass paired promotion."""
    from .evaluation import promotion_decision

    patterns = []
    for case in report.get("missing_case_evidence", []):
        pattern = sorted(set(case["pattern"]))
        if not pattern or "missing-feature-schema" in pattern:
            continue
        if case.get(kind if kind == "baseline" else "learned", {}).get("evaluated_events", 0) < 1:
            continue
        if kind == "catboost":
            paired = case.get("paired_locked_2025", {})
            learned, baseline = paired.get("learned", {}), paired.get("baseline", {})
            if not learned.get("evaluated_events") or not baseline.get("evaluated_events"):
                continue
            if not promotion_decision(learned, baseline):
                continue
        if pattern not in patterns:
            patterns.append(pattern)
    return sorted(patterns)


class ModelRecord(Contract):
    version: Identifier
    target: Target
    kind: Literal["baseline", "catboost"]
    approved: bool
    approved_at: UTCDate
    training_cutoff: UTCDate
    metadata_path: str
    model_path: str
    history_path: str
    report_path: str
    hashes: dict[str, str]
    temperature: float = Field(gt=0, allow_inf_nan=False)
    baseline_name: str
    validation_events: int = Field(ge=1)
    evaluation: Metrics
    validated_missing_patterns: list[list[str]] = Field(default_factory=list)


class ModelRegistry(Contract):
    schema_version: Literal["1.0"] = "1.0"
    models: list[ModelRecord]


def select_model(registry: ModelRegistry, target: str, now: datetime) -> ModelRecord | None:
    candidates = [
        m
        for m in registry.models
        if m.target == target and m.approved and m.training_cutoff < now and m.approved_at <= now
    ]
    return max(candidates, key=lambda m: (m.training_cutoff, m.approved_at)) if candidates else None


def verified_path(root: Path, relative: str, hashes: dict[str, str]) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("model file is missing or outside registry directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(relative):
        raise ValueError("model artifact hash mismatch")
    return path


def metric_contract(summary: dict) -> Metrics:
    return Metrics.model_validate(
        {
            "probability_loss": summary.get("probability_loss", summary.get("normalized_nll")),
            "position_mae": summary["position_mae"],
            "win_brier": summary["win_brier"],
            "podium_brier": summary["podium_brier"],
            "top10_brier": summary.get("top10_brier", summary.get("top_ten_brier")),
        }
    )


def approve_model(
    metadata_path: str | Path,
    report_path: str | Path,
    history_path: str | Path,
    registry_path: str | Path,
    *,
    kind: Literal["baseline", "catboost"] = "baseline",
) -> ModelRecord:
    """Release only evaluated artifacts, preserving report and training provenance."""
    metadata_path, report_path, history_path = map(Path, (metadata_path, report_path, history_path))
    metadata, report = (
        json.loads(metadata_path.read_text(encoding="utf-8")),
        json.loads(report_path.read_text(encoding="utf-8")),
    )
    target = metadata["target"]
    if report["target"] != target:
        raise ValueError("report target does not match model")
    if not metadata.get("training_config") or metadata["training_config"] != report.get(
        "training_config"
    ):
        raise ValueError("training recipe does not match the evaluated recipe")
    digest = metadata.get("input_dataset_sha256", "")
    if (
        len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or digest != report.get("input_dataset_sha256")
    ):
        raise ValueError("training dataset does not match the evaluated dataset")
    if (metadata.get("baseline_name"), metadata.get("baseline_temperature")) != (
        report.get("baseline_name"),
        report.get("baseline_temperature"),
    ):
        raise ValueError("baseline selection or calibration differs from evaluation")
    summary = report["periods"]["locked_2025"]["learned" if kind == "catboost" else "baseline"]
    if summary.get("evaluated_events", 0) < 1:
        raise ValueError("approval requires completed locked historical evaluation")
    if kind == "catboost":
        from .evaluation import promotion_decision

        paired = report.get("promotion_paired_2025", {})
        learned, baseline = paired.get("learned", {}), paired.get("baseline", {})
        if (
            not report.get("promotion_passed", False)
            or not learned
            or not baseline
            or not promotion_decision(learned, baseline)
        ):
            raise ValueError("learned model has not passed the baseline promotion gate")
    if metadata.get("calibration_oof_events", 0) < 1:
        raise ValueError("approval requires chronological calibration evidence")
    registry_path = Path(registry_path)
    root = registry_path.parent
    registry = (
        ModelRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else ModelRegistry(models=[])
    )
    version = metadata["version"] + ("-baseline" if kind == "baseline" else "")
    if any(m.version == version for m in registry.models):
        raise ValueError("model version already registered; releases are immutable")
    # Restrict the directory component before performing any file operation.
    from pydantic import TypeAdapter

    TypeAdapter(Identifier).validate_python(version)
    model_path = metadata_path.parent / f"{target}.cbm"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("model file differs from training metadata")
    release = root / "approved" / version
    release.mkdir(parents=True, exist_ok=False)
    paths = {}
    hashes = {}
    for key, source, name in [
        ("model_path", model_path, "model.cbm"),
        ("metadata_path", metadata_path, "metadata.json"),
        ("report_path", report_path, "evaluation.json"),
        ("history_path", history_path, "history.parquet"),
    ]:
        destination = release / name
        shutil.copyfile(source, destination)
        relative = destination.relative_to(root).as_posix()
        paths[key] = relative
        hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    record = ModelRecord(
        version=version,
        target=target,
        kind=kind,
        approved=True,
        approved_at=datetime.now(UTC),
        training_cutoff=metadata["training_cutoff"],
        temperature=metadata["temperature"]
        if kind == "catboost"
        else metadata.get("baseline_temperature", 1.0),
        baseline_name=metadata.get("baseline_name", "blend"),
        hashes=hashes,
        **paths,
        validation_events=summary["evaluated_events"],
        evaluation=metric_contract(summary),
        validated_missing_patterns=validated_patterns(report, kind),
    )
    registry.models.append(record)
    atomic_write(registry_path, canonical_json(registry.model_dump(mode="json")))
    return record
