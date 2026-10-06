"""Content-addressed, immutable forecast publication with explicit time gates."""

import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .contracts import Analysis, EventTelemetry, Forecast, SiteData


def canonical_json(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def publication_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".publication.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(
            "publication already running; inspect any stale lock before recovery"
        ) from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish_forecast(
    forecast: Forecast, directory: str | Path, *, now: datetime | None = None
) -> Path:
    forecast = Forecast.model_validate(forecast.model_dump())
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("publication clock must be timezone-aware")
    if forecast.kind == "issued" and (now > forecast.cutoff_at or forecast.generated_at > now):
        raise ValueError(
            "issued publication must happen before cutoff with an honest generation time"
        )
    root = Path(directory)
    relative = f"{forecast.kind}/{forecast.id}.json"
    path = root / relative
    content = canonical_json(forecast.model_dump(mode="json"))
    digest = hashlib.sha256(content).hexdigest()
    with publication_lock(root):
        ledger_path = root / "hashes.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
        if relative in ledger or path.exists():
            if not path.exists() or ledger.get(relative) != digest or path.read_bytes() != content:
                raise ValueError("forecast is immutable; refusing replacement")
            return path
        # A crash between these writes fails closed at verify_archive (unindexed file).
        atomic_write(path, content)
        ledger[relative] = digest
        atomic_write(ledger_path, canonical_json(ledger))
    return path


def verify_archive(directory: str | Path, previous_hashes: dict | None = None) -> int:
    root = Path(directory)
    ledger_path = root / "hashes.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    for name, old_hash in (previous_hashes or {}).items():
        if ledger.get(name) != old_hash:
            raise ValueError(f"immutable hash changed or removed: {name}")
    files = {
        p.relative_to(root).as_posix()
        for kind in ("issued", "reconstructed", "demonstration")
        for p in (root / kind).glob("*.json")
    }
    if files != set(ledger):
        raise ValueError("archive has missing or unindexed forecasts")
    for name, expected in ledger.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("unsafe archive path")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"forecast hash mismatch: {name}")
        forecast = Forecast.model_validate_json(content)
        if name != f"{forecast.kind}/{forecast.id}.json":
            raise ValueError("forecast identity disagrees with archive path")
    return len(ledger)


def publish_site(site: SiteData, directory: str | Path) -> None:
    validated = SiteData.model_validate(site.model_dump())
    destination = Path(directory) / "site.json"
    payload = validated.model_dump(mode="json")
    if destination.exists():
        old = json.loads(destination.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "generated_at"} == {
            k: v for k, v in payload.items() if k != "generated_at"
        }:
            return
    atomic_write(destination, canonical_json(payload))


def publish_telemetry(telemetry: EventTelemetry, directory: str | Path) -> Path:
    """Keep traces out of site.json; the static build splits them per driver."""
    validated = EventTelemetry.model_validate(telemetry.model_dump())
    destination = Path(directory) / "telemetry" / f"{validated.event_id}.json"
    payload = (
        json.dumps(
            validated.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if not destination.exists() or destination.read_bytes() != payload:
        atomic_write(destination, payload)
    return destination


def verify_telemetry(analyses: list[Analysis], directory: str | Path) -> int:
    """Every trace file belongs to a published analysis and names only its drivers."""
    drivers = {a.event_id: {d.driver_id for d in a.drivers} for a in analyses}
    files = sorted((Path(directory) / "telemetry").glob("*.json"))
    for path in files:
        telemetry = EventTelemetry.model_validate_json(path.read_text(encoding="utf-8"))
        if path.stem != telemetry.event_id:
            raise ValueError(f"telemetry file name disagrees with its event: {path.name}")
        if telemetry.event_id not in drivers:
            raise ValueError(f"telemetry has no published analysis: {path.name}")
        if unknown := {t.driver_id for t in telemetry.traces} - drivers[telemetry.event_id]:
            raise ValueError(f"telemetry drivers absent from the analysis: {sorted(unknown)}")
    return len(files)


def verify_archive_history(directory: str | Path, *, repo_root: str | Path = ".") -> int:
    """Reject coordinated artifact/ledger rewrites against every committed ledger."""
    root = Path(repo_root).resolve()
    archive = Path(directory).resolve()
    ledger_path = (archive / "hashes.json").relative_to(root).as_posix()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise ValueError("cannot verify publication history: " + result.stderr.strip())
        return result.stdout

    if git("rev-parse", "--is-shallow-repository").strip() != "false":
        raise ValueError("publication history requires a full Git checkout")
    commits = git("log", "--format=%H", "--", ledger_path).splitlines()
    previous = {}
    for commit in commits:
        ledger = json.loads(git("show", f"{commit}:{ledger_path}"))
        for path, digest in ledger.items():
            if path in previous and previous[path] != digest:
                raise ValueError(f"immutable hash changed or removed in history: {path}")
            previous[path] = digest
    verify_archive(archive, previous)
    return len(commits)
