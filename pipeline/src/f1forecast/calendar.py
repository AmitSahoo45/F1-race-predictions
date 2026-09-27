"""Provider calendar and explicitly sourced pre-session entry lists."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from .contracts import Driver, Event, SiteData
from .providers import JolpicaProvider
from .publication import publish_site

CIRCUIT_ZONES = {
    "bahrain": "Asia/Bahrain", "jeddah": "Asia/Riyadh", "albert_park": "Australia/Melbourne",
    "suzuka": "Asia/Tokyo", "shanghai": "Asia/Shanghai", "miami": "America/New_York",
    "imola": "Europe/Rome", "monaco": "Europe/Monaco", "villeneuve": "America/Toronto",
    "catalunya": "Europe/Madrid", "red_bull_ring": "Europe/Vienna", "silverstone": "Europe/London",
    "hungaroring": "Europe/Budapest", "spa": "Europe/Brussels", "zandvoort": "Europe/Amsterdam",
    "monza": "Europe/Rome", "baku": "Asia/Baku", "marina_bay": "Asia/Singapore",
    "americas": "America/Chicago", "rodriguez": "America/Mexico_City",
    "interlagos": "America/Sao_Paulo", "vegas": "America/Los_Angeles", "losail": "Asia/Qatar",
    "yas_marina": "Asia/Dubai", "madring": "Europe/Madrid", "madrid": "Europe/Madrid",
    "paul_ricard": "Europe/Paris", "portimao": "Europe/Lisbon", "istanbul": "Europe/Istanbul",
}


def event_from_race(race: dict) -> Event:
    season, round_number = int(race["season"]), int(race["round"])
    event_id = f"{season}-{round_number:02d}"
    sessions = []
    mapping = [("FirstPractice", "FP1", 1), ("SecondPractice", "FP2", 1),
               ("ThirdPractice", "FP3", 1), ("SprintQualifying", "SQ", 1),
               ("SprintShootout", "SQ", 1), ("Sprint", "S", 1), ("Qualifying", "Q", 1)]
    for key, kind, hours in mapping + [("Race", "R", 2)]:
        value = race if key == "Race" else race.get(key)
        if not value or not value.get("time"):
            continue
        if any(s["kind"] == kind for s in sessions):
            continue
        start = datetime.fromisoformat(f"{value['date']}T{value['time']}")
        sessions.append({"id": f"{event_id}-{kind}", "kind": kind, "start": start,
                         "end": start + timedelta(hours=hours)})
    targets = [{"target": {"Q": "qualifying", "R": "race"}[s["kind"]],
                "session_id": s["id"], "cutoff_at": s["start"] - timedelta(minutes=30),
                "state": "scheduled", "reason": None, "forecast_id": None}
               for s in sessions if s["kind"] in ("Q", "R")]
    circuit = race["Circuit"]
    return Event.model_validate({"id": event_id, "season": season, "round": round_number,
        "name": race["raceName"], "circuit": circuit["circuitId"],
        "country": circuit["Location"]["country"],
        "timezone": CIRCUIT_ZONES.get(circuit["circuitId"], "UTC"),
        "sessions": sorted(sessions, key=lambda s: s["start"]), "entrants": [], "targets": targets})


def sync_calendar(year: int, site_dir: str | Path, cache_dir: str | Path, *, refresh: bool = False) -> SiteData:
    root = Path(site_dir)
    previous = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
    old = {e.id: e for e in previous.events}
    events = []
    for race in JolpicaProvider(cache_dir, cache_ttl_s=0 if refresh else 3600).fetch_races(year):
        event = event_from_race(race)
        existing = old.get(event.id)
        if existing:
            event.entrants = existing.entrants
            for target in event.targets:
                old_target = next((t for t in existing.targets if t.target == target.target), None)
                if old_target and old_target.cutoff_at == target.cutoff_at:
                    event.targets = [old_target if t.target == target.target else t for t in event.targets]
                if old_target and old_target.forecast_id:
                    event.targets = [old_target if t.target == target.target else t for t in event.targets]
                    # Preserve the schedule against which an immutable forecast was issued.
                    frozen = next(s for s in existing.sessions if s.id == old_target.session_id)
                    event.sessions = [frozen if s.id == frozen.id else s for s in event.sessions]
        events.append(event)
    real_old = [e for e in previous.events if e.season != year and e.id.startswith(f"{e.season}-")]
    events.extend(real_old)
    known = {e.id for e in events}
    site = SiteData(generated_at=datetime.now(UTC), demo_notice=None,
        events=events, forecasts=[f for f in previous.forecasts if f.event_id in known],
        analyses=[a for a in previous.analyses if a.event_id in known], evaluation=previous.evaluation)
    publish_site(site, root)
    return site


def set_entrants(site_dir: str | Path, event_id: str, entrants: list[dict], *, target: str,
                 observed_at: datetime | None = None, source: str = "Operator verified entry list") -> None:
    root = Path(site_dir)
    site = SiteData.model_validate_json((root / "site.json").read_text(encoding="utf-8"))
    event = next(e for e in site.events if e.id == event_id)
    state = next(t for t in event.targets if t.target == target)
    if state.forecast_id:
        raise ValueError("cannot replace this target roster after its forecast is frozen")
    roster = [Driver.model_validate(d) for d in entrants]
    if len(roster) < 2:
        raise ValueError("a verified roster requires at least two entrants")
    known = {d.id: d for d in event.entrants}
    known.update({d.id: d for d in roster})
    event.entrants = list(known.values())
    state.entrant_ids = [d.id for d in roster]
    state.roster_source = source
    state.roster_observed_at = observed_at or datetime.now(UTC)
    publish_site(site, root)
