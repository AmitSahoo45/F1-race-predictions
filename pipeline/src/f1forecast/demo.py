"""Explicitly synthetic, reproducible UI fixtures. Never evaluation evidence."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .contracts import Event, EventTelemetry, Forecast, SiteData
from .probabilities import sample_orders
from .publication import publish_forecast, publish_site, publish_telemetry


def create_demo(destination: str | Path) -> SiteData:
    destination = Path(destination)
    reference_path = destination / "reference-circuit.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
    created = datetime(2026, 9, 1, 9, tzinfo=UTC)
    names = ["Alex Vega", "Noah Cross", "Leo Laurent", "Kai Moreno", "Eli Sterling",
             "Max Arden", "Luca Vale", "Finn Sato", "Theo Silva", "Oscar Reed",
             "Jules Costa", "Nico Park", "Milo Hart", "Rory Chen", "Arlo Quinn",
             "Enzo Stone", "Felix Lane", "Remy Cole", "Sami Fox", "Dante West"]
    teams = ["Vertex", "Helix", "Forma", "Atlas", "Kinetic", "Veloce", "Vector", "Nova", "Flux", "Aero"]
    colors = ["#d8fa60", "#78ccff", "#ff9985", "#d4a5ff", "#6be7c8", "#f5c773",
              "#c0c8d0", "#ff91c7", "#80bf9b", "#aab4ff"]
    entrants = [{"id": f"driver-{i + 1:02}", "code": name.split()[1][:3].upper(),
                 "name": name, "team": teams[i // 2], "color": colors[i // 2]}
                for i, name in enumerate(names)]
    events = []
    for ident, title, day, state in [("demonstration", "Demonstration Grand Prix", 1, "reconstructed"),
                                    ("missing-data", "Unavailable forecast example", 8, "unavailable"),
                                    ("upcoming-example", "Scheduled weekend example", 98, "scheduled")]:
        start = created + timedelta(days=day, hours=3)
        sessions = [{"id": f"{ident}-{kind.lower()}", "kind": kind,
                     "start": start + timedelta(hours=offset),
                     "end": start + timedelta(hours=offset + duration)}
                    for kind, offset, duration in [("FP1", -28, 1), ("FP2", -24, 1),
                                                   ("FP3", -3, 1), ("Q", 0, 1), ("R", 24, 2)]]
        targets = [{"target": target, "session_id": f"{ident}-{code}",
                    "cutoff_at": sessions[index]["start"] - timedelta(minutes=30),
                    "state": state,
                    "reason": "Required qualifying data was not available before the cutoff."
                    if state == "unavailable" else None,
                    "forecast_id": f"demo-{target}" if state == "reconstructed" else None}
                   for target, code, index in [("qualifying", "q", 3), ("race", "r", 4)]]
        events.append(Event.model_validate({"id": ident, "season": 2026,
            "round": len(events) + 1, "name": title, "circuit": "Bahrain reference layout",
            "country": "Demonstration", "timezone": "UTC", "sessions": sessions,
            "entrants": entrants, "targets": targets}))
    forecasts = []
    for target in events[0].targets:
        scores = np.linspace(1.8, -1.8, len(entrants))
        if target.target == "race":
            scores[0], scores[1] = scores[1], scores[0]
        samples = sample_orders([d["id"] for d in entrants], scores, temperature=.65, seed=42)
        rows = []
        ranking = np.argsort(samples.expected_positions, kind="stable")
        for rank, idx in enumerate(ranking, start=1):
            probabilities = samples.position_probabilities[idx]
            rows.append({"driver_id": entrants[idx]["id"], "rank": rank,
                "expected_position": float(samples.expected_positions[idx]),
                "p10": int(np.searchsorted(np.cumsum(probabilities), .1)) + 1,
                "p90": int(np.searchsorted(np.cumsum(probabilities), .9)) + 1,
                "win_probability": float(probabilities[0]),
                "podium_probability": float(sum(probabilities[:3])),
                "top10_probability": float(sum(probabilities[:10])),
                "position_probabilities": probabilities.tolist(), "explanations": []})
        forecasts.append(Forecast.model_validate({"id": f"demo-{target.target}",
            "event_id": events[0].id, "target": target.target, "kind": "demonstration",
            "generated_at": created, "cutoff_at": target.cutoff_at,
            "training_cutoff": created - timedelta(days=31), "model_version": "synthetic-demo-v1",
            "model_kind": "baseline", "sample_count": 20000, "seed": 42,
            "coverage": {"available_sessions": ["FP1", "FP2", "FP3"], "missing_sessions": [],
                "summary": "Synthetic inputs illustrate the interface; these are not F1 predictions.",
                "flags": ["synthetic", "not-validated"]}, "drivers": rows}))
    driver_analysis, traces = [], []
    for i, driver in enumerate(entrants):
        points = [{"distance_m": float(x), "speed_kph": float(195 + 105 * np.cos(x / 510 + i * .015)),
                   "throttle_pct": float(np.clip(55 + 45 * np.cos(x / 510 + i * .015), 0, 100)),
                   "brake": bool(np.cos(x / 510 + i * .015) < -.65)} for x in np.linspace(0, 5200, 80)]
        driver_analysis.append({"driver_id": driver["id"], "practice_pace_s": 91.2 + .12 * i,
            "long_run_pace_s": 96.1 + .09 * i,
            "stints": [{"compound": "MEDIUM", "laps": 12, "pace_s": 96.1 + .09 * i},
                       {"compound": "SOFT", "laps": 4, "pace_s": 91.2 + .12 * i}]})
        traces.append({"driver_id": driver["id"], "session_id": f"{events[0].id}-fp3",
                       "points": points})
    telemetry = EventTelemetry.model_validate({"event_id": events[0].id, "traces": traces})
    site = SiteData.model_validate({"generated_at": created,
        "demo_notice": "Demonstration data: fictional drivers, synthetic forecasts and telemetry. No predictive accuracy has been established.",
        "events": events, "forecasts": forecasts,
        "analyses": [{"event_id": events[0].id, "kind": "demonstration",
            "source": "Synthetic drivers and telemetry. Circuit artwork: " + reference.get(
                "source", "unavailable") + ".",
            "circuit_points": reference.get("points", []), "drivers": driver_analysis, "actuals": []}],
        "evaluation": {"status": "pending", "summary": "Validation comes before claims.",
            "seasons": [], "comparison": [], "limitations": [
                "No historical evaluation or prospective forecast has been completed in this demonstration.",
                "Public telemetry omits team fuel loads, setups and many sensor channels.",
                "2026 performance must be evaluated separately from earlier technical regulations.",
                "Top-ten probability is a points-position proxy, not the probability of awarded points."]}})
    destination = Path(destination)
    for forecast in forecasts:
        publish_forecast(forecast, destination / "forecasts")
    publish_telemetry(telemetry, destination)
    publish_site(site, destination)
    return site
