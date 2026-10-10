"""Completed observations must publish independently of forecast issuance."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from f1forecast import cli
from f1forecast.archive import load_event_snapshots
from f1forecast.contracts import ActualResult, Analysis, Event, EventTelemetry, SiteData
from f1forecast.operations import has_weekend_work, retain_published_observations, run_tick
from f1forecast.publication import publish_forecast, publish_site, verify_archive, verify_telemetry
from test_analysis_export import source_session
from test_weekend_rehearsal import Q_START, _site


@pytest.fixture
def weekend(tmp_path):
    site = _site()
    event = site.events[0].model_dump()
    event["entrants"] = []
    for target in event["targets"]:
        target.update(entrant_ids=[], roster_source=None, roster_observed_at=None)
    for kind, hours in (("SQ", -20), ("S", -3)):
        start = Q_START + timedelta(hours=hours)
        event["sessions"].append({
            "id": f"2026-01-{kind}", "kind": kind,
            "start": start, "end": start + timedelta(hours=1),
        })
    site.events = [Event.model_validate(event)]
    site_dir, archive = tmp_path / "site", tmp_path / "archive"
    registry = tmp_path / "registry.json"
    registry.write_text('{"schema_version":"1.0","models":[]}')
    publish_site(site, site_dir)
    return site, site_dir, archive, registry


def install_provider(monkeypatch, site, *, failures=(), partial=(), missing_telemetry=()):
    """Replace only external feeds; exercise real normalization/archive/export."""
    fetched = []

    class Provider:
        def fetch_session(self, year, event, kind):
            assert (year, event) == (2026, 1)
            fetched.append(kind)
            if kind in failures:
                raise RuntimeError(f"{kind} feed is delayed")
            session = next(s for s in site.events[0].sessions if s.kind == kind)
            raw = replace(
                source_session(kind), season=2026, event_id="2026-01",
                circuit_id="test_circuit", session_start=pd.Timestamp(session.start),
                session_end=pd.Timestamp(session.end), completed=kind not in partial,
            )
            if kind in missing_telemetry:
                raw.telemetry = {}
            return raw

    monkeypatch.setattr("f1forecast.ingestion.FastF1Provider", lambda *_: Provider())
    monkeypatch.setattr("f1forecast.providers.JolpicaProvider.fetch_races", lambda *_: [])
    monkeypatch.setattr("f1forecast.providers.JolpicaProvider.fetch_drivers", lambda *_: [])
    return fetched


def saved_site(site_dir):
    return SiteData.model_validate_json((site_dir / "site.json").read_text())


def test_morning_tick_publishes_practice_and_sprint_archive_without_models(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    clock = Q_START - timedelta(hours=6)
    fetched = install_provider(monkeypatch, site)
    assert has_weekend_work(site, clock)
    report = run_tick(site_dir, archive, registry, now=clock)
    persisted = saved_site(site_dir)
    assert report["errors"] == []
    assert set(fetched) == {"FP1", "SQ"}
    assert {m["session_type"] for m, _, _ in load_event_snapshots(archive, "2026-01")} == {
        "FP1", "SQ",
    }
    assert persisted.forecasts == []
    assert [t.state for t in persisted.events[0].targets] == ["scheduled", "scheduled"]
    assert all(t.entrant_ids == [] for t in persisted.events[0].targets)
    assert [d.code for d in persisted.events[0].entrants] == ["NOR"]
    assert persisted.analyses[0].drivers[0].practice_pace_s == 92.0
    telemetry = EventTelemetry.model_validate_json(
        (site_dir / "telemetry/2026-01.json").read_text()
    )
    assert telemetry.traces[0].session_id == "2026-01-FP1"
    assert verify_telemetry(persisted.analyses, site_dir) == 1

    before = {p: p.read_bytes() for p in site_dir.rglob("*.json")}
    run_tick(site_dir, archive, registry, now=clock + timedelta(minutes=10))
    assert len(fetched) == 2  # Complete cached sessions need no external calls.
    assert before == {p: p.read_bytes() for p in site_dir.rglob("*.json")}


def test_sprint_is_downloaded_when_complete_but_qualifying_is_not(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    fetched = install_provider(monkeypatch, site)
    run_tick(site_dir, archive, registry, now=Q_START - timedelta(hours=1))
    assert set(fetched) == {"FP1", "SQ", "S"}
    assert saved_site(site_dir).forecasts == []


def test_delayed_session_does_not_block_other_sessions_and_is_retried(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    fetched = install_provider(monkeypatch, site, failures={"SQ"})
    clock = Q_START - timedelta(hours=6)
    report = run_tick(site_dir, archive, registry, now=clock)
    assert set(fetched) == {"FP1", "SQ"}
    assert any("SQ feed is delayed" in e["reason"] for e in report["errors"])
    assert saved_site(site_dir).analyses[0].drivers[0].practice_pace_s == 92.0
    fetched = install_provider(monkeypatch, site)
    report = run_tick(site_dir, archive, registry, now=clock + timedelta(hours=1))
    assert fetched == ["SQ"]
    assert report["errors"] == []


def test_partial_practice_is_not_published_and_retries_after_finalization(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    clock = Q_START - timedelta(hours=22)
    fetched = install_provider(monkeypatch, site, partial={"FP1"})
    run_tick(site_dir, archive, registry, now=clock)
    assert fetched == ["FP1"]
    assert saved_site(site_dir).analyses == []
    assert saved_site(site_dir).events[0].entrants == []
    assert not (site_dir / "telemetry/2026-01.json").exists()
    fetched = install_provider(monkeypatch, site)
    run_tick(site_dir, archive, registry, now=clock + timedelta(minutes=10))
    assert fetched == ["FP1"]
    assert saved_site(site_dir).analyses[0].drivers[0].practice_pace_s == 92.0


def test_results_publish_even_when_forecast_was_unavailable(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    fetched = install_provider(monkeypatch, site)
    run_tick(site_dir, archive, registry, now=Q_START + timedelta(hours=2))
    persisted = saved_site(site_dir)
    assert set(fetched) == {"FP1", "SQ", "S", "Q"}
    assert persisted.events[0].targets[0].state == "unavailable"
    assert persisted.forecasts == []
    assert [(a.target, a.position) for a in persisted.analyses[0].actuals] == [("qualifying", 1)]


def test_session_refresh_preserves_issued_forecasts_and_frozen_rosters(weekend, monkeypatch):
    _, site_dir, archive, registry = weekend
    site = _site(q_issued=True)
    publish_forecast(site.forecasts[0], site_dir / "forecasts", now=Q_START - timedelta(minutes=40))
    publish_site(site, site_dir)
    before = {p: p.read_bytes() for p in (site_dir / "forecasts").rglob("*.json")}
    frozen = site.forecasts[0].model_dump()
    install_provider(monkeypatch, site)
    run_tick(site_dir, archive, registry, now=Q_START + timedelta(hours=2))
    persisted = saved_site(site_dir)
    assert persisted.forecasts[0].model_dump() == frozen
    assert persisted.events[0].targets[0].entrant_ids == ["a", "b"]
    assert {d.id for d in persisted.events[0].entrants} == {"a", "b", "nor"}
    assert before == {p: p.read_bytes() for p in (site_dir / "forecasts").rglob("*.json")}
    verify_archive(site_dir / "forecasts")


def test_finalized_practice_publishes_pace_without_optional_telemetry(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    install_provider(monkeypatch, site, missing_telemetry={"FP1"})
    report = run_tick(site_dir, archive, registry, now=Q_START - timedelta(hours=22))
    persisted = saved_site(site_dir)
    assert report["errors"] == []
    assert persisted.analyses[0].drivers[0].practice_pace_s == 92.0
    telemetry = EventTelemetry.model_validate_json(
        (site_dir / "telemetry/2026-01.json").read_text()
    )
    assert telemetry.traces == []


def test_cache_loss_and_failed_practice_refetch_preserve_published_observations(weekend, monkeypatch):
    site, site_dir, archive, registry = weekend
    clock = Q_START - timedelta(hours=22)
    install_provider(monkeypatch, site)
    run_tick(site_dir, archive, registry, now=clock)
    before = saved_site(site_dir).analyses[0]
    trace_bytes = (site_dir / "telemetry/2026-01.json").read_bytes()

    # A hosted runner can lose its raw-session cache between ticks.
    fresh_archive = archive.with_name("replacement-archive")
    install_provider(monkeypatch, site, failures={"FP1"})
    run_tick(site_dir, fresh_archive, registry, now=Q_START - timedelta(hours=6))
    after = saved_site(site_dir).analyses[0]
    assert after.drivers == before.drivers
    assert after.circuit_points == before.circuit_points
    assert before.source in after.source
    assert (site_dir / "telemetry/2026-01.json").read_bytes() == trace_bytes
    saved_bytes = (site_dir / "site.json").read_bytes()
    run_tick(site_dir, fresh_archive, registry, now=Q_START - timedelta(hours=5))
    assert (site_dir / "site.json").read_bytes() == saved_bytes

    # Missing practice must not prevent a later qualifying result from appearing.
    run_tick(site_dir, fresh_archive, registry, now=Q_START + timedelta(hours=2))
    after_q = saved_site(site_dir).analyses[0]
    assert after_q.drivers == before.drivers
    assert [(a.target, a.position) for a in after_q.actuals] == [("qualifying", 1)]

    race_archive = archive.with_name("race-only-archive")
    install_provider(monkeypatch, site, failures={"FP1", "SQ", "S", "Q"})
    run_tick(site_dir, race_archive, registry, now=Q_START + timedelta(days=1, hours=3))
    after_race = saved_site(site_dir).analyses[0]
    assert after_race.drivers == before.drivers
    assert {(a.target, a.status) for a in after_race.actuals} == {
        ("qualifying", "Finished"), ("race", "Disqualified"),
    }


def test_cli_does_not_require_approved_model_to_refresh_observations(weekend, monkeypatch, capsys):
    site, site_dir, archive, registry = weekend
    clock = Q_START - timedelta(hours=6)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz or UTC)

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr("f1forecast.operations.datetime", Clock)
    install_provider(monkeypatch, site)
    monkeypatch.setattr("sys.argv", [
        "f1forecast", "tick", "--site-data", str(site_dir),
        "--archive", str(archive), "--registry", str(registry),
    ])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report.get("status") != "idle"
    assert saved_site(site_dir).analyses[0].drivers[0].practice_pace_s == 92.0


def test_retention_keeps_missing_driver_results_but_accepts_corrections(tmp_path):
    previous = Analysis(
        event_id="2026-01", kind="observed", source="Earlier finalized observations",
        circuit_points=[], drivers=[], actuals=[
            ActualResult(target="qualifying", driver_id="a", position=1, status="Finished"),
            ActualResult(target="qualifying", driver_id="b", position=2, status="Finished"),
        ],
    )
    current = previous.model_copy(deep=True, update={
        "actuals": [ActualResult(
            target="qualifying", driver_id="a", position=None, status="Disqualified",
        )],
    })
    telemetry = EventTelemetry(event_id="2026-01", traces=[])
    retain_published_observations(previous, current, telemetry, tmp_path)
    assert {(a.driver_id, a.position, a.status) for a in current.actuals} == {
        ("a", None, "Disqualified"), ("b", 2, "Finished"),
    }


@pytest.mark.parametrize("clock", [Q_START - timedelta(days=2), Q_START + timedelta(days=9)])
def test_observation_refresh_is_bounded_to_recent_completed_sessions(weekend, monkeypatch, clock):
    site, site_dir, archive, registry = weekend
    fetched = install_provider(monkeypatch, site)
    assert not has_weekend_work(site, clock)
    run_tick(site_dir, archive, registry, now=clock)
    assert fetched == []
    assert saved_site(site_dir).analyses == []
