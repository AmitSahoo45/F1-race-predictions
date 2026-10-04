"""Offline boundary tests for provider ingestion and immutable local snapshots."""

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from f1forecast.archive import load_summaries, write_snapshot
from f1forecast.ingestion import ingest_session, normalize_session
from f1forecast.providers import (
    FastF1Provider,
    JolpicaProvider,
    RawSession,
    _fastest_lap_trace,
    session_is_complete,
)

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.mark.parametrize("placeholder", [None, "None", " nan ", "NULL", ""])
def test_team_placeholder_laps_use_same_session_result_identity(placeholder):
    raw = raw_session()
    raw.laps["Team"] = placeholder
    raw.results["TeamName"] = "RB F1 Team"
    summary, tables, coverage = normalize_session(raw, NOW)
    assert summary.team_id.tolist() == ["racing_bulls"]
    assert coverage["missing_team_drivers"] == []
    assert tables["results"].TeamName.tolist() == ["RB F1 Team"]
    pd.testing.assert_series_equal(tables["laps"].Team, raw.laps.Team)


def test_missing_team_identity_is_explicit_and_keeps_zero_lap_entrant():
    raw = raw_session()
    raw.laps["Team"] = "None"
    raw.results["TeamName"] = None
    raw.results.loc[1] = {"Abbreviation": "RES", "DriverNumber": "99", "TeamName": None}
    summary, _, coverage = normalize_session(raw, NOW)
    assert set(summary.driver_code) == {"NOR", "RES"}
    assert summary.team_id.isna().all()
    assert coverage["missing_team_drivers"] == ["NOR", "RES"]
    assert coverage["complete"] is False


def test_team_alias_normalization_preserves_2024_rb_identity():
    raw = raw_session()
    raw.season = 2024
    raw.laps["Team"] = "RB"
    summary, _, _ = normalize_session(raw, NOW)
    assert summary.team_id.tolist() == ["rb"]


def test_fastest_lap_trace_preserves_car_channels_and_source_positions():
    class Car(pd.DataFrame):
        @property
        def _constructor(self):
            return Car

        def add_distance(self):
            result = self.copy()
            result["Distance"] = [0.0, 100.0]
            return result

        def add_relative_distance(self):
            return self

        def add_driver_ahead(self):
            raise AssertionError("unused driver-ahead computation must not run")

    class Joined(pd.DataFrame):
        def slice_by_lap(self, _lap, *, interpolate_edges=False):
            assert interpolate_edges
            return self

    class Positions(pd.DataFrame):
        def merge_channels(self, car):
            joined = pd.merge_asof(
                pd.DataFrame(car).sort_values("Date"),
                pd.DataFrame(self).sort_values("Date"),
                on="Date",
                direction="nearest",
            )
            return Joined(joined)

    class Lap:
        def get_car_data(self, *, pad, pad_side):
            assert (pad, pad_side) == (1, "both")
            return Car(
                {
                    "Date": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z"]),
                    "Speed": [200, 250],
                    "Throttle": [50, 100],
                    "Brake": [False, True],
                }
            )

        def get_pos_data(self, *, pad, pad_side):
            assert (pad, pad_side) == (1, "both")
            return Positions(
                {
                    "Date": pd.to_datetime(
                        ["2025-01-01T00:00:00.1Z", "2025-01-01T00:00:01.1Z"]
                    ),
                    "X": [10.0, 20.0],
                    "Y": [30.0, 40.0],
                }
            )

        def get_telemetry(self):
            raise AssertionError("expensive merged convenience API must not run")

    trace = _fastest_lap_trace(Lap())
    assert trace["Distance"].tolist() == [0.0, 100.0]
    assert trace["Speed"].tolist() == [200, 250]
    assert trace["Throttle"].tolist() == [50, 100]
    assert trace["Brake"].tolist() == [False, True]
    assert trace[["X", "Y"]].values.tolist() == [[10.0, 30.0], [20.0, 40.0]]


def test_fastf1_provider_paces_fetches_across_process_restarts(tmp_path, monkeypatch):
    budget = tmp_path / ".fastf1-pacing.json"
    budget.write_text(
        json.dumps({"last_fetch_start_utc": datetime.now(UTC).isoformat()}), encoding="utf-8"
    )
    sleeps = []
    monkeypatch.setattr("f1forecast.providers.time.sleep", sleeps.append)
    FastF1Provider(tmp_path)._pace_fetch()
    assert len(sleeps) == 1
    assert 89 <= sleeps[0] <= 90
    assert "last_fetch_start_utc" in json.loads(budget.read_text(encoding="utf-8"))


def test_fastf1_provider_retries_hard_rate_limit_after_window(tmp_path, monkeypatch):
    from fastf1.exceptions import RateLimitExceededError

    attempts = []
    sleeps = []

    class Session:
        def load(self, **_kwargs):
            attempts.append("load")
            if len(attempts) == 1:
                raise RateLimitExceededError("any API: 500 calls/h")

    monkeypatch.setattr("fastf1.get_session", lambda *_args: Session())
    monkeypatch.setattr("f1forecast.providers.time.sleep", sleeps.append)
    FastF1Provider(tmp_path)._load_session(2025, 1, "FP1")
    assert attempts == ["load", "load"]
    assert sleeps == [3601]


def test_fastf1_same_practice_driver_list_keeps_zero_lap_entrant(tmp_path, monkeypatch):
    class Session:
        api_path = "/static/2026/2026-03-06_Australian_Grand_Prix/2026-03-06_Practice_1/"

    source = {
        "14": {
            "RacingNumber": "14",
            "Tla": "ALO",
            "TeamName": "Aston Martin",
            "TeamColour": "229971",
            "FirstName": "Fernando",
            "LastName": "Alonso",
        },
        "18": {
            "RacingNumber": "18",
            "Tla": "STR",
            "TeamName": "Aston Martin",
            "TeamColour": "229971",
            "FirstName": "Lance",
            "LastName": "Stroll",
        },
    }
    calls = []
    monkeypatch.setattr("fastf1.get_session", lambda *_args: Session())
    monkeypatch.setattr(
        "fastf1.get_event_schedule",
        lambda _year, **_kwargs: SimpleNamespace(
            get_event_by_round=lambda _round: SimpleNamespace(get_session=lambda _kind: Session())
        ),
    )
    monkeypatch.setattr("fastf1.Cache.enable_cache", lambda *_args: None)
    monkeypatch.setattr(
        "fastf1._api.driver_info", lambda path: calls.append(path) or source
    )
    rows, provenance = FastF1Provider(tmp_path).fetch_driver_list(2026, 1, "FP1")
    assert calls == [Session.api_path]
    assert rows["Abbreviation"].tolist() == ["ALO", "STR"]
    assert rows["DriverNumber"].tolist() == ["14", "18"]
    assert rows["FullName"].tolist() == ["Fernando Alonso", "Lance Stroll"]
    assert rows["TeamColor"].tolist() == ["229971", "229971"]
    assert "Position" not in rows and "Status" not in rows
    assert provenance["url"].endswith("/DriverList.jsonStream")
    assert len(provenance["sha256"]) == 64
    assert provenance["hash_kind"] == "canonical_driver_info_json"


def test_fastf1_driver_list_reuses_one_schedule_lookup_per_year(tmp_path, monkeypatch):
    class Session:
        api_path = "/static/example/"

    class Event:
        def get_session(self, _kind):
            return Session()

    class Schedule:
        def get_event_by_round(self, _round):
            return Event()

    schedule_calls = []
    monkeypatch.setattr(
        "fastf1.get_event_schedule", lambda year, **_kwargs: schedule_calls.append(year) or Schedule()
    )
    monkeypatch.setattr("fastf1.get_session", lambda *_args: Session())
    monkeypatch.setattr("fastf1.Cache.enable_cache", lambda *_args: None)
    monkeypatch.setattr(
        "fastf1._api.driver_info",
        lambda _path: {"1": {"RacingNumber": "1", "Tla": "AAA"}},
    )
    provider = FastF1Provider(tmp_path)
    provider.fetch_driver_list(2031, 1, "FP1")
    provider.fetch_driver_list(2031, 1, "FP2")
    assert schedule_calls == [2031]


def test_fastf1_schedule_uses_utc_session_date_and_records_source_hash(tmp_path, monkeypatch):
    class Event:
        def get_session_date(self, kind, *, utc):
            assert (kind, utc) == ("R", True)
            return pd.Timestamp("2024-11-24T06:00:00")

        def to_dict(self):
            return {"RoundNumber": 22, "Session5DateUtc": pd.Timestamp("2024-11-24T06:00:00")}

    class Schedule:
        def get_event_by_round(self, round_number):
            assert round_number == 22
            return Event()

    def schedule(year, *, backend=None):
        assert backend == "fastf1"  # Provenance names this backend's JSON URL.
        return Schedule()

    monkeypatch.setattr("fastf1.get_event_schedule", schedule)
    monkeypatch.setattr("fastf1.Cache.enable_cache", lambda *_args: None)
    start, source = FastF1Provider(tmp_path).fetch_schedule(2024, 22, "R")
    assert start == pd.Timestamp("2024-11-24T06:00:00Z")
    assert source["url"].endswith("/schedule_2024.json")
    assert len(source["sha256"]) == 64
    assert source["digest_basis"] == "canonical_event_schedule_row"


def test_alternate_source_file_hashes_and_completion_evidence_are_archived(tmp_path):
    class Provider:
        def fetch_session(self, *_args):
            raw = raw_session()
            raw.source = "Pinned public timing JSON"
            raw.source_files = [
                {
                    "url": "https://example.test/data.json",
                    "sha256": "a" * 64,
                    "bytes": 1234,
                    "commit": "b" * 40,
                }
            ]
            raw.completion_evidence = "Jolpica final classification"
            raw.driver_ids = {"NOR": "norris"}
            return raw

    snapshot = ingest_session(
        2025, 1, "FP1", tmp_path, fastf1_provider=Provider(), retrieved_at=NOW
    )
    assert snapshot.manifest["source_files"][0]["sha256"] == "a" * 64
    assert snapshot.manifest["coverage"]["completion_evidence"] == "Jolpica final classification"
    assert snapshot.summary.iloc[0]["driver_id"] == "norris"
    assert snapshot.sources["schedule"] == "Pinned public timing JSON event metadata"
    assert snapshot.manifest["extraction_version"] is None


def raw_session(*, duplicate=False, missing_channels=False):
    laps = pd.DataFrame(
        [
            {
                "Driver": "NOR",
                "DriverNumber": "4",
                "Team": "McLaren",
                "LapNumber": 1,
                "LapTime": np.timedelta64(90, "s"),
                "LapStartTime": np.timedelta64(5, "m"),
                "Compound": "MEDIUM",
                "TyreLife": 2,
                "IsAccurate": True,
                "Deleted": False,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            },
            {
                "Driver": "NOR",
                "DriverNumber": "4",
                "Team": "McLaren",
                "LapNumber": 2,
                "LapTime": np.timedelta64(91, "s"),
                "LapStartTime": np.timedelta64(7, "m"),
                "Compound": "MEDIUM",
                "TyreLife": 3,
                "IsAccurate": True,
                "Deleted": False,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            },
        ]
    )
    if duplicate:
        laps.loc[len(laps)] = laps.iloc[0]
    results = pd.DataFrame(
        [
            {
                "Abbreviation": "NOR",
                "DriverNumber": "4",
                "TeamName": "McLaren",
                "Position": 1,
                "Status": "Finished",
            }
        ]
    )
    telemetry = (
        {}
        if missing_channels
        else {
            "NOR": pd.DataFrame(
                {
                    "Distance": [0.0, 100.0],
                    "Speed": [200.0, 250.0],
                    "Throttle": [50.0, 100.0],
                    "Brake": [False, True],
                    "X": [1.0, 2.0],
                    "Y": [3.0, 4.0],
                }
            )
        }
    )
    return RawSession(
        season=2025,
        event_id="2025-01",
        circuit_id="albert_park",
        session_type="FP1",
        session_start=pd.Timestamp("2025-03-14T01:30:00Z"),
        session_end=pd.Timestamp("2025-03-14T02:30:00Z"),
        laps=laps,
        results=results,
        weather=pd.DataFrame([{"AirTemp": 22.0, "TrackTemp": 34.0}]),
        telemetry=telemetry,
        source="FastF1",
        completed=True,
    )


def test_jolpica_paginates_with_identifying_agent_and_caches(tmp_path):
    requested = []

    def respond(request):
        requested.append(request)
        offset = int(request.url.params["offset"])
        races = [{"round": str(i + 1)} for i in range(offset, min(offset + 100, 101))]
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "total": "101",
                    "limit": "100",
                    "offset": str(offset),
                    "RaceTable": {"Races": races},
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = JolpicaProvider(tmp_path, client=client, min_interval_s=0)
    assert len(provider.fetch_races(2025)) == 101
    assert len(provider.fetch_races(2025)) == 101
    assert [int(r.url.params["offset"]) for r in requested] == [0, 100]
    assert all(int(r.url.params["limit"]) <= 100 for r in requested)
    assert all("F1Forecast/" in r.headers["user-agent"] for r in requested)


def test_fastf1_finalised_status_proves_completion():
    assert session_is_complete(
        pd.DataFrame({"Status": ["Started", "Finished", "Finalised", "Ends"]})
    )
    assert not session_is_complete(pd.DataFrame({"Status": ["Started", "Finished"]}))


def test_jolpica_429_respects_retry_after_and_budget(tmp_path):
    attempts = 0
    waits = []

    def respond(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json={
                "MRData": {"total": "0", "limit": "100", "offset": "0", "RaceTable": {"Races": []}}
            },
        )

    provider = JolpicaProvider(
        tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        min_interval_s=0,
        sleep=waits.append,
        max_requests=2,
    )
    assert provider.fetch_races(2025) == []
    assert waits == [2.0]
    assert attempts == 2
    with pytest.raises(RuntimeError, match="budget"):
        provider.fetch_races(2026)


def test_jolpica_stale_cache_revalidates_but_fresh_cache_does_not(tmp_path):
    attempts = 0

    def respond(_request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "total": "1",
                    "limit": "100",
                    "offset": "0",
                    "RaceTable": {"Races": [{"round": str(attempts)}]},
                }
            },
        )

    provider = JolpicaProvider(
        tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        min_interval_s=0,
        cache_ttl_s=3600,
    )
    assert provider.fetch_races(2025)[0]["round"] == "1"
    assert provider.fetch_races(2025)[0]["round"] == "1"
    cached = next(tmp_path.glob("2025_*.json"))
    os.utime(cached, (1, 1))
    assert provider.fetch_races(2025)[0]["round"] == "2"
    assert attempts == 2


@pytest.mark.parametrize(
    "target,route,key", [("Q", "qualifying", "QualifyingResults"), ("R", "results", "Results")]
)
def test_jolpica_classification_endpoint_paginates_nested_race_rows(tmp_path, target, route, key):
    requests = []

    def respond(request):
        requests.append(request)
        offset = int(request.url.params["offset"])
        items = [{"position": str(index + 1)} for index in range(offset, min(offset + 100, 101))]
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "total": "101",
                    "limit": "100",
                    "offset": str(offset),
                    "RaceTable": {"Races": [{key: items}]},
                }
            },
        )

    provider = JolpicaProvider(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(respond)), min_interval_s=0
    )
    rows = provider.fetch_results(2025, 4, target)
    assert len(rows) == 101
    assert [int(request.url.params["offset"]) for request in requests] == [0, 100]
    assert all(request.url.path.endswith(f"/2025/4/{route}.json") for request in requests)


def test_normalization_preserves_missing_channels_and_availability():
    summary, tables, coverage = normalize_session(raw_session(missing_channels=True), NOW)
    row = summary.iloc[0]
    assert row["event_id"] == "2025-01"
    assert row["session_id"] == "2025-01-FP1"
    assert row["driver_id"] == "nor"
    assert row["pace_s"] == 90.5
    assert row["usable_laps"] == 2
    assert pd.isna(row["mean_speed"])
    assert row["available_at"] == pd.Timestamp(NOW)
    assert row["provenance"] == "retrospective"
    assert coverage["missing_telemetry_drivers"] == ["NOR"]
    assert tables["telemetry"].empty


def test_normalization_rejects_duplicate_lap_identity():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_session(raw_session(duplicate=True), NOW)


@pytest.mark.parametrize("accurate", [False, True])
def test_interrupted_race_lap_is_preserved_but_excluded_from_pace(accurate):
    # Pinned 2022 Monaco race: ALB lap 29 lasted 1297.333 s (IsAccurate=False).
    raw = raw_session()
    raw.session_type = "R"
    raw.laps["LapTime"] = pd.to_timedelta([90.0, 1297.333], unit="s")
    raw.laps.loc[1, "IsAccurate"] = accurate
    summary, tables, coverage = normalize_session(raw, NOW)
    assert tables["laps"]["lap_seconds"].tolist() == [90.0, 1297.333]
    assert tables["laps"]["usable"].tolist() == [True, False]
    assert summary.iloc[0]["pace_s"] == 90.0
    assert summary.iloc[0]["usable_laps"] == 1
    assert coverage["out_of_pace_range_laps"] == 1


@pytest.mark.parametrize("seconds", [-1, 86400])
def test_invalid_lap_duration_still_rejects_source_units(seconds):
    raw = raw_session()
    raw.laps["LapTime"] = pd.to_timedelta([90, seconds], unit="s")
    with pytest.raises(ValueError, match="lap time seconds") as error:
        normalize_session(raw, NOW)
    assert type(error.value).__name__ == "SourceValidationError"


def test_duplicate_lap_timestamp_is_an_expected_source_failure():
    raw = raw_session()
    raw.laps.loc[1, "LapStartTime"] = raw.laps.loc[0, "LapStartTime"]
    with pytest.raises(ValueError, match="duplicate lap timestamp") as error:
        normalize_session(raw, NOW)
    assert type(error.value).__name__ == "SourceValidationError"


def test_inaccurate_zero_timestamp_placeholders_are_retained_and_excluded():
    # Pinned Spain 2025 FP3 repeats zero LapStartTime only on inaccurate source rows.
    raw = raw_session()
    extra = raw.laps.iloc[[1]].copy()
    extra["LapNumber"] = 3
    raw.laps = pd.DataFrame([*raw.laps.to_dict("records"), *extra.to_dict("records")])
    raw.laps["LapStartTime"] = pd.to_timedelta([300, 0, 0], unit="s")
    raw.laps["IsAccurate"] = [True, False, False]
    summary, tables, coverage = normalize_session(raw, NOW)
    assert summary.iloc[0]["pace_s"] == 90.0
    assert tables["laps"]["LapStartTime"].dt.total_seconds().tolist() == [300, 0, 0]
    assert tables["laps"]["usable"].tolist() == [True, False, False]
    assert coverage["invalid_lap_timestamp_rows"] == 2


def test_corrupt_driver_trace_is_preserved_separately_without_losing_valid_session(tmp_path):
    raw = raw_session()
    raw.laps = pd.DataFrame([
        *raw.laps.to_dict("records"),
        *raw.laps.assign(Driver="PIA", DriverNumber="81").to_dict("records"),
    ])
    raw.telemetry["PIA"] = raw.telemetry["NOR"].copy()
    # Pinned Belgium 2026 FP2 GAS lap 12 has distances down to -4209.187 m.
    raw.telemetry["NOR"].loc[0, "Distance"] = -4209.187175981231
    summary, tables, coverage = normalize_session(raw, NOW)
    summary = summary.set_index("driver_code")
    assert pd.isna(summary.loc["NOR", "mean_speed"])
    assert summary.loc["PIA", "mean_speed"] == 225.0
    assert set(tables["telemetry"]["driver_id"]) == {"pia"}
    assert tables["rejected_telemetry"]["Distance"].tolist() == [-4209.187175981231, 100.0]
    assert set(tables["rejected_telemetry"]["driver_code"]) == {"NOR"}
    assert coverage["missing_telemetry_drivers"] == ["NOR"]
    assert "distance metres" in coverage["rejected_telemetry_drivers"]["NOR"]
    assert coverage["session_complete"]
    assert not coverage["complete"]
    manifest = write_snapshot(
        tmp_path, {"session_id": "2025-01-FP1", "source": "FastF1",
                   "retrieved_at": NOW.isoformat(), "coverage": coverage},
        summary.reset_index(), tables,
    )
    preserved = pd.read_parquet(
        tmp_path / "snapshots" / manifest["snapshot_id"] / "rejected_telemetry.parquet"
    )
    assert preserved["Distance"].tolist() == [-4209.187175981231, 100.0]


def test_fastf1_pacing_uses_injected_deadline_sleep(tmp_path):
    budget = tmp_path / ".fastf1-pacing.json"
    budget.write_text(
        json.dumps({"last_fetch_start_utc": datetime.now(UTC).isoformat()}), encoding="utf-8"
    )
    sleeps = []
    provider = FastF1Provider(tmp_path, sleep=sleeps.append)
    provider._pace_fetch()
    assert len(sleeps) == 1
    assert 89 <= sleeps[0] <= 90


def test_normalization_uses_telemetry_units_and_real_geometry():
    summary, tables, coverage = normalize_session(raw_session(), NOW)
    row = summary.iloc[0]
    assert row["mean_speed"] == 225.0
    assert row["mean_throttle"] == 0.75
    assert row["brake_fraction"] == 0.5
    assert tables["telemetry"]["speed_kph"].tolist() == [200.0, 250.0]
    assert tables["circuit"]["x"].tolist() == [1.0, 2.0]
    assert coverage["missing_telemetry_drivers"] == []


def test_unresolved_identity_preserves_trace_without_attaching_it_to_a_driver(tmp_path):
    raw = raw_session()
    raw.driver_ids = {"NOR": None, "PIA": "piastri"}
    raw.laps = pd.DataFrame([
        *raw.laps.to_dict("records"),
        *raw.laps.assign(Driver="PIA", DriverNumber="81").to_dict("records"),
    ])
    raw.telemetry["PIA"] = raw.telemetry["NOR"].copy()

    summary, tables, coverage = normalize_session(raw, NOW)
    by_code = summary.set_index("driver_code")
    assert pd.isna(by_code.loc["NOR", "driver_id"])
    assert pd.isna(by_code.loc["NOR", "mean_speed"])
    assert by_code.loc["NOR", "pace_s"] == 90.5
    assert by_code.loc["PIA", "driver_id"] == "piastri"
    assert by_code.loc["PIA", "mean_speed"] == 225.0
    assert set(tables["telemetry"]["driver_id"]) == {"piastri"}
    assert set(tables["rejected_telemetry"]["driver_code"]) == {"NOR"}
    assert tables["rejected_telemetry"]["Speed"].tolist() == [200.0, 250.0]
    assert coverage["missing_telemetry_drivers"] == ["NOR"]
    assert "identity" in coverage["rejected_telemetry_drivers"]["NOR"]
    assert not coverage["complete"]

    manifest = write_snapshot(
        tmp_path,
        {"session_id": "2025-01-FP1", "source": "FastF1",
         "retrieved_at": NOW.isoformat(), "coverage": coverage},
        summary,
        tables,
    )
    preserved = pd.read_parquet(
        tmp_path / "snapshots" / manifest["snapshot_id"] / "rejected_telemetry.parquet"
    )
    assert preserved["Speed"].tolist() == [200.0, 250.0]
    assert preserved["driver_code"].tolist() == ["NOR", "NOR"]


def test_normalization_clamps_submetre_fastf1_distance_interpolation():
    raw = raw_session()
    raw.telemetry["NOR"].loc[0, "Distance"] = -0.02
    _, tables, _ = normalize_session(raw, NOW)
    assert tables["telemetry"]["distance_m"].tolist() == [0.0, 100.0]


def test_result_only_nonstarter_retains_status_without_inventing_laps():
    raw = raw_session()
    raw.results = pd.concat(
        [
            raw.results,
            pd.DataFrame(
                [
                    {
                        "Abbreviation": "PIA",
                        "DriverNumber": "81",
                        "TeamName": "McLaren",
                        "Position": pd.NA,
                        "Status": "Did not start",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary, _, coverage = normalize_session(raw, NOW)
    nonstarter = summary.set_index("driver_code").loc["PIA"]
    assert nonstarter["status"] == "Did not start"
    assert pd.isna(nonstarter["position"])
    assert nonstarter["usable_laps"] == 0
    assert coverage["missing_lap_drivers"] == ["PIA"]


def test_stationary_telemetry_may_repeat_distance_with_distinct_timestamps():
    raw = raw_session()
    raw.telemetry["NOR"] = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-14T01:32:00Z", "2025-03-14T01:32:01Z"]),
            "Distance": [0.0, 0.0],
            "Speed": [0.0, 0.0],
            "Throttle": [0.0, 0.0],
            "Brake": [True, True],
        }
    )
    _, tables, _ = normalize_session(raw, NOW)
    assert tables["telemetry"]["distance_m"].tolist() == [0.0, 0.0]


def test_small_throttle_overshoot_is_capped_to_fraction():
    raw = raw_session()
    raw.telemetry["NOR"].loc[1, "Throttle"] = 104.0
    summary, tables, _ = normalize_session(raw, NOW)
    assert tables["telemetry"]["throttle_fraction"].tolist() == [0.5, 1.0]
    assert summary.iloc[0]["mean_throttle"] == 0.75


def test_qualifying_gap_compares_same_qualifying_phase():
    raw = raw_session()
    raw.session_type = "Q"
    raw.laps = pd.DataFrame(
        [
            *raw.laps.to_dict("records"),
            *raw.laps.assign(
                Driver="PIA", DriverNumber="81", LapTime=np.timedelta64(87, "s")
            ).to_dict("records"),
        ]
    )
    raw.results = pd.DataFrame(
        [
            {
                "Abbreviation": "NOR",
                "DriverNumber": "4",
                "TeamName": "McLaren",
                "Position": 1,
                "Status": "Finished",
                "Q1": np.timedelta64(90, "s"),
                "Q2": np.timedelta64(88, "s"),
                "Q3": pd.NaT,
            },
            {
                "Abbreviation": "PIA",
                "DriverNumber": "81",
                "TeamName": "McLaren",
                "Position": 18,
                "Status": "Finished",
                "Q1": np.timedelta64(91, "s"),
                "Q2": pd.NaT,
                "Q3": pd.NaT,
            },
        ]
    )
    summary, _, _ = normalize_session(raw, NOW)
    assert summary.set_index("driver_code").loc["NOR", "qualifying_gap"] == 0.0
    assert summary.set_index("driver_code").loc["PIA", "qualifying_gap"] == 1.0


@pytest.mark.parametrize("status", ["Did not start", "Disqualified", "DNS", "DSQ"])
def test_nonclassification_status_clears_numeric_provider_position(status):
    raw = raw_session()
    raw.session_type = "R"
    raw.results.loc[0, "Status"] = status
    summary, _, _ = normalize_session(raw, NOW)
    assert pd.isna(summary.iloc[0]["position"])
    assert summary.iloc[0]["status"] == status


def test_archive_is_resumable_and_keeps_original_snapshot(tmp_path):
    summary, tables, coverage = normalize_session(raw_session(), NOW)
    manifest = {
        "event_id": "2025-01",
        "session_id": "2025-01-FP1",
        "retrieved_at": NOW.isoformat(),
        "source": "FastF1",
        "coverage": coverage,
    }
    first = write_snapshot(tmp_path, manifest, summary, tables)
    second = write_snapshot(tmp_path, manifest, summary, tables)
    assert second == first
    loaded = load_summaries(tmp_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["source_snapshot_id"] == first["snapshot_id"]
    assert bool(loaded.iloc[0]["session_complete"])
    assert list(tmp_path.glob("snapshots/*/summary.parquet"))


def test_ingest_session_returns_archive_snapshot_without_refetch(tmp_path):
    class Provider:
        calls = 0

        def fetch_session(self, year, event, session):
            self.calls += 1
            assert (year, event, session) == (2025, 1, "FP1")
            return raw_session()

    provider = Provider()
    first = ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    second = ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    assert provider.calls == 1
    assert first.manifest["snapshot_id"] == second.manifest["snapshot_id"]
    assert len(second.summary) == 1


def test_refresh_preserves_old_snapshot_and_selects_newest(tmp_path):
    class Provider:
        calls = 0

        def fetch_session(self, *_args):
            self.calls += 1
            raw = raw_session()
            raw.laps.loc[0, "LapTime"] = np.timedelta64(89 if self.calls == 2 else 90, "s")
            return raw

    provider = Provider()
    first = ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    second = ingest_session(
        2025,
        1,
        "FP1",
        tmp_path,
        fastf1_provider=provider,
        retrieved_at=NOW + timedelta(hours=1),
        refresh=True,
    )
    assert provider.calls == 2
    assert first.manifest["snapshot_id"] != second.manifest["snapshot_id"]
    assert len(list((tmp_path / "snapshots").iterdir())) == 2
    loaded = load_summaries(tmp_path)
    assert loaded.iloc[0]["pace_s"] == 90.0
    assert loaded.iloc[0]["source_snapshot_id"] == second.manifest["snapshot_id"]


def test_incomplete_snapshot_is_refetched_on_next_ingest(tmp_path):
    class Provider:
        calls = 0

        def fetch_session(self, *_args):
            self.calls += 1
            return raw_session(missing_channels=self.calls == 1)

    provider = Provider()
    first = ingest_session(2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    second = ingest_session(
        2025, 1, "FP1", tmp_path, fastf1_provider=provider, retrieved_at=NOW + timedelta(hours=1)
    )
    assert first.manifest["coverage"]["complete"] is False
    assert second.manifest["coverage"]["complete"] is True
    assert provider.calls == 2


def test_explicit_prospective_capture_retains_actual_fetch_time():
    raw = raw_session()
    summary, _, _ = normalize_session(raw, NOW, provenance="prospective")
    assert summary.iloc[0]["provenance"] == "prospective"
    assert summary.iloc[0]["available_at"] == pd.Timestamp(NOW)


def test_sprint_qualifying_session_is_archived_without_race_label():
    raw = raw_session()
    raw.session_type = "SQ"
    summary, _, _ = normalize_session(raw, NOW)
    assert summary.iloc[0]["session_type"] == "SQ"
    assert summary.iloc[0]["session_id"] == "2025-01-SQ"


def test_jolpica_identity_join_uses_stable_driver_id(tmp_path):
    class FastProvider:
        def fetch_session(self, *_args):
            return raw_session()

    class Jolpica:
        def fetch_races(self, _year):
            return [{"round": "1", "Circuit": {"circuitId": "albert_park"}}]

        def fetch_drivers(self, _year):
            return [{"driverId": "norris", "code": "NOR", "permanentNumber": "4"}]

    snapshot = ingest_session(
        2025,
        1,
        "FP1",
        tmp_path,
        fastf1_provider=FastProvider(),
        jolpica_provider=Jolpica(),
        retrieved_at=NOW,
    )
    assert snapshot.summary.iloc[0]["driver_id"] == "norris"
    assert snapshot.manifest["coverage"]["unmatched_identity_drivers"] == []


def test_in_progress_race_snapshot_is_marked_incomplete_and_refetched(tmp_path):
    class Provider:
        calls = 0

        def fetch_session(self, *_args):
            self.calls += 1
            raw = raw_session()
            raw.session_type = "R"
            raw.completed = self.calls > 1
            return raw

    provider = Provider()
    first = ingest_session(2025, 1, "R", tmp_path, fastf1_provider=provider, retrieved_at=NOW)
    assert not bool(load_summaries(tmp_path).iloc[0]["session_complete"])
    second = ingest_session(
        2025, 1, "R", tmp_path, fastf1_provider=provider, retrieved_at=NOW + timedelta(hours=1)
    )
    assert first.manifest["coverage"]["session_complete"] is False
    assert second.manifest["coverage"]["session_complete"] is True
    assert provider.calls == 2
