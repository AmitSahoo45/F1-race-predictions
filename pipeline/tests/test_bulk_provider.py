"""Offline contract tests for pinned, historical public JSON ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pandas as pd
import pytest
from f1forecast.bulk_provider import COMMITS, TracingInsightsProvider
from f1forecast.ingestion import normalize_session


def _race() -> dict:
    return {
        "season": "2022",
        "round": "1",
        "raceName": "Bahrain Grand Prix",
        "Circuit": {"circuitId": "bahrain", "Location": {"country": "Bahrain"}},
        "FirstPractice": {"date": "2022-03-18", "time": "12:00:00Z"},
        "SecondPractice": {"date": "2022-03-18", "time": "15:00:00Z"},
        "ThirdPractice": {"date": "2022-03-19", "time": "12:00:00Z"},
        "Qualifying": {"date": "2022-03-19", "time": "15:00:00Z"},
        "date": "2022-03-20",
        "time": "15:00:00Z",
    }


def _laps() -> dict:
    return {
        "drv": ["LEC", "LEC", "VER", "VER"],
        "dNum": ["16", "16", "1", "1"],
        "team": ["Ferrari", "Ferrari", "Red Bull Racing", "Red Bull Racing"],
        "lap": [1, 2, 1, 2],
        "time": [92.0, 91.0, 94.0, 93.0],
        "iacc": [True, True, True, True],
        "del": [False, False, False, False],
        "pin": ["None"] * 4,
        "pout": ["None"] * 4,
        "life": [1, 2, 1, 2],
        "compound": ["SOFT"] * 4,
        "stint": [1, 1, 1, 1],
        "pb": [False, True, False, True],
        "lSD": [
            "2022-03-18T12:05:00",
            "2022-03-18T12:07:00",
            "2022-03-18T12:06:00",
            "2022-03-18T12:08:00",
        ],
        "lST": [300.0, 420.0, 360.0, 480.0],
        "sesT": [392.0, 511.0, 454.0, 573.0],
    }


class _Jolpica:
    def fetch_races(self, year):
        assert year == 2022
        return [_race()]

    def fetch_drivers(self, year):
        assert year == 2022
        return [
            {"code": "LEC", "driverId": "leclerc"},
            {"code": "VER", "driverId": "max_verstappen"},
        ]

    def fetch_results(self, year, round_number, target):
        assert (year, round_number) == (2022, 1)
        if target == "Q":
            return [
                {
                    "position": "1",
                    "number": "16",
                    "Driver": {"code": "LEC", "driverId": "leclerc"},
                    "Constructor": {"name": "Ferrari"},
                    "Q1": "1:31.000",
                    "Q2": "1:30.000",
                    "Q3": "1:29.000",
                },
                {
                    "position": "2",
                    "number": "1",
                    "Driver": {"code": "VER", "driverId": "max_verstappen"},
                    "Constructor": {"name": "Red Bull Racing"},
                    "Q1": "1:32.000",
                    "Q2": "1:31.000",
                    "Q3": "1:29.500",
                },
            ]
        return [
            {
                "position": "1",
                "number": "16",
                "status": "Finished",
                "Driver": {"code": "LEC", "driverId": "leclerc"},
                "Constructor": {"name": "Ferrari"},
            },
            {
                "position": "2",
                "number": "1",
                "status": "Finished",
                "Driver": {"code": "VER", "driverId": "max_verstappen"},
                "Constructor": {"name": "Red Bull Racing"},
            },
        ]


def _fixture_transport(*, rate_limit_first: bool = False, missing_weather: bool = False):
    requested = []
    first = True

    def respond(request):
        nonlocal first
        requested.append(str(request.url))
        if rate_limit_first and first:
            first = False
            return httpx.Response(429, headers={"Retry-After": "2"})
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "session_laptimes.json":
            payload = _laps()
        elif name == "weather.json" and not missing_weather:
            payload = {"wAT": [22.0, 24.0], "wTT": [31.0, 33.0]}
        elif name == "drivers.json":
            payload = {"drivers": [{"driver": "LEC"}, {"driver": "VER"}]}
        elif name in {"2_tel.json"}:
            payload = {
                "tel": {
                    "time": [0.0, 0.5],
                    "distance": [0.0, 100.0],
                    "speed": [200.0, 250.0],
                    "throttle": [50.0, 100.0],
                    "brake": [0, 1],
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                }
            }
        else:
            return httpx.Response(404)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(respond), requested


def _provider(transport, *, clock=None, sleep=None):
    clock = clock or [0.0]

    class PracticeRoster:
        def fetch_schedule(self, year, event, session_type):
            starts = {
                "FP1": "2022-03-18T12:00:00Z",
                "Q": "2022-03-19T15:00:00Z",
                "R": "2022-03-20T15:00:00Z",
            }
            return pd.Timestamp(starts[session_type]), {
                "url": "https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_2022.json",
                "sha256": "2" * 64, "bytes": 10,
                "source": "FastF1 UTC event schedule",
            }

        def fetch_driver_list(self, year, event, session_type):
            return pd.DataFrame(
                [
                    {"Abbreviation": "LEC", "DriverNumber": "16", "TeamName": "Ferrari"},
                    {"Abbreviation": "VER", "DriverNumber": "1", "TeamName": "Red Bull Racing"},
                ]
            ), {
                "url": "https://livetiming.formula1.com/static/2022/DriverList.jsonStream",
                "sha256": "0" * 64,
                "bytes": 2,
                "source": "FastF1 DriverList",
            }

    return TracingInsightsProvider(
        _Jolpica(),
        practice_roster=PracticeRoster(),
        client=httpx.Client(transport=transport),
        now=lambda: datetime(2026, 9, 26, tzinfo=UTC),
        monotonic=lambda: clock[0],
        sleep=sleep or (lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
    )


def test_practice_adapter_maps_pinned_laps_weather_and_only_fastest_telemetry():
    transport, requested = _fixture_transport()
    raw = _provider(transport).fetch_session(2022, 1, "FP1")
    assert raw.completed
    assert raw.source == f"TracingInsights/2022@{COMMITS[2022]}"
    assert raw.driver_ids == {"LEC": "leclerc", "VER": "max_verstappen"}
    assert len(raw.laps) == 4
    assert raw.laps["LapTime"].dt.total_seconds().tolist() == [92.0, 91.0, 94.0, 93.0]
    assert len(raw.telemetry) == 2
    assert raw.telemetry["LEC"]["Speed"].tolist() == [200.0, 250.0]
    assert len([url for url in requested if url.endswith("_tel.json")]) == 2
    assert all(f"/{COMMITS[2022]}/" in url for url in requested)
    assert len(raw.source_files) == 9  # JSON, traces, calendars, identities, DriverList.
    assert all(
        len(item["sha256"]) == 64 and item["url"].startswith("https://")
        for item in raw.source_files
    )
    summary, _, coverage = normalize_session(raw, datetime(2026, 9, 26, tzinfo=UTC))
    assert coverage["session_complete"]
    assert summary.set_index("driver_code").loc["LEC", "pace_s"] == 91.5
    assert summary.set_index("driver_code").loc["LEC", "mean_throttle"] == 0.75
    assert summary.set_index("driver_code").loc["LEC", "air_temp"] == 23.0


def test_qualifying_uses_jolpica_classification_without_telemetry_requests():
    transport, requested = _fixture_transport()
    raw = _provider(transport).fetch_session(2022, 1, "Q")
    assert raw.completed
    assert raw.telemetry == {}
    assert not any(url.endswith("_tel.json") for url in requested)
    assert raw.results["Position"].tolist() == [1, 2]
    assert raw.results["Q3"].dt.total_seconds().tolist() == [89.0, 89.5]
    summary, _, coverage = normalize_session(raw, datetime(2026, 9, 26, tzinfo=UTC))
    assert summary.set_index("driver_code").loc["LEC", "qualifying_gap"] == 0.0
    assert summary.set_index("driver_code").loc["VER", "qualifying_gap"] == 0.5
    assert coverage["complete"]
    assert coverage["telemetry_omission_reason"] == "target-session telemetry excluded"
    assert coverage["missing_telemetry_drivers"] == ["LEC", "VER"]


def test_bulk_adapter_fails_closed_on_missing_required_json():
    transport, _ = _fixture_transport(missing_weather=True)
    with pytest.raises(ValueError, match="weather.json"):
        _provider(transport).fetch_session(2022, 1, "FP1")


def test_bulk_adapter_honors_retry_after_and_records_content_hash():
    transport, requested = _fixture_transport(rate_limit_first=True)
    clock = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    raw = _provider(transport, clock=clock, sleep=sleep).fetch_session(2022, 1, "Q")
    assert len(requested) == 4  # One 429, then three required archive JSON files.
    assert any(wait >= 2.0 for wait in waits)
    assert any(wait == pytest.approx(0.25) for wait in waits)
    assert all(item["commit"] == COMMITS[2022] for item in raw.source_files if "commit" in item)
    assert all(len(item["sha256"]) == len(sha256().hexdigest()) for item in raw.source_files)


def test_practice_driver_list_keeps_zero_lap_entrant_without_target_results():
    transport, _ = _fixture_transport()
    provider = _provider(transport)

    class ThreeEntrants(type(provider.practice_roster)):
        def fetch_driver_list(self, year, event, session_type):
            return pd.DataFrame(
                [
                    {"Abbreviation": "LEC", "DriverNumber": "16", "TeamName": "Ferrari"},
                    {"Abbreviation": "VER", "DriverNumber": "1", "TeamName": "Red Bull Racing"},
                    {"Abbreviation": "ALO", "DriverNumber": "14", "TeamName": "Aston Martin"},
                ]
            ), {
                "url": "https://livetiming.formula1.com/DriverList.jsonStream",
                "sha256": "1" * 64,
                "bytes": 3,
                "source": "FastF1 DriverList",
            }

    provider.practice_roster = ThreeEntrants()
    raw = provider.fetch_session(2022, 1, "FP1")
    summary, _, coverage = normalize_session(raw, datetime(2026, 9, 26, tzinfo=UTC))
    absent = summary.set_index("driver_code").loc["ALO"]
    assert absent["usable_laps"] == 0
    assert pd.isna(absent["pace_s"])
    assert pd.isna(absent["position"])
    assert raw.source_labels["results"] == "FastF1 same-session DriverList"
    assert coverage["session_complete"]


def test_duplicate_bulk_lap_identity_is_an_expected_source_failure():
    payload = _laps()
    payload["lap"][1] = 1
    with pytest.raises(ValueError, match="duplicate bulk lap identity") as error:
        TracingInsightsProvider._laps(payload)
    assert type(error.value).__name__ == "SourceValidationError"


def test_null_string_bulk_identity_cannot_become_a_driver():
    payload = _laps()
    payload["drv"][0] = "None"
    with pytest.raises(ValueError, match="missing bulk driver identity"):
        TracingInsightsProvider._laps(payload)


def test_unexpected_practice_roster_code_error_remains_visible():
    transport, _ = _fixture_transport()
    provider = _provider(transport)

    class BrokenRoster(type(provider.practice_roster)):
        def fetch_driver_list(self, *_args):
            raise TypeError("unexpected programming error")

    provider.practice_roster = BrokenRoster()
    with pytest.raises(TypeError, match="unexpected programming error"):
        provider.fetch_session(2022, 1, "FP1")


def test_bulk_session_uses_verified_fastf1_utc_start_instead_of_wrong_calendar_day():
    transport, _ = _fixture_transport()
    provider = _provider(transport)

    class CorrectedSchedule(type(provider.practice_roster)):
        def fetch_schedule(self, year, event, session_type):
            _, record = super().fetch_schedule(year, event, session_type)
            # Mirrors the verified Vegas UTC-date discrepancy without guessing from laps.
            return pd.Timestamp("2022-03-21T15:00:00Z"), record

    provider.practice_roster = CorrectedSchedule()
    raw = provider.fetch_session(2022, 1, "R")
    assert raw.session_start == pd.Timestamp("2022-03-21T15:00:00Z")
    assert raw.session_end >= pd.Timestamp("2022-03-21T17:00:00Z")
    assert "FastF1 UTC" in raw.source_labels["schedule"]
    assert any(record.get("source") == "FastF1 UTC event schedule" for record in raw.source_files)


def test_bulk_laps_retain_source_stint_numbers():
    payload = _laps()
    payload["stint"] = [1, 2, 1, 3]
    laps = TracingInsightsProvider._laps(payload)
    assert laps["Stint"].tolist() == [1, 2, 1, 3]


def test_source_stint_restoration_joins_exact_driver_lap_identity_without_reordering():
    from f1forecast.bulk_provider import restore_source_stints

    payload = _laps()
    payload["stint"] = [1, 2, 1, 3]
    archived = TracingInsightsProvider._laps(payload).drop(columns=["Stint"], errors="ignore")
    archived = archived.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    observed = restore_source_stints(archived, payload)
    assert observed is not None
    assert observed["Stint"].tolist() == [3, 1, 1, 2]
    pd.testing.assert_frame_equal(observed.drop(columns="Stint"), archived)
    assert "Stint" not in archived


def test_source_stint_restoration_refuses_missing_identities():
    from f1forecast.bulk_provider import restore_source_stints

    payload = _laps()
    archived = TracingInsightsProvider._laps(payload).iloc[:-1]
    with pytest.raises(ValueError, match="identity mismatch"):
        restore_source_stints(archived, payload)


def test_source_stint_restoration_reports_absent_source_field():
    from f1forecast.bulk_provider import restore_source_stints

    payload = _laps()
    archived = TracingInsightsProvider._laps(payload)
    del payload["stint"]
    assert restore_source_stints(archived, payload) is None
