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
    assert len(raw.source_files) == 8  # Archive JSON, traces, calendar, identities, DriverList.
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
    summary, _, _ = normalize_session(raw, datetime(2026, 9, 26, tzinfo=UTC))
    assert summary.set_index("driver_code").loc["LEC", "qualifying_gap"] == 0.0
    assert summary.set_index("driver_code").loc["VER", "qualifying_gap"] == 0.5


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

    class ThreeEntrants:
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
