"""Team evidence must be eligible, identity-matched and preserve source observations."""

from copy import deepcopy
from hashlib import sha256

import pandas as pd
import pytest
from f1forecast.team_supplements import apply_team_supplement


def example(tmp_path):
    evidence = tmp_path / "entry-list.pdf"
    evidence.write_bytes(b"%PDF-1.4 synthetic source fixture")
    manifest = {
        "snapshot_id": "2026-07-FP1_original", "session_id": "2026-07-FP1",
        "event_id": "2026-07", "season": 2026, "circuit_id": "catalunya",
        "session_type": "FP1", "session_start": "2026-06-12T11:30:00Z",
        "retrieved_at": "2026-09-26T12:00:00Z", "provenance": "retrospective",
        "source": "Pinned timing source", "source_files": [{"url": "original"}],
        "sources": {"laps": "Pinned timing source"},
        "coverage": {"session_complete": True, "complete": False,
                     "missing_telemetry_drivers": [], "missing_results": False},
    }
    summary = pd.DataFrame({
        "session_id": ["2026-07-FP1"] * 2, "event_id": ["2026-07"] * 2,
        "season": [2026] * 2, "circuit_id": ["catalunya"] * 2,
        "driver_code": ["ARO", "BRO"], "driver_id": [None, "browning"],
        "team_id": ["None", None],
        "available_at": pd.to_datetime(["2026-09-26T12:00:00Z"] * 2),
    })
    tables = {
        "laps": pd.DataFrame({"Driver": ["ARO"], "DriverNumber": ["97"], "Team": ["None"]}),
        "results": pd.DataFrame({"Abbreviation": ["ARO", "BRO"],
                                 "DriverNumber": ["97", "46"], "TeamName": [None, None]}),
    }
    supplement = {
        "session_id": "2026-07-FP1", "event_id": "2026-07", "season": 2026,
        "circuit_id": "catalunya", "source_event_name": "Barcelona-Catalunya Grand Prix",
        "source": {"url": "https://www.fia.com/entry-list.pdf",
                   "sha256": sha256(evidence.read_bytes()).hexdigest(),
                   "published_at": "2026-06-12T08:56:00Z",
                   "retrieved_at": "2026-10-04T18:15:00Z"},
        "drivers": {"ARO": {"number": "97", "team_id": "audi"},
                    "BRO": {"number": "46", "team_id": "williams"}},
    }
    return manifest, summary, tables, supplement, evidence


def test_entry_list_fills_only_team_identity_and_retains_parent_source(tmp_path):
    manifest, summary, tables, supplement, evidence = example(tmp_path)
    original_manifest, original_summary = deepcopy(manifest), summary.copy(deep=True)
    original_tables = {name: table.copy(deep=True) for name, table in tables.items()}
    output = apply_team_supplement(manifest, summary, tables, supplement, evidence)
    assert output.summary["team_id"].tolist() == ["audi", "williams"]
    assert output.summary["driver_id"].tolist() == [None, "browning"]
    assert output.summary["available_at"].eq(pd.Timestamp("2026-10-04T18:15:00Z")).all()
    assert output.manifest["parent_snapshot_id"] == "2026-07-FP1_original"
    assert "snapshot_id" not in output.manifest
    assert output.manifest["source_files"][0] == {"url": "original"}
    assert output.manifest["source_files"][-1]["sha256"] == supplement["source"]["sha256"]
    assert output.manifest["team_identity_supplement"]["filled_driver_codes"] == ["ARO", "BRO"]
    assert output.manifest["coverage"]["missing_team_drivers"] == []
    assert output.manifest["coverage"]["complete"]
    assert manifest == original_manifest
    pd.testing.assert_frame_equal(summary, original_summary)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original_tables[name])
        pd.testing.assert_frame_equal(output.tables[name], original_tables[name])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entry-list.pdf"]


@pytest.mark.parametrize("case", ["event", "season", "session", "circuit", "code", "number",
                                  "conflict", "late", "retrieved_before_publication", "hash"])
def test_ineligible_or_conflicting_supplement_changes_nothing(tmp_path, case):
    manifest, summary, tables, supplement, evidence = example(tmp_path)
    if case in {"event", "season", "session", "circuit"}:
        key = {"event": "event_id", "season": "season", "session": "session_id",
               "circuit": "circuit_id"}[case]
        supplement[key] = "wrong"
    elif case == "code":
        supplement["drivers"]["XXX"] = supplement["drivers"].pop("ARO")
    elif case == "number":
        supplement["drivers"]["ARO"]["number"] = "46"
    elif case == "conflict":
        summary.loc[0, "team_id"] = "ferrari"
    elif case == "late":
        supplement["source"]["published_at"] = "2026-06-12T11:31:00Z"
    elif case == "retrieved_before_publication":
        supplement["source"]["retrieved_at"] = "2026-06-12T08:55:00Z"
    else:
        evidence.write_bytes(b"%PDF-modified")
    original_manifest, original_summary = deepcopy(manifest), summary.copy(deep=True)
    original_tables = {name: table.copy(deep=True) for name, table in tables.items()}
    with pytest.raises(ValueError):
        apply_team_supplement(manifest, summary, tables, supplement, evidence)
    assert manifest == original_manifest
    pd.testing.assert_frame_equal(summary, original_summary)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original_tables[name])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entry-list.pdf"]


def test_valid_existing_team_and_missing_channel_flags_are_preserved(tmp_path):
    manifest, summary, tables, supplement, evidence = example(tmp_path)
    summary.loc[0, "team_id"] = "audi"
    manifest["coverage"]["missing_telemetry_drivers"] = ["BRO"]
    output = apply_team_supplement(manifest, summary, tables, supplement, evidence)
    assert output.summary["team_id"].tolist() == ["audi", "williams"]
    assert output.manifest["team_identity_supplement"]["filled_driver_codes"] == ["BRO"]
    assert not output.manifest["coverage"]["complete"]
