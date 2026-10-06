"""A learned release falls back to its paired baseline only for cases the baseline validated.

Synthetic fixtures prove the selection policy, not F1 accuracy.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from f1forecast.contracts import Event, SiteData
from f1forecast.features import NUMERIC_FEATURES, build_feature_rows, missing_feature_groups
from f1forecast.modeling import fit_ranker
from f1forecast.operations import UnvalidatedCase, forecast_for_case, run_tick, target_rows
from f1forecast.publication import publish_site
from f1forecast.registry import ModelRecord, ModelRegistry, fallback_model
from test_release import small_training_set

Q_START = datetime(2026, 10, 10, 13, tzinfo=UTC)
NOW = Q_START - timedelta(minutes=40)
TRAINING_CUTOFF = datetime(2026, 9, 1, tzinfo=UTC)
APPROVED = datetime(2026, 10, 1, tzinfo=UTC)
LEARNED_CASE = ["missing-long-run-pace"]
BASELINE_ONLY_CASE = ["missing-telemetry"]


@pytest.fixture(scope="module")
def model_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("model") / "model.cbm"
    fit_ranker(small_training_set(), iterations=5).save_model(str(path))
    return path.read_bytes()


def _driver(identifier: str) -> dict:
    return {
        "id": identifier,
        "code": identifier.upper() * 3,
        "name": identifier.upper(),
        "team": "Test Team",
        "color": "#abcdef",
    }


def fp1_only_event() -> Event:
    return Event.model_validate(
        {
            "id": "2026-01",
            "season": 2026,
            "round": 1,
            "name": "Synthetic FP1-only weekend",
            "circuit": "test_circuit",
            "country": "Testland",
            "timezone": "UTC",
            "sessions": [
                {
                    "id": "2026-01-FP1",
                    "kind": "FP1",
                    "start": Q_START - timedelta(days=1),
                    "end": Q_START - timedelta(hours=23),
                },
                {"id": "2026-01-Q", "kind": "Q", "start": Q_START, "end": Q_START + timedelta(hours=1)},
            ],
            "entrants": [_driver("a"), _driver("b")],
            "targets": [
                {
                    "target": "qualifying",
                    "session_id": "2026-01-Q",
                    "cutoff_at": Q_START - timedelta(minutes=30),
                    "state": "scheduled",
                    "entrant_ids": ["a", "b"],
                    "roster_source": "Operator verified entry list",
                    "roster_observed_at": Q_START - timedelta(hours=2),
                }
            ],
        }
    )


def feature_rows(*missing: str) -> pd.DataFrame:
    rows = pd.DataFrame({name: [1.0, 2.0] for name in NUMERIC_FEATURES})
    rows["driver_id"] = ["a", "b"]
    rows["baseline_score"] = [1.0, 0.0]
    rows["used_session_ids"] = [("2026-01-FP1",), ("2026-01-FP1",)]
    for name in missing:
        rows.loc[0, name] = np.nan
    return rows


def release(
    root: Path,
    version: str,
    kind: str,
    patterns: list[list[str]],
    model: bytes,
    *,
    target: str = "qualifying",
    history: bytes = b"synthetic history",
    report: bytes = b"synthetic evaluation",
    approved_at: datetime = APPROVED,
) -> ModelRecord:
    folder = root / "approved" / version
    folder.mkdir(parents=True)
    paths, hashes = {}, {}
    for key, name, content in (
        ("model_path", "model.cbm", model),
        ("metadata_path", "metadata.json", b"{}"),
        ("history_path", "history.parquet", history),
        ("report_path", "evaluation.json", report),
    ):
        (folder / name).write_bytes(content)
        relative = f"approved/{version}/{name}"
        paths[key] = relative
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return ModelRecord.model_validate(
        {
            "version": version,
            "target": target,
            "kind": kind,
            "approved": True,
            "approved_at": approved_at,
            "training_cutoff": TRAINING_CUTOFF,
            "hashes": hashes,
            "temperature": 1.0,
            "baseline_name": "blended",
            "validation_events": 1,
            "evaluation": {
                "probability_loss": 0.9,
                "position_mae": 1.0,
                "win_brier": 0.1,
                "podium_brier": 0.1,
                "top10_brier": 0.1,
            },
            "validated_missing_patterns": patterns,
            **paths,
        }
    )


def paired_releases(root: Path, model: bytes, **history) -> tuple[ModelRecord, ModelRecord]:
    """Baseline approved first, then the learned model from the same evaluation."""
    baseline = release(
        root, "q-model-baseline", "baseline", [LEARNED_CASE, BASELINE_ONLY_CASE], model, **history
    )
    learned = release(
        root,
        "q-model",
        "catboost",
        [LEARNED_CASE],
        model,
        approved_at=APPROVED + timedelta(minutes=1),
        **history,
    )
    return learned, baseline


def _forecast(root: Path, rows: pd.DataFrame, registry: ModelRegistry, record: ModelRecord):
    event = fp1_only_event()
    return forecast_for_case(event, event.targets[0], rows, registry, record, root, now=NOW)


@pytest.mark.parametrize("missing", [(), ("long_run_pace_gap_s",)])
def test_learned_release_serves_complete_inputs_and_its_own_validated_case(
    tmp_path, model_bytes, missing
):
    learned, baseline = paired_releases(tmp_path, model_bytes)
    registry = ModelRegistry(models=[baseline, learned])
    forecast = _forecast(tmp_path, feature_rows(*missing), registry, learned)
    assert (forecast.model_kind, forecast.model_version) == ("catboost", "q-model")
    assert "Baseline used" not in forecast.coverage.summary


def test_case_only_the_paired_baseline_validated_is_published_as_labelled_baseline(
    tmp_path, model_bytes
):
    learned, baseline = paired_releases(tmp_path, model_bytes)
    registry = ModelRegistry(models=[baseline, learned])
    forecast = _forecast(tmp_path, feature_rows("mean_speed"), registry, learned)
    assert (forecast.model_kind, forecast.model_version) == ("baseline", "q-model-baseline")
    assert forecast.coverage.flags == BASELINE_ONLY_CASE
    assert forecast.coverage.summary.endswith(
        " Baseline used: CatBoost release q-model has not been validated for this missing-data"
        " case."
    )
    assert forecast.input_snapshot_ids[0] == "history-sha256:" + learned.hashes[learned.history_path]
    assert np.isclose(sum(d.win_probability for d in forecast.drivers), 1)


def test_case_neither_release_validated_remains_unavailable(tmp_path, model_bytes):
    learned, baseline = paired_releases(tmp_path, model_bytes)
    registry = ModelRegistry(models=[baseline, learned])
    with pytest.raises(UnvalidatedCase, match="missing-observed-conditions"):
        _forecast(tmp_path, feature_rows("air_temp"), registry, learned)


def test_missing_required_inputs_are_never_rescued_by_the_baseline(tmp_path, model_bytes):
    learned, baseline = paired_releases(tmp_path, model_bytes)
    registry = ModelRegistry(models=[baseline, learned])
    rows = feature_rows()
    rows["practice_pace_gap_s"] = np.nan
    with pytest.raises(ValueError, match="required practice pace") as failure:
        _forecast(tmp_path, rows, registry, learned)
    assert not isinstance(failure.value, UnvalidatedCase)


def test_baseline_release_has_no_further_fallback(tmp_path, model_bytes):
    learned, baseline = paired_releases(tmp_path, model_bytes)
    registry = ModelRegistry(models=[baseline, learned])
    assert fallback_model(registry, learned, NOW) == baseline
    assert fallback_model(registry, baseline, NOW) is None
    with pytest.raises(UnvalidatedCase):
        _forecast(tmp_path, feature_rows("air_temp"), registry, baseline)


@pytest.mark.parametrize(
    "change",
    [
        {"history": b"different history"},
        {"report": b"different evaluation"},
        {"target": "race"},
        {"approved_at": NOW + timedelta(minutes=1)},
    ],
    ids=["other-history", "other-evaluation", "other-target", "approved-after-clock"],
)
def test_fallback_requires_an_eligible_baseline_from_the_same_evaluation_and_history(
    tmp_path, model_bytes, change
):
    learned = release(tmp_path, "q-model", "catboost", [LEARNED_CASE], model_bytes)
    other = release(
        tmp_path, "q-model-baseline", "baseline", [BASELINE_ONLY_CASE], model_bytes, **change
    )
    registry = ModelRegistry(models=[other, learned])
    assert fallback_model(registry, learned, NOW) is None
    with pytest.raises(UnvalidatedCase):
        _forecast(tmp_path, feature_rows("mean_speed"), registry, learned)


def fp1_summaries() -> pd.DataFrame:
    end = Q_START - timedelta(hours=23)
    return pd.DataFrame(
        [
            {
                "event_id": "2026-01",
                "season": 2026,
                "circuit_id": "test_circuit",
                "driver_id": driver,
                "team_id": "Test Team",
                "session_id": "2026-01-FP1",
                "session_type": "FP1",
                "session_end": end,
                "observed_at": end,
                "available_at": end + timedelta(minutes=30),
                "provenance": "prospective",
                "pace_s": pace,
                "usable_laps": 20.0,
            }
            for driver, pace in (("a", 90.0), ("b", 90.5))
        ]
    )


def weekend(
    tmp_path: Path, model: bytes, summaries: pd.DataFrame, baseline_case: list[str], monkeypatch
) -> tuple[Path, Path]:
    """A site and a paired registry whose history holds this weekend's FP1 summaries."""
    history = tmp_path / "history.parquet"
    summaries.to_parquet(history, index=False)
    models = tmp_path / "models"
    baseline = release(
        models, "q-model-baseline", "baseline", [baseline_case], model,
        history=history.read_bytes(),
    )
    learned = release(
        models, "q-model", "catboost", [LEARNED_CASE], model,
        history=history.read_bytes(), approved_at=APPROVED + timedelta(minutes=1),
    )
    registry_path = models / "registry.json"
    registry_path.write_text(
        ModelRegistry(models=[baseline, learned]).model_dump_json(), encoding="utf-8"
    )
    site_dir = tmp_path / "site-data"
    publish_site(
        SiteData.model_validate(
            {
                "generated_at": NOW - timedelta(days=1),
                "demo_notice": None,
                "events": [fp1_only_event().model_dump()],
                "forecasts": [],
                "analyses": [],
                "evaluation": {
                    "status": "pending",
                    "summary": "Awaiting evaluation.",
                    "seasons": [],
                    "comparison": [],
                    "limitations": [],
                },
            }
        ),
        site_dir,
    )
    monkeypatch.setattr(
        "f1forecast.providers.JolpicaProvider",
        lambda *_args, **_kwargs: pytest.fail("network provider was called"),
    )
    return site_dir, registry_path


def test_weekend_tick_issues_the_paired_baseline_for_an_fp1_only_case(
    tmp_path, model_bytes, monkeypatch
):
    rows = build_feature_rows(fp1_summaries(), target_rows(fp1_only_event(), "qualifying"))
    observed_case = missing_feature_groups(rows, "qualifying")
    assert observed_case and "missing-long-run-pace" in observed_case
    site_dir, registry_path = weekend(
        tmp_path, model_bytes, fp1_summaries(), observed_case, monkeypatch
    )

    report = run_tick(site_dir, tmp_path / "archive", registry_path, now=NOW, ingest=False)
    assert report["errors"] == []
    assert report["issued"] == ["2026-01-qualifying-issued"]
    site = SiteData.model_validate_json((site_dir / "site.json").read_text(encoding="utf-8"))
    forecast = site.forecasts[0]
    assert (forecast.model_kind, forecast.model_version) == ("baseline", "q-model-baseline")
    assert forecast.coverage.flags == observed_case
    assert site.events[0].targets[0].state == "issued"

    saved = next((site_dir / "forecasts").rglob("*.json")).read_bytes()
    again = run_tick(
        site_dir, tmp_path / "archive", registry_path, now=NOW + timedelta(minutes=5), ingest=False
    )
    assert again["issued"] == again["errors"] == []
    assert next((site_dir / "forecasts").rglob("*.json")).read_bytes() == saved


def test_practice_finalized_after_cutoff_is_never_rescued_and_ends_unavailable(
    tmp_path, model_bytes, monkeypatch
):
    delayed = fp1_summaries()
    delayed["available_at"] = Q_START - timedelta(minutes=10)
    site_dir, registry_path = weekend(
        tmp_path, model_bytes, delayed, ["missing-long-run-pace"], monkeypatch
    )

    before = run_tick(site_dir, tmp_path / "archive", registry_path, now=NOW, ingest=False)
    assert before["issued"] == []
    assert [e["reason"] for e in before["errors"]] == ["required practice pace is unavailable"]
    target = SiteData.model_validate_json((site_dir / "site.json").read_text()).events[0].targets[0]
    assert target.state == "scheduled"

    after = run_tick(
        site_dir, tmp_path / "archive", registry_path, now=Q_START - timedelta(minutes=5),
        ingest=False,
    )
    assert after["unavailable"] == ["2026-01/qualifying"]
    site = SiteData.model_validate_json((site_dir / "site.json").read_text())
    assert site.events[0].targets[0].state == "unavailable"
    assert site.forecasts == []
    assert not list((site_dir / "forecasts").rglob("*.json"))
