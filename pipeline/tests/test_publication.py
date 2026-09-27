import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from f1forecast.contracts import Forecast
from f1forecast.publication import publish_forecast, verify_archive
from test_contracts import forecast_payload


def test_history_check_rejects_rewriting_both_forecast_and_ledger(tmp_path, monkeypatch):
    from f1forecast.publication import verify_archive_history

    history = {}

    def read_only_history(args, **_kwargs):
        if "rev-parse" in args:
            output = "false\n"
        elif "log" in args:
            output = "\n".join(reversed(history))
        elif "show" in args:
            output = history[args[-1].split(":")[0]]
        else:
            pytest.fail("Only read-only Git history queries are allowed")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("f1forecast.publication.subprocess.run", read_only_history)
    root = tmp_path / "site-data" / "forecasts"
    forecast = Forecast.model_validate({**forecast_payload(), "kind": "issued"})
    path = publish_forecast(forecast, root, now=datetime(2026, 9, 1, 10, tzinfo=UTC))
    history["original"] = (root / "hashes.json").read_text()
    assert verify_archive_history(root, repo_root=tmp_path) == 1
    payload = json.loads(path.read_text())
    payload["seed"] = 99
    path.write_text(json.dumps(payload))
    ledger = root / "hashes.json"
    ledger.write_text(
        json.dumps(
            {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    )
    history["rewritten"] = ledger.read_text()
    assert verify_archive(root) == 1  # Internally consistent but historically dishonest.
    with pytest.raises(ValueError, match="changed or removed"):
        verify_archive_history(root, repo_root=tmp_path)


def test_issued_forecast_is_immutable_and_archive_detects_tampering(tmp_path):
    payload = forecast_payload()
    payload["kind"] = "issued"
    forecast = Forecast.model_validate(payload)
    now = datetime(2026, 9, 1, 10, tzinfo=UTC)
    path = publish_forecast(forecast, tmp_path, now=now)
    assert path == publish_forecast(forecast, tmp_path, now=now)
    assert verify_archive(tmp_path) == 1
    edited = forecast.model_copy(update={"seed": 99})
    with pytest.raises(ValueError, match="immutable"):
        publish_forecast(edited, tmp_path, now=now)
    path.write_text(path.read_text().replace('"seed": 42', '"seed": 99'))
    with pytest.raises(ValueError, match="hash"):
        verify_archive(tmp_path)


def test_missed_cutoff_cannot_be_published_as_issued(tmp_path):
    forecast = Forecast.model_validate({**forecast_payload(), "kind": "issued"})
    with pytest.raises(ValueError, match="cutoff"):
        publish_forecast(forecast, tmp_path, now=datetime(2026, 9, 1, 11, tzinfo=UTC))
    assert not list(tmp_path.rglob("*.json"))


def test_reconstruction_and_demo_are_separate_from_issued_archive(tmp_path):
    forecast = Forecast.model_validate(forecast_payload())
    result = publish_forecast(forecast, tmp_path)
    assert result.parent.name == "demonstration"
    assert verify_archive(tmp_path) == 1


def test_missing_or_added_archive_entries_fail_verification(tmp_path):
    forecast = Forecast.model_validate(forecast_payload())
    path = publish_forecast(forecast, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_archive(tmp_path)
