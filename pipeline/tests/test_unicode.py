"""Published names must survive Windows hosts whose default encoding is not UTF-8."""

from datetime import UTC, datetime
from pathlib import Path

from f1forecast.calendar import set_entrants
from f1forecast.contracts import SiteData
from f1forecast.publication import publish_site
from test_weekend_rehearsal import _driver, _site


def test_entry_update_preserves_provider_unicode_on_windows(tmp_path, monkeypatch):
    site = _site()
    site.events[0].name = "São Paulo Grand Prix"
    site.events[0].entrants[0].name = "Nico Hülkenberg"
    publish_site(site, tmp_path)
    original_read = Path.read_text
    def legacy_default_read(path, encoding=None, errors=None, **kwargs):
        return original_read(path, encoding=encoding or "cp1252", errors=errors, **kwargs)
    monkeypatch.setattr(Path, "read_text", legacy_default_read)
    set_entrants(tmp_path, "2026-01", [_driver("b"), _driver("c")], target="race",
                 observed_at=datetime(2026, 9, 1, tzinfo=UTC))
    updated = SiteData.model_validate_json((tmp_path / "site.json").read_text(encoding="utf-8"))
    assert updated.events[0].name == "São Paulo Grand Prix"
    assert updated.events[0].entrants[0].name == "Nico Hülkenberg"
