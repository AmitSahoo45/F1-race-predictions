"""Canonical identifiers without inventing missing observed affiliations."""

from __future__ import annotations


def canonical_team_id(value: object, season: int | None = None) -> str | None:
    """Normalize source names while keeping season-specific team brands distinct.

    Jolpica's RB F1 Team label and FastF1's Racing Bulls label coexist in
    2025-onward observations. Earlier RB/AlphaTauri identities remain separate.
    Non-string values and textual null placeholders supply no team evidence.
    """
    if not isinstance(value, str):
        return None
    normalized = "_".join(value.strip().lower().split())
    if normalized in {"", "none", "nan", "null", "<na>"}:
        return None
    if season is not None and season >= 2025 and normalized == "rb_f1_team":
        return "racing_bulls"
    return normalized
