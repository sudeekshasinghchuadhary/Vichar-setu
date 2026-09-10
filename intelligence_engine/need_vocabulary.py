"""Canonical need-type vocabulary (single source of truth).

Used by NeedAnalyzer normalization, SupportNeed/SchemeSupport
validation, and SupportPlanner matching so one spelling always means
one thing. Small on purpose: only categories with clear evidence.
Distinct financial concepts are never merged with each other.

Unrecognized wording is preserved as cleaned text (never invented),
matching the existing conservative behavior.
"""

from typing import Any, Optional

CANONICAL_NEED_TYPES: tuple[str, ...] = (
    "machinery",
    "working_capital",
    "training",
    "marketing",
    "infrastructure",
)

_NEED_TYPE_ALIASES: dict[str, str] = {
    "machine": "machinery",
    "machines": "machinery",
    "machinery": "machinery",
    "equipment": "machinery",
    "equipments": "machinery",
    "working capital": "working_capital",
    "working-capital": "working_capital",
    "workingcapital": "working_capital",
    "working_capital": "working_capital",
    "working fund": "working_capital",
    "working funds": "working_capital",
    "training": "training",
    "trainings": "training",
    "skill training": "training",
    "marketing": "marketing",
    "advertising": "marketing",
    "advertisement": "marketing",
    "promotion": "marketing",
    "infrastructure": "infrastructure",
    "infra": "infrastructure",
    "construction": "infrastructure",
    "renovation": "infrastructure",
}


def clean_token(value: Any) -> Optional[str]:
    """Collapse whitespace/casing; None for empty/non-string values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned if cleaned else None


def normalize_need_type(value: Any) -> tuple[str | None, bool]:
    """Map to canonical type, preserving unknown wording.

    Returns (normalized, recognized). Casing/whitespace variants of a
    known alias collapse to one canonical type; unrecognized non-empty
    wording is kept as cleaned text (recognized=False).
    """
    cleaned = clean_token(value)
    if cleaned is None:
        return None, False
    canonical = _NEED_TYPE_ALIASES.get(cleaned.lower())
    if canonical is not None:
        return canonical, True
    return cleaned, False


def is_canonical_need_type(value: Any) -> bool:
    """True when the value is exactly a canonical need type."""
    return isinstance(value, str) and value in CANONICAL_NEED_TYPES
