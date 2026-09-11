"""Shared state/social-category vocabulary (single source of truth).

Used by ProfileProcessor normalization and UserProfile field validators
so extraction-path and direct-construction profiles normalize identically.
Small on purpose: only Uttar Pradesh variants and the five social
categories already covered in English, plus unambiguous Devanagari forms
and widely used Latin-script variants of the same referents.

Unrecognized wording passes through cleaned but unmapped (never invented).
Every canonical output re-keys to itself (lowercased), so normalization
is idempotent — safe to apply in both the processor and validators.
"""

from typing import Any, Optional

_STATE_ALIASES: dict[str, str] = {
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "u p": "Uttar Pradesh",
    "yupi": "Uttar Pradesh",
    "उत्तर प्रदेश": "Uttar Pradesh",
}

_SOCIAL_CATEGORY_ALIASES: dict[str, str] = {
    "sc": "SC",
    "scheduled caste": "SC",
    "schedule caste": "SC",
    "अनुसूचित जाति": "SC",
    "st": "ST",
    "scheduled tribe": "ST",
    "schedule tribe": "ST",
    "अनुसूचित जनजाति": "ST",
    "obc": "OBC",
    "other backward class": "OBC",
    "other backward classes": "OBC",
    "अन्य पिछड़ा वर्ग": "OBC",
    "general": "General",
    "सामान्य": "General",
    "ews": "EWS",
}


def clean_token(value: Any) -> Optional[str]:
    """Collapse whitespace; None for empty/non-string values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned if cleaned else None


def normalize_state(value: Any) -> Optional[str]:
    """Map obvious state equivalents; leave others as-is (cleaned)."""
    cleaned = clean_token(value)
    if cleaned is None:
        return None
    return _STATE_ALIASES.get(cleaned.lower(), cleaned)


def normalize_social_category(value: Any) -> Optional[str]:
    """Map obvious social-category equivalents; leave others as-is (cleaned)."""
    cleaned = clean_token(value)
    if cleaned is None:
        return None
    return _SOCIAL_CATEGORY_ALIASES.get(cleaned.lower(), cleaned)
