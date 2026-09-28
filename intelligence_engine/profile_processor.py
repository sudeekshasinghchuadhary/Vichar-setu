"""Profile processing: LLM output -> validation -> normalization.

Flow: natural-language text -> LLMClient.extract_profile_data()
-> Pydantic UserProfile validation -> deterministic normalization.
Missing information stays None. No facts are inferred.
This module never decides eligibility or scores schemes.
"""

import re
from typing import Any, Optional

from pydantic import ValidationError

from intelligence_engine.llm_client import LLMClient
from intelligence_engine.profile_vocabulary import (
    normalize_social_category,
    normalize_state,
)
from intelligence_engine.schemas import UserProfile


class ProfileProcessingError(ValueError):
    """Raised when LLM output is malformed or fails UserProfile validation."""


def _clean_str(value: Any) -> Optional[str]:
    """Strip whitespace; return None for empty/non-string values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned if cleaned else None


_AMOUNT_PATTERN = re.compile(
    r"^\s*(?:\u20b9|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?\s*$",
    re.IGNORECASE,
)


def _parse_amount(value: Any) -> Any:
    """Parse conservative currency strings to numbers (no inference).

    Handles optional ₹/Rs/INR symbols, Indian digit grouping, decimals,
    and lakh/crore units. Anything else passes through untouched for
    Pydantic to accept or reject; never guesses.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    match = _AMOUNT_PATTERN.match(value)
    if match is None:
        return value
    number = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "").lower()
    if unit.startswith("lac") or unit.startswith("lakh"):
        return number * 100000
    if unit.startswith("crore"):
        return number * 10000000
    return number


def normalize_profile_data(data: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic normalization to extracted fields (no inference)."""
    normalized: dict[str, Any] = dict(data)
    if "state" in normalized:
        normalized["state"] = normalize_state(normalized["state"])
    if "social_category" in normalized:
        normalized["social_category"] = normalize_social_category(
            normalized["social_category"]
        )
    for key in ("annual_family_income", "estimated_project_cost"):
        if key in normalized:
            normalized[key] = _parse_amount(normalized[key])
    amount = normalized.get("annual_family_income")
    if (
        normalized.get("income_period") == "monthly"
        and isinstance(amount, (int, float))
        and not isinstance(amount, bool)
    ):
        normalized["annual_family_income"] = amount * 12
        normalized["income_period"] = "annual"
    for key in ("gender", "district", "occupation", "purpose", "project_type", "education_level"):
        if key in normalized:
            normalized[key] = _clean_str(normalized[key])
    return normalized


class ProfileProcessor:
    """Convert natural-language profiles into validated UserProfile objects."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Store the LLM client abstraction (no provider SDK here)."""
        self.llm_client = llm_client

    def process_profile(self, user_text: str) -> UserProfile:
        """Extract, validate, and normalize a UserProfile from raw text.

        Args:
            user_text: Natural-language user description.

        Returns:
            Validated, normalized UserProfile with missing fields as None.

        Raises:
            ProfileProcessingError: On malformed LLM output or validation failure.
        """
        extracted: Any = self.llm_client.extract_profile_data(user_text)
        if not isinstance(extracted, dict):
            raise ProfileProcessingError(
                f"LLM output must be a dict, got {type(extracted).__name__}."
            )
        try:
            profile = UserProfile.model_validate(normalize_profile_data(extracted))
        except ValidationError as exc:
            raise ProfileProcessingError(f"Invalid profile data: {exc}") from exc
        return profile
