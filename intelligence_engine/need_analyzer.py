"""Need analysis: natural language -> structured goals and support needs.

Flow: user text -> NeedExtractor.extract_need_data() -> Pydantic
SupportNeed/NeedAnalysisResult validation -> deterministic
normalization (need-type aliases, duplicate merging, total).

The LLM/extractor only understands wording. It must NOT invent
amounts, guess periods, decide eligibility, or recommend schemes.
Arithmetic (totals, merging) is deterministic code. Missing
information stays unknown. This module never scores or ranks schemes.
"""

from typing import Any, Optional

from pydantic import ValidationError

from intelligence_engine.llm_client import NeedExtractor
from intelligence_engine.need_vocabulary import normalize_need_type
from intelligence_engine.schemas import NeedAnalysisResult, SupportNeed

__all__ = ["NeedAnalyzer", "NeedAnalysisError", "normalize_need_type"]


class NeedAnalysisError(ValueError):
    """Raised for malformed extractor output or validation failures."""


_PRIORITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def _clean_str(value: Any) -> Optional[str]:
    """Strip whitespace; return None for empty/non-string values."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned if cleaned else None


def normalize_need_data(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize raw need dicts; collect transparency notes (no inference)."""
    raw_needs = data.get("needs", [])
    if not isinstance(raw_needs, list):
        raise NeedAnalysisError("Extractor 'needs' must be a list.")
    normalized: list[dict[str, Any]] = []
    notes: list[str] = []
    for entry in raw_needs:
        if not isinstance(entry, dict):
            raise NeedAnalysisError("Each extractor need must be a dict.")
        item = dict(entry)
        need_type, recognized = normalize_need_type(item.get("need_type"))
        if need_type is None:
            raise NeedAnalysisError("Each need requires a non-empty need_type.")
        item["need_type"] = need_type
        if not recognized:
            notes.append(f"Unrecognized need type kept as stated: '{need_type}'.")
        if "context" in item:
            item["context"] = _clean_str(item["context"])
        if item.get("amount_period") is None:
            item["amount_period"] = "unknown"
        normalized.append(item)
    return normalized, notes


def merge_duplicate_needs(needs: list[SupportNeed]) -> list[SupportNeed]:
    """Merge same (need_type, amount_period) needs deterministically.

    Amounts: both known -> summed; one known -> kept; none -> None.
    Contexts joined with "; ". Priority keeps the highest explicit one.
    Order follows first occurrence.
    """
    merged: dict[tuple[str, str], SupportNeed] = {}
    for need in needs:
        key = (need.need_type, need.amount_period)
        existing = merged.get(key)
        if existing is None:
            merged[key] = need
            continue
        if existing.amount is not None and need.amount is not None:
            amount: float | None = existing.amount + need.amount
        else:
            amount = existing.amount if existing.amount is not None else need.amount
        contexts = [part for part in (existing.context, need.context) if part]
        priorities = [part for part in (existing.priority, need.priority) if part]
        best = max(priorities, key=lambda item: _PRIORITY_RANK[item]) if priorities else None
        merged[key] = SupportNeed(
            need_type=need.need_type,
            amount=amount,
            amount_period=need.amount_period,
            context="; ".join(contexts) if contexts else None,
            priority=best,
        )
    return list(merged.values())


def total_one_time(needs: list[SupportNeed]) -> float | None:
    """Sum one-time amounts only; None when none are known (periods never mixed)."""
    known = [need.amount for need in needs if need.amount_period == "one_time" and need.amount is not None]
    if not known:
        return None
    return float(sum(known))


class NeedAnalyzer:
    """Convert natural-language goals/needs into validated structures."""

    def __init__(self, extractor: NeedExtractor) -> None:
        """Store the injected need-extraction backend."""
        self.extractor = extractor

    def analyze(self, user_text: str) -> NeedAnalysisResult:
        """Extract, validate, normalize, and total the user's needs.

        Args:
            user_text: Natural-language user description.

        Returns:
            Validated NeedAnalysisResult with missing data left unknown.

        Raises:
            NeedAnalysisError: On malformed output or validation failure.
        """
        extracted: Any = self.extractor.extract_need_data(user_text)
        if not isinstance(extracted, dict):
            raise NeedAnalysisError(
                f"Extractor output must be a dict, got {type(extracted).__name__}."
            )
        normalized, notes = normalize_need_data(extracted)
        try:
            needs = [SupportNeed.model_validate(item) for item in normalized]
        except ValidationError as exc:
            raise NeedAnalysisError(f"Invalid need data: {exc}") from exc
        merged = merge_duplicate_needs(needs)
        if not isinstance(extracted.get("business_goal", None), (str, type(None))):
            raise NeedAnalysisError("Extractor 'business_goal' must be a string when provided.")
        return NeedAnalysisResult(
            business_goal=_clean_str(extracted.get("business_goal")),
            needs=merged,
            total_requested=total_one_time(merged),
            notes=notes,
        )
