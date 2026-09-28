"""Clarification wording: deterministic questions, LLM phrasing only.

Flow: ClarificationRequest -> build_prompt() (fields + questions +
fixed grounding instructions, nothing else) ->
ClarificationWordingProvider -> strict field-for-field validation ->
reworded ClarificationRequest OR the original request unchanged.

The LLM decides HOW a question is worded (including Hindi/Hinglish
when asked). It never decides WHAT is asked: count, identifiers,
order, and missing-field semantics come from build_clarification()
alone. No DB, API, schemes, or eligibility logic here.
"""

from typing import Any, Optional

from intelligence_engine.llm_client import ClarificationWordingProvider
from intelligence_engine.schemas import (
    ClarificationQuestion,
    ClarificationRequest,
)

_GROUNDING_RULES = (
    "Reword each question below in natural, friendly language"
    "{language}. Respond in the user's language/style as shown in the "
    "user-text excerpt when one is supplied; do not translate unless "
    "necessary for natural wording. Preserve the meaning of each "
    "deterministic question. Rules: return a JSON array with exactly one "
    "object per input question, each as {{\"field\": <exact field>, "
    "\"question\": <reworded text>}}; keep every field identifier "
    "byte-identical, in the same order, with no additions, removals, "
    "duplicates, or renames; keep each question non-empty; do not add "
    "facts, requirements, eligibility verdicts, or guarantees; do not "
    "infer or fill in the user's missing data."
)

_BANNED_FRAGMENTS = (
    "eligible",
    "not eligible",
    "approved",
    "guaranteed",
    "guarantee",
    "rejected",
    "ineligible",
)

_MAX_EXCERPT_CHARS = 500


def _excerpt(user_text: Any) -> str:
    """Bounded whitespace-collapsed excerpt, or empty when unusable."""
    if not isinstance(user_text, str):
        return ""
    return " ".join(user_text.split())[:_MAX_EXCERPT_CHARS]


def _script_hint(text: str) -> str:
    """Fallback script signal only: Devanagari present or not.

    Never authoritative: an explicit caller hint always wins, and this
    is used solely to phrase the language instruction when no hint or
    usable excerpt exists. Mixed content stays undescribed rather than
    forced into one language.
    """
    return "Devanagari script observed" if any("\u0900" <= ch <= "\u097f" for ch in text) else ""


def build_prompt(
    request: ClarificationRequest,
    language_hint: Optional[str] = None,
    user_text: Optional[str] = None,
) -> str:
    """Render request content, optional user-text context, and rules into a prompt.

    Priority: explicit language_hint wins; otherwise the bounded user
    excerpt supplies language/style context (with a weak script signal
    only when no hint exists); neither supplied keeps prior behavior.
    """
    hint = language_hint.strip() if isinstance(language_hint, str) and language_hint.strip() else ""
    excerpt = _excerpt(user_text)
    if hint:
        language = f" in {hint}"
    elif excerpt:
        language = " matching the language/style of the user text below"
    else:
        language = ""
    lines = ["Clarification questions to reword:"]
    for item in request.questions:
        lines.append(f"- [{item.field}] {item.question}")
    if excerpt:
        lines += ["", "User text (language/style context only — asks nothing new):", excerpt]
    instructions = _GROUNDING_RULES.format(language=language)
    if not hint and excerpt and _script_hint(excerpt):
        instructions += f" Note: {_script_hint(excerpt)}."
    return "\n".join(lines) + "\n\nInstructions: " + instructions


def _check_items(items: Any, request: ClarificationRequest) -> list[str]:
    """Return failed check names (empty means the rewrite is grounded)."""
    if not isinstance(items, list) or not items:
        return ["well_formed_list"]
    expected = [item.field for item in request.questions]
    if len(items) != len(expected):
        return ["exact_count"]
    seen: set[str] = set()
    for entry in items:
        if not isinstance(entry, dict):
            return ["well_formed_items"]
        field = entry.get("field")
        question = entry.get("question")
        if not isinstance(field, str) or not isinstance(question, str):
            return ["well_formed_items"]
        if field in seen:
            return ["no_duplicates"]
        seen.add(field)
        if not question.strip():
            return ["non_empty_text"]
        lowered = question.lower()
        if any(fragment in lowered for fragment in _BANNED_FRAGMENTS):
            return ["no_verdicts_or_guarantees"]
    if [entry["field"] for entry in items] != expected:
        return ["exact_fields_and_order"]
    return []


class ClarificationGenerator:
    """Reword clarification questions through a validated LLM provider."""

    def __init__(self, provider: ClarificationWordingProvider) -> None:
        """Store the injected wording backend (no decision authority)."""
        self.provider = provider

    def generate(
        self,
        request: ClarificationRequest,
        language_hint: Optional[str] = None,
        user_text: Optional[str] = None,
    ) -> ClarificationRequest:
        """Return a reworded copy, or the original request on any violation.

        Args:
            request: Source of truth; never modified.
            language_hint: Authoritative desired wording language (e.g.
                "Hindi", "Hinglish"). Wins over user_text when both given.
                Deterministic fallback stays as authored.
            user_text: Optional raw user wording as language/style
                context only. Bounded excerpt; never new requirements.

        Returns:
            New ClarificationRequest with identical fields/order and
            reworded questions, or the original request unchanged when
            the provider fails or violates the contract.
        """
        try:
            items: Any = self.provider.rewrite_questions(
                build_prompt(request, language_hint, user_text)
            )
        except Exception:
            return request
        if _check_items(items, request):
            return request
        return ClarificationRequest(
            missing_fields=list(request.missing_fields),
            questions=[
                ClarificationQuestion(field=entry["field"], question=entry["question"].strip())
                for entry in items
            ],
            current_partial_result=request.current_partial_result,
        )
