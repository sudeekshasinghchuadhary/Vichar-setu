"""Clarification intelligence: uncertainty -> deterministic questions.

Derives a ClarificationRequest from an IntelligenceResult's existing
structured gaps: EligibilityResult.missing_information (only from
needs_information verdicts — failed schemes need no follow-up) and
unknown need amounts/periods. Pure function, no LLM, no storage, no
decisions. Never asks for anything the current result does not require;
never converts unknowns into assumptions.
"""

from typing import Literal

from intelligence_engine.schemas import (
    ClarificationQuestion,
    ClarificationRequest,
    IntelligenceResult,
    SchemeSupport,
    SupportNeed,
    ValueConflict,
)

FinancialUnknownCause = Literal[
    "need_amount_unknown",
    "need_period_unknown",
    "support_amount_unknown",
    "no_period_matched_support",
    "period_mismatch",
]


def classify_financial_unknown(
    need: SupportNeed, supports: list[SchemeSupport]
) -> FinancialUnknownCause | None:
    """Classify why financial analysis cannot run for one need, if at all.

    Pure structural check over already-available data, using the same
    candidate-support contract as FinancialIntelligence.coverage_for_need
    (supports already filtered to the need; eligibility filtering is the
    caller's concern). Returns None when analysis can run. Never
    computes, never asks, never guesses, never converts unknown to zero.
    """
    if need.amount is None:
        return "need_amount_unknown"
    if need.amount_period == "unknown":
        return "need_period_unknown"
    relevant = [s for s in supports if need.need_type in s.covers_need_types]
    matched = [s for s in relevant if s.max_amount_period == need.amount_period]
    if not matched:
        if any(s.max_amount_period != "unknown" for s in relevant):
            return "period_mismatch"
        return "no_period_matched_support"
    if all(s.max_amount is None for s in matched):
        return "support_amount_unknown"
    return None


def _eligible_supports_by_need(
    result: IntelligenceResult,
) -> dict[tuple[str, str], list[SchemeSupport]]:
    """Rebuild coverage_for_need-style candidates from the support plan.

    Uses SupportMapping evidence already in the result (eligible schemes
    only, mirroring the orchestrator's coverage input). Fields copy 1:1;
    nothing is invented. A missing plan falls back to no candidates,
    which classifies as scheme-side unknown and therefore asks nothing.
    """
    grouped: dict[tuple[str, str], list[SchemeSupport]] = {}
    plan = result.support_plan
    if plan is None:
        return grouped
    for need_plan in plan.need_plans:
        key = (need_plan.need.need_type, need_plan.need.amount_period)
        supports = [
            SchemeSupport(
                support_type=mapping.support_type,
                covers_need_types=[need_plan.need.need_type],
                max_amount=mapping.max_amount,
                max_amount_period=mapping.max_amount_period,
            )
            for mapping in need_plan.mappings
            if mapping.eligibility_status == "eligible"
        ]
        grouped.setdefault(key, []).extend(supports)
    return grouped

_FIELD_QUESTIONS: dict[str, str] = {
    "age": "How old are you?",
    "gender": "What is your gender?",
    "social_category": "Which social category do you belong to (SC/ST/OBC/General/EWS)?",
    "state": "Which state do you live in?",
    "district": "Which district do you live in?",
    "occupation": "What is your occupation?",
    "annual_family_income": "What is your annual family income?",
    "purpose": "What do you need support for?",
    "project_type": "What type of project is this?",
    "estimated_project_cost": "What is the estimated cost of your project?",
    "education_level": "What is your education level?",
}


def _field_question(field: str) -> str:
    """Fixed wording per known field; humanized fallback for the rest."""
    if field in _FIELD_QUESTIONS:
        return _FIELD_QUESTIONS[field]
    return f"Please provide your {field.replace('_', ' ')}."


def build_clarification(result: IntelligenceResult) -> ClarificationRequest | None:
    """Build questions for genuinely missing inputs, or None when complete.

    Args:
        result: Product result whose uncertainty is already structured.

    Returns:
        ClarificationRequest with deduplicated fields in first-seen
        order, or None when nothing is missing.
    """
    missing: list[str] = []
    questions: list[ClarificationQuestion] = []

    def _add(field: str, question: str) -> None:
        """Append one grounded question unless already asked."""
        if field not in missing:
            missing.append(field)
            questions.append(ClarificationQuestion(field=field, question=question))

    for insight in result.schemes:
        eligibility = insight.eligibility
        if eligibility is not None and eligibility.status == "needs_information":
            for field in eligibility.missing_information:
                _add(field, _field_question(field))

    if result.needs is not None:
        supports_by_need = _eligible_supports_by_need(result)
        for index, need in enumerate(result.needs.needs):
            cause = classify_financial_unknown(
                need, supports_by_need.get((need.need_type, need.amount_period), [])
            )
            if cause == "need_amount_unknown":
                _add(
                    f"needs[{index}].amount",
                    f"Approximately how much funding/support is required for '{need.need_type}'?",
                )
            elif cause == "need_period_unknown":
                _add(
                    f"needs[{index}].amount_period",
                    f"Is the {need.amount:g} for '{need.need_type}' needed "
                    "one-time, monthly, or annually?",
                )
            # Scheme-side causes (support_amount_unknown,
            # no_period_matched_support, period_mismatch) intentionally
            # ask nothing: the user cannot fix backend support data.
            # Routing is exactly equivalent to the previous field checks:
            # need-side unknowns ask, everything else stays silent.

    if not missing:
        return None
    return ClarificationRequest(
        missing_fields=missing, questions=questions, current_partial_result=result
    )


_MONEY_FIELDS = ("annual_family_income", "estimated_project_cost")


def _format_conflict_value(field: str, value: object) -> str:
    """Deterministic display: money with rupee sign, wholes without decimals."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        text = str(int(value)) if float(value).is_integer() else str(value)
        return f"₹{text}" if field in _MONEY_FIELDS else text
    return str(value)


def _conflict_label(field: str) -> str:
    """Human label for profile fields and disputed need amounts."""
    if field.startswith("needs:"):
        parts = field.split(":", 2)
        if len(parts) == 3:
            return f"'{parts[1]}' support amount ({parts[2]})"
    return field.replace("_", " ")


def build_conflict_request(
    conflicts: list[ValueConflict], partial: IntelligenceResult
) -> ClarificationRequest:
    """Build explicit conflict questions preserving both disputed values.

    Each question names the field with the form value and the extracted
    value and asks which is correct. No winner is picked here; the
    backend resolves via resolve_conflicts() after the user confirms.
    """
    questions = [
        ClarificationQuestion(
            field=conflict.field,
            question=(
                f"You provided two different values for {_conflict_label(conflict.field)}: "
                f"{_format_conflict_value(conflict.field, conflict.form_value)} (form) and "
                f"{_format_conflict_value(conflict.field, conflict.extracted_value)} (text). "
                "Which one is correct?"
            ),
        )
        for conflict in conflicts
    ]
    return ClarificationRequest(
        missing_fields=[],
        questions=questions,
        conflicts=list(conflicts),
        current_partial_result=partial,
    )
