"""Clarification intelligence: uncertainty -> deterministic questions.

Derives a ClarificationRequest from an IntelligenceResult's existing
structured gaps: EligibilityResult.missing_information (only from
needs_information verdicts — failed schemes need no follow-up) and
unknown need amounts/periods. Pure function, no LLM, no storage, no
decisions. Never asks for anything the current result does not require;
never converts unknowns into assumptions.
"""

from intelligence_engine.schemas import (
    ClarificationQuestion,
    ClarificationRequest,
    IntelligenceResult,
    ValueConflict,
)

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
        for index, need in enumerate(result.needs.needs):
            if need.amount is None:
                _add(
                    f"needs[{index}].amount",
                    f"Approximately how much funding/support is required for '{need.need_type}'?",
                )
            elif need.amount_period == "unknown":
                _add(
                    f"needs[{index}].amount_period",
                    f"Is the {need.amount:g} for '{need.need_type}' needed "
                    "one-time, monthly, or annually?",
                )

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
