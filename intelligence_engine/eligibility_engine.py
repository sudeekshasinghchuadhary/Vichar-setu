"""Eligibility engine: deterministic rule comparisons only.

Compares a validated UserProfile against a Scheme's structured
eligibility_rules and returns an EligibilityResult with status
"eligible" | "not_eligible" | "needs_information".

Rules evaluated (only when present on the scheme):
min_age, max_age, max_annual_income, occupations, min_education,
categories, states.

Policy:
- Rule absent -> not a restriction, skip it.
- Rule present + user value present + satisfied -> continue.
- Rule present + user value present + failed -> "not_eligible".
- Rule present + user value missing -> "needs_information"
  (collect ALL missing fields, one reason each).
- Known failure + missing fields -> "not_eligible" (failure wins,
  but missing fields are still reported, never called eligible).
- Malformed/unknown rule format -> raise EligibilityEngineError,
  never silently treat as satisfied.

Single source of truth: evaluate_rule_outcomes() produces one
structured outcome per rule; evaluate() aggregates them and
NearMissEngine reuses them. No second eligibility implementation.

Must NEVER import or call the LLM. No scores or ranking here.
"""

from dataclasses import dataclass
from typing import Any, Literal

from intelligence_engine.schemas import EligibilityResult, Scheme, UserProfile


class EligibilityEngineError(ValueError):
    """Raised when a scheme's eligibility_rules are malformed or unsupported."""


_SUPPORTED_RULES: tuple[str, ...] = (
    "min_age",
    "max_age",
    "max_annual_income",
    "occupations",
    "min_education",
    "categories",
    "states",
)

_EDUCATION_RANKS: dict[str, int] = {
    "illiterate": 0,
    "primary": 1,
    "upper primary": 2,
    "secondary": 3,
    "10th": 4,
    "matric": 4,
    "matriculation": 4,
    "12th": 5,
    "intermediate": 5,
    "higher secondary": 5,
    "diploma": 5,
    "iti": 5,
    "graduate": 6,
    "graduation": 6,
    "bachelor": 6,
    "postgraduate": 7,
    "master": 7,
    "phd": 8,
    "doctorate": 8,
}


@dataclass(frozen=True)
class RuleOutcome:
    """Structured outcome of one rule check (shared with NearMissEngine).

    INTERNAL-ONLY: never returned across the Intelligence API boundary.
    Public consumers use EligibilityResult/FailedCriterion (Pydantic),
    which carry the same information in serializable form.

    dimension: UserProfile field the rule constrains ("age",
        "annual_family_income", "occupation", "education_level",
        "social_category", "state").
    rule: eligibility_rules key that produced this outcome.
    status: "passed", "failed", or "missing" (user value absent).
    reason: Human-readable explanation (same strings as EligibilityResult).
    user_value: The profile value checked (None when missing).
    required: Human-readable requirement (e.g. ">= 18", "<= 300000",
        "one of: Farmer", "at least: 12th").
    difference: Miss distance where meaningful (bound missed by this
        much, always >= 0); None for categorical mismatches.
    """

    dimension: str = ""
    rule: str = ""
    status: Literal["passed", "failed", "missing"] = "passed"
    reason: str = ""
    user_value: Any = None
    required: str = ""
    difference: float | None = None


def _is_number(value: Any) -> bool:
    """Return True for real int/float values (excluding bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_money(value: float) -> str:
    """Format a money value without decimals when whole."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _education_rank(level: str) -> int | None:
    """Return the rank of a known education level, else None."""
    return _EDUCATION_RANKS.get(" ".join(level.split()).lower())


def _check_list_rule(
    rule_value: Any,
    rule_name: str,
    user_value: str | None,
    user_field: str,
    label: str,
) -> tuple[str | None, str | None, str | None]:
    """Check a list-membership rule. Returns (pass_reason, fail_reason, missing_reason)."""
    if not isinstance(rule_value, list) or not all(isinstance(v, str) for v in rule_value):
        raise EligibilityEngineError(f"Rule '{rule_name}' must be a list of strings.")
    if len(rule_value) == 0:
        return None, None, None
    if user_value is None:
        return None, None, f"{label} information is required to determine eligibility."
    allowed = [v.lower() for v in rule_value]
    if user_value.lower() in allowed:
        return f"{label} '{user_value}' matches the required {label.lower()}: {', '.join(rule_value)}.", None, None
    return None, f"{label} '{user_value}' does not match the required {label.lower()}: {', '.join(rule_value)}.", None


def evaluate_rule_outcomes(profile: UserProfile, rules: dict[str, Any]) -> list[RuleOutcome]:
    """Evaluate each defined rule into a structured outcome (single source of truth).

    Args:
        profile: Validated user profile (missing fields are None).
        rules: Scheme eligibility_rules dict.

    Returns:
        One RuleOutcome per defined rule, in fixed dimension order.
        Rules that are empty lists impose nothing and yield no outcome.

    Raises:
        EligibilityEngineError: On malformed or unsupported rule formats.
    """
    unknown = [key for key in rules if key not in _SUPPORTED_RULES]
    if unknown:
        raise EligibilityEngineError(f"Unsupported eligibility rule(s): {', '.join(sorted(unknown))}.")
    outcomes: list[RuleOutcome] = []

    # min_age
    if "min_age" in rules:
        rule = rules["min_age"]
        if not _is_number(rule):
            raise EligibilityEngineError("Rule 'min_age' must be a number.")
        if profile.age is None:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="min_age",
                    status="missing",
                    reason="Age information is required to determine eligibility.",
                    user_value=None,
                    required=f">= {rule}",
                    difference=None,
                )
            )
        elif profile.age < rule:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="min_age",
                    status="failed",
                    reason=f"Age {profile.age} is below the minimum age requirement of {rule}.",
                    user_value=profile.age,
                    required=f">= {rule}",
                    difference=float(rule - profile.age),
                )
            )
        else:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="min_age",
                    status="passed",
                    reason=f"Age {profile.age} meets the minimum age requirement of {rule}.",
                    user_value=profile.age,
                    required=f">= {rule}",
                    difference=None,
                )
            )

    # max_age
    if "max_age" in rules:
        rule = rules["max_age"]
        if not _is_number(rule):
            raise EligibilityEngineError("Rule 'max_age' must be a number.")
        if profile.age is None:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="max_age",
                    status="missing",
                    reason="Age information is required to determine eligibility.",
                    user_value=None,
                    required=f"<= {rule}",
                    difference=None,
                )
            )
        elif profile.age > rule:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="max_age",
                    status="failed",
                    reason=f"Age {profile.age} is above the maximum age requirement of {rule}.",
                    user_value=profile.age,
                    required=f"<= {rule}",
                    difference=float(profile.age - rule),
                )
            )
        else:
            outcomes.append(
                RuleOutcome(
                    dimension="age",
                    rule="max_age",
                    status="passed",
                    reason=f"Age {profile.age} meets the maximum age requirement of {rule}.",
                    user_value=profile.age,
                    required=f"<= {rule}",
                    difference=None,
                )
            )

    # max_annual_income
    if "max_annual_income" in rules:
        rule = rules["max_annual_income"]
        if not _is_number(rule):
            raise EligibilityEngineError("Rule 'max_annual_income' must be a number.")
        if profile.annual_family_income is None:
            outcomes.append(
                RuleOutcome(
                    dimension="annual_family_income",
                    rule="max_annual_income",
                    status="missing",
                    reason="Annual family income information is required to determine eligibility.",
                    user_value=None,
                    required=f"<= {_format_money(rule)}",
                    difference=None,
                )
            )
        elif profile.annual_family_income > rule:
            outcomes.append(
                RuleOutcome(
                    dimension="annual_family_income",
                    rule="max_annual_income",
                    status="failed",
                    reason=(
                        f"Annual family income of \u20b9{_format_money(profile.annual_family_income)}"
                        f" exceeds the maximum allowed income of \u20b9{_format_money(rule)}."
                    ),
                    user_value=profile.annual_family_income,
                    required=f"<= {_format_money(rule)}",
                    difference=float(profile.annual_family_income - rule),
                )
            )
        else:
            outcomes.append(
                RuleOutcome(
                    dimension="annual_family_income",
                    rule="max_annual_income",
                    status="passed",
                    reason=(
                        f"Annual family income of \u20b9{_format_money(profile.annual_family_income)}"
                        f" is within the maximum allowed income of \u20b9{_format_money(rule)}."
                    ),
                    user_value=profile.annual_family_income,
                    required=f"<= {_format_money(rule)}",
                    difference=None,
                )
            )

    # occupations
    if "occupations" in rules:
        ok, bad, need = _check_list_rule(
            rules["occupations"], "occupations", profile.occupation, "occupation", "Occupation"
        )
        required = f"one of: {', '.join(rules['occupations'])}" if rules["occupations"] else ""
        if ok is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="occupation",
                    rule="occupations",
                    status="passed",
                    reason=ok,
                    user_value=profile.occupation,
                    required=required,
                    difference=None,
                )
            )
        elif bad is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="occupation",
                    rule="occupations",
                    status="failed",
                    reason=bad,
                    user_value=profile.occupation,
                    required=required,
                    difference=None,
                )
            )
        elif need is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="occupation",
                    rule="occupations",
                    status="missing",
                    reason=need,
                    user_value=None,
                    required=required,
                    difference=None,
                )
            )

    # min_education
    if "min_education" in rules:
        rule = rules["min_education"]
        if not isinstance(rule, str) or not rule.strip():
            raise EligibilityEngineError("Rule 'min_education' must be a non-empty string.")
        if profile.education_level is None:
            outcomes.append(
                RuleOutcome(
                    dimension="education_level",
                    rule="min_education",
                    status="missing",
                    reason="Education information is required to determine eligibility.",
                    user_value=None,
                    required=f"at least: {rule}",
                    difference=None,
                )
            )
        else:
            required_rank = _education_rank(rule)
            user_rank = _education_rank(profile.education_level)
            if required_rank is not None and user_rank is not None:
                if user_rank >= required_rank:
                    outcomes.append(
                        RuleOutcome(
                            dimension="education_level",
                            rule="min_education",
                            status="passed",
                            reason=(
                                f"Education '{profile.education_level}' meets the minimum education "
                                f"requirement of '{rule}'."
                            ),
                            user_value=profile.education_level,
                            required=f"at least: {rule}",
                            difference=None,
                        )
                    )
                else:
                    outcomes.append(
                        RuleOutcome(
                            dimension="education_level",
                            rule="min_education",
                            status="failed",
                            reason=(
                                f"Education '{profile.education_level}' does not meet the minimum education "
                                f"requirement of '{rule}'."
                            ),
                            user_value=profile.education_level,
                            required=f"at least: {rule}",
                            difference=float(required_rank - user_rank),
                        )
                    )
            elif profile.education_level.strip().lower() == rule.strip().lower():
                outcomes.append(
                    RuleOutcome(
                        dimension="education_level",
                        rule="min_education",
                        status="passed",
                        reason=(
                            f"Education '{profile.education_level}' meets the minimum education "
                            f"requirement of '{rule}'."
                        ),
                        user_value=profile.education_level,
                        required=f"at least: {rule}",
                        difference=None,
                    )
                )
            else:
                outcomes.append(
                    RuleOutcome(
                        dimension="education_level",
                        rule="min_education",
                        status="failed",
                        reason=(
                            f"Education '{profile.education_level}' does not meet the minimum education "
                            f"requirement of '{rule}'."
                        ),
                        user_value=profile.education_level,
                        required=f"at least: {rule}",
                        difference=None,
                    )
                )

    # categories (social_category)
    if "categories" in rules:
        ok, bad, need = _check_list_rule(
            rules["categories"], "categories", profile.social_category, "social_category", "Social category"
        )
        required = f"one of: {', '.join(rules['categories'])}" if rules["categories"] else ""
        if ok is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="social_category",
                    rule="categories",
                    status="passed",
                    reason=ok,
                    user_value=profile.social_category,
                    required=required,
                    difference=None,
                )
            )
        elif bad is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="social_category",
                    rule="categories",
                    status="failed",
                    reason=bad,
                    user_value=profile.social_category,
                    required=required,
                    difference=None,
                )
            )
        elif need is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="social_category",
                    rule="categories",
                    status="missing",
                    reason=need,
                    user_value=None,
                    required=required,
                    difference=None,
                )
            )

    # states
    if "states" in rules:
        ok, bad, need = _check_list_rule(
            rules["states"], "states", profile.state, "state", "State"
        )
        required = f"one of: {', '.join(rules['states'])}" if rules["states"] else ""
        if ok is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="state",
                    rule="states",
                    status="passed",
                    reason=ok,
                    user_value=profile.state,
                    required=required,
                    difference=None,
                )
            )
        elif bad is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="state",
                    rule="states",
                    status="failed",
                    reason=bad,
                    user_value=profile.state,
                    required=required,
                    difference=None,
                )
            )
        elif need is not None:
            outcomes.append(
                RuleOutcome(
                    dimension="state",
                    rule="states",
                    status="missing",
                    reason=need,
                    user_value=None,
                    required=required,
                    difference=None,
                )
            )

    return outcomes


class EligibilityEngine:
    """Deterministic evaluator of UserProfile against Scheme rules."""

    def evaluate(self, profile: UserProfile, scheme: Scheme) -> EligibilityResult:
        """Evaluate one profile against one scheme's mandatory rules.

        Args:
            profile: Validated user profile (missing fields are None).
            scheme: Scheme with structured eligibility_rules.

        Returns:
            EligibilityResult with status, reasons, and missing_information.

        Raises:
            EligibilityEngineError: On malformed or unsupported rule formats.
        """
        outcomes = evaluate_rule_outcomes(profile, scheme.eligibility_rules)

        passed: list[str] = []
        failed: list[str] = []
        missing: list[str] = []
        missing_reasons: list[str] = []
        for outcome in outcomes:
            if outcome.status == "passed":
                passed.append(outcome.reason)
            elif outcome.status == "failed":
                failed.append(outcome.reason)
            elif outcome.dimension not in missing:
                missing.append(outcome.dimension)
                missing_reasons.append(outcome.reason)

        if failed:
            return EligibilityResult(
                scheme_id=scheme.id,
                status="not_eligible",
                reasons=failed + missing_reasons,
                missing_information=missing,
            )
        if missing:
            return EligibilityResult(
                scheme_id=scheme.id,
                status="needs_information",
                reasons=missing_reasons,
                missing_information=missing,
            )
        if not passed:
            return EligibilityResult(
                scheme_id=scheme.id,
                status="eligible",
                reasons=["No mandatory eligibility rules defined for this scheme."],
            )
        return EligibilityResult(scheme_id=scheme.id, status="eligible", reasons=passed)

    def evaluate_many(self, profile: UserProfile, schemes: list[Scheme]) -> list[EligibilityResult]:
        """Evaluate one profile against many schemes in order."""
        return [self.evaluate(profile, scheme) for scheme in schemes]
