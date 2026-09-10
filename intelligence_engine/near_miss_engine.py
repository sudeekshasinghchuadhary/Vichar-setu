"""Near-miss analysis: how close a failed eligibility result is.

Flow: EligibilityEngine -> not_eligible -> NearMissEngine -> NearMissResult.

A near miss means: exactly the known mandatory criteria pass except a
small configurable number of failed dimensions (default: 1), with no
unknown (missing) dimensions. Missing information is never a failure
and disqualifies near-miss status — closeness cannot be claimed about
requirements whose values are unknown.

NearMissEngine NEVER changes eligibility: it reads the status, analyzes
the shared rule outcomes (same source of truth as EligibilityEngine),
and reports. It uses no embeddings, match scores, LLM, database, or API.

Each FailedCriterion carries machine-usable fields (criterion,
user_value, required, difference) so a future What-If engine can
simulate "what would change this outcome" directly from NearMissResult.
"""

from intelligence_engine.eligibility_engine import (
    EligibilityEngine,
    RuleOutcome,
    evaluate_rule_outcomes,
)
from intelligence_engine.schemas import (
    EligibilityResult,
    FailedCriterion,
    NearMissResult,
    Scheme,
    UserProfile,
)


def _format_number(value: float) -> str:
    """Format a difference without decimals when whole."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _proximity_detail(dimension: str, difference: float | None) -> str | None:
    """Humanize a numeric miss distance; None when distance is meaningless."""
    if difference is None:
        return None
    if dimension == "annual_family_income":
        return f"Over by \u20b9{_format_number(difference)}."
    if dimension == "age":
        return f"Off by {_format_number(difference)} year(s)."
    if dimension == "education_level":
        return f"Short by {_format_number(difference)} education level(s)."
    return None


class NearMissEngine:
    """Analyze not_eligible outcomes for rule-based closeness."""

    def __init__(self, max_failed_criteria: int = 1, require_no_missing: bool = True) -> None:
        """Configure the transparent closeness definition.

        Args:
            max_failed_criteria: Maximum failed dimensions still "close".
                Default 1 ("one criterion away"); no percentages, no
                hidden thresholds.
            require_no_missing: When True (default), any unknown dimension
                disqualifies near-miss status.
        """
        if not isinstance(max_failed_criteria, int) or max_failed_criteria < 1:
            raise ValueError("max_failed_criteria must be a positive integer.")
        self.max_failed_criteria = max_failed_criteria
        self.require_no_missing = require_no_missing

    def analyze(
        self,
        profile: UserProfile,
        scheme: Scheme,
        eligibility: EligibilityResult | None = None,
    ) -> NearMissResult:
        """Analyze how close the profile is on a failed scheme.

        Args:
            profile: Validated user profile.
            scheme: Scheme whose rules were evaluated.
            eligibility: Existing result to respect (never recomputed
                into a different status). Computed when omitted.

        Returns:
            NearMissResult. is_near_miss is True only for
            status "not_eligible" with few enough failures and (by
            default) no missing dimensions.
        """
        if eligibility is None:
            eligibility = EligibilityEngine().evaluate(profile, scheme)
        outcomes = evaluate_rule_outcomes(profile, scheme.eligibility_rules)

        by_dimension: dict[str, list[RuleOutcome]] = {}
        for outcome in outcomes:
            by_dimension.setdefault(outcome.dimension, []).append(outcome)

        failed: list[FailedCriterion] = []
        satisfied: list[str] = []
        missing_dims: list[str] = []
        failed_reasons: list[str] = []
        for dimension, dim_outcomes in by_dimension.items():
            if any(outcome.status == "failed" for outcome in dim_outcomes):
                failed_outcomes = [o for o in dim_outcomes if o.status == "failed"]
                failed.append(
                    FailedCriterion(
                        criterion=dimension,
                        user_value=failed_outcomes[0].user_value,
                        required=" and ".join(o.required for o in failed_outcomes),
                        difference=max(
                            (o.difference for o in failed_outcomes if o.difference is not None),
                            default=None,
                        ),
                    )
                )
                failed_reasons.extend(o.reason for o in failed_outcomes)
            elif any(outcome.status == "missing" for outcome in dim_outcomes):
                missing_dims.append(dimension)
            else:
                satisfied.append(dimension)

        total = len(by_dimension)
        is_near_miss = (
            eligibility.status == "not_eligible"
            and 0 < len(failed) <= self.max_failed_criteria
            and (not self.require_no_missing or len(missing_dims) == 0)
        )

        reasons: list[str] = []
        if total == 0:
            reasons.append("No mandatory eligibility rules defined for this scheme.")
        else:
            satisfied_label = ", ".join(satisfied) if satisfied else "none"
            reasons.append(
                f"You meet {len(satisfied)} of {total} mandatory criteria ({satisfied_label})."
            )
        reasons.extend(failed_reasons)
        for criterion in failed:
            detail = _proximity_detail(criterion.criterion, criterion.difference)
            if detail is not None:
                reasons.append(detail)
        for dimension in missing_dims:
            reasons.append(f"{dimension} information is still needed for a full assessment.")
        if is_near_miss:
            reasons.append(
                f"Only {len(failed)} of {total} criteria failed \u2014 close to eligible."
            )
            reasons.append("Near miss does not guarantee approval.")

        return NearMissResult(
            scheme_id=scheme.id,
            is_near_miss=is_near_miss,
            failed_criteria=failed,
            satisfied_criteria=satisfied,
            total_criteria=total,
            reasons=reasons,
        )
