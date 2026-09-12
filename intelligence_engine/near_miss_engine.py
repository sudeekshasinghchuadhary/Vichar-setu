"""Near-miss analysis: how close a failed eligibility result is.

Flow: EligibilityEngine -> not_eligible -> NearMissEngine -> NearMissResult.

INTERIM POLICY: CONSERVATIVE-OFF. No criterion currently has
domain-validated closeness support (see _VALIDATED_CLOSENESS_CRITERIA),
so is_near_miss is always False. The architecture is preserved: when a
criterion gains validated support, adding its dimension name to that set
is the entire policy change — no engine rewrite needed.

NearMissEngine NEVER changes eligibility: it reads the status, analyzes
the shared rule outcomes (same source of truth as EligibilityEngine),
and reports. It uses no embeddings, match scores, LLM, database, or API.

Each FailedCriterion carries machine-usable fields (criterion,
user_value, required, difference) so evidence stays available for
explanation, What-If, and any future validated policy.
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


# POLICY BOUNDARY: dimensions with domain-validated closeness support.
# Empty under the interim CONSERVATIVE-OFF policy: no positive
# near-miss classification is possible until a criterion is added here
# together with its validating evidence and tests. Adding a name is the
# whole of any future per-criterion enablement; no other code changes.
_VALIDATED_CLOSENESS_CRITERIA: frozenset[str] = frozenset()


def _criterion_has_closeness_support(dimension: str) -> bool:
    """Whether a failed dimension has validated closeness support."""
    return dimension in _VALIDATED_CLOSENESS_CRITERIA


class NearMissEngine:
    """Analyze not_eligible outcomes for rule-based closeness."""

    def __init__(self, require_no_missing: bool = True) -> None:
        """Configure the transparent closeness definition.

        Args:
            require_no_missing: When True (default), any unknown dimension
                disqualifies near-miss status.
        """
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
            status "not_eligible" where every failed dimension has
            validated closeness support and (by default) no missing
            dimensions. No count cap, no aggregate score.
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
        supported = [_criterion_has_closeness_support(criterion.criterion) for criterion in failed]
        is_near_miss = (
            eligibility.status == "not_eligible"
            and len(failed) > 0
            and (not self.require_no_missing or len(missing_dims) == 0)
            and all(supported)
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
