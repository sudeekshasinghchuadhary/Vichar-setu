"""What-if simulation: hypothetical profiles through the real EligibilityEngine.

Flow: current profile + current eligibility + requested changes ->
temporary hypothetical UserProfile (validated, never persisted) ->
EXISTING EligibilityEngine on the same verified rules ->
structured current-vs-hypothetical comparison.

Single source of truth: all decisions come from EligibilityEngine
(evaluate() and evaluate_rule_outcomes()). No second engine, no
duplicated rules, no LLM, no scores, no approval predictions, and
the original profile is never mutated.
"""

from typing import Any

from pydantic import ValidationError

from intelligence_engine.eligibility_engine import EligibilityEngine, evaluate_rule_outcomes
from intelligence_engine.schemas import (
    AppliedChange,
    EligibilityResult,
    Scheme,
    UserProfile,
    WhatIfChange,
    WhatIfResult,
)


class WhatIfError(ValueError):
    """Raised for unknown fields or invalid hypothetical values (never guessed)."""


class WhatIfEngine:
    """Simulate hypothetical profile edits against verified scheme rules."""

    def __init__(self, eligibility_engine: EligibilityEngine | None = None) -> None:
        """Store the injected engine (default: a real EligibilityEngine)."""
        self.eligibility_engine = eligibility_engine or EligibilityEngine()

    def simulate(
        self,
        profile: UserProfile,
        scheme: Scheme,
        changes: list[WhatIfChange] | dict[str, Any],
        current: EligibilityResult | None = None,
    ) -> WhatIfResult:
        """Apply hypothetical changes and compare eligibility outcomes.

        Args:
            profile: Real user profile (never mutated).
            scheme: Scheme with verified eligibility rules.
            changes: List of WhatIfChange, or a {field: value} dict.
            current: Existing eligibility result (evaluated when omitted).

        Returns:
            WhatIfResult with applied changes, both outcomes, the
            temporary profile, and deterministic comparison reasons.

        Raises:
            WhatIfError: For unknown fields or values failing UserProfile
                validation (e.g. negative income, wrong types).
        """
        change_list = self._normalize_changes(changes)
        known_fields = set(UserProfile.model_fields)
        for change in change_list:
            if change.field not in known_fields:
                raise WhatIfError(f"Unknown profile field: '{change.field}'.")

        merged = profile.model_dump()
        for change in change_list:
            merged[change.field] = change.hypothetical_value
        try:
            hypothetical_profile = UserProfile.model_validate(merged)
        except ValidationError as exc:
            raise WhatIfError(f"Invalid hypothetical profile: {exc}") from exc

        if current is None:
            current = self.eligibility_engine.evaluate(profile, scheme)
        hypothetical = self.eligibility_engine.evaluate(hypothetical_profile, scheme)

        current_by_dim = self._status_by_dimension(profile, scheme)
        hypo_by_dim = self._status_by_dimension(hypothetical_profile, scheme)
        changed_criteria = sorted(
            dim for dim in current_by_dim if current_by_dim[dim] != hypo_by_dim.get(dim)
        )
        remaining_failures = sorted(dim for dim, status in hypo_by_dim.items() if status == "failed")

        applied = [
            AppliedChange(
                field=change.field,
                current_value=getattr(profile, change.field),
                hypothetical_value=change.hypothetical_value,
            )
            for change in change_list
        ]
        reasons: list[str] = []
        for item in applied:
            reasons.append(
                f"Changed {item.field} from {item.current_value!r} to {item.hypothetical_value!r}."
            )
        reasons.append(f"Current eligibility: {current.status}.")
        reasons.append(f"Hypothetical eligibility: {hypothetical.status}.")
        if changed_criteria:
            reasons.append(f"Criteria with changed outcomes: {', '.join(changed_criteria)}.")
        if remaining_failures:
            reasons.append(f"Still failing: {', '.join(remaining_failures)}.")
        if hypothetical.status == "eligible":
            reasons.append(
                "The hypothetical profile satisfies the currently defined rules. "
                "This is not a guarantee of real-world approval."
            )

        return WhatIfResult(
            scheme_id=scheme.id,
            changes=applied,
            current=current,
            hypothetical=hypothetical,
            hypothetical_profile=hypothetical_profile,
            status_changed=current.status != hypothetical.status,
            changed_criteria=changed_criteria,
            remaining_failures=remaining_failures,
            reasons=reasons,
        )

    @staticmethod
    def _normalize_changes(changes: list[WhatIfChange] | dict[str, Any]) -> list[WhatIfChange]:
        """Accept a list or a plain {field: value} dict deterministically."""
        if isinstance(changes, dict):
            return [WhatIfChange(field=field, hypothetical_value=value) for field, value in changes.items()]
        if isinstance(changes, list) and all(isinstance(item, WhatIfChange) for item in changes):
            return list(changes)
        raise WhatIfError("changes must be a list of WhatIfChange or a {field: value} dict.")

    @staticmethod
    def _status_by_dimension(profile: UserProfile, scheme: Scheme) -> dict[str, str]:
        """Worst outcome status per eligibility dimension (failed > missing > passed)."""
        rank = {"passed": 0, "missing": 1, "failed": 2}
        by_dim: dict[str, str] = {}
        for outcome in evaluate_rule_outcomes(profile, scheme.eligibility_rules):
            if rank[outcome.status] > rank.get(by_dim.get(outcome.dimension, "passed"), 0):
                by_dim[outcome.dimension] = outcome.status
        return by_dim
