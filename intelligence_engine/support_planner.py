"""Support planning: map structured needs to verified scheme support.

Flow: NeedAnalysisResult + schemes + eligibility results ->
per-need mappings with eligibility-aware, amount-aware coverage ->
SupportPlan with gaps and next actions.

Rules (never violated):
- Only structured SchemeSupport entries map; free-text benefit,
  descriptions, URLs, and documents never imply support or amounts.
- Coverage only when: need amount known + need and support periods equal
  and explicitly known + support max_amount known + eligibility eligible.
  Unknown or mismatched periods never convert; monthly is never annualized.
- Unknown max ≠ 0, unknown ≠ full coverage, loan ≠ grant/subsidy,
  scheme fit ≠ guaranteed funding.
- Schemes are never summed: best single-scheme coverage only, because
  no convergence data exists (convergence_unknown stays True).
- Inputs are never mutated. No LLM, embeddings, DB, or API.
"""

from intelligence_engine.schemas import (
    EligibilityResult,
    NeedPlan,
    Scheme,
    SchemeSupport,
    SupportMapping,
    SupportNeed,
    SupportPlan,
    UserProfile,
)


def _format_money(value: float) -> str:
    """Format amounts without decimals when whole."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


class SupportPlanningEngine:
    """Deterministic need-to-support mapper and gap analyzer."""

    def plan(
        self,
        profile: UserProfile,
        needs: list[SupportNeed],
        schemes: list[Scheme],
        eligibility_results: list[EligibilityResult] | None = None,
    ) -> SupportPlan:
        """Build a SupportPlan without mutating inputs.

        Args:
            profile: Validated user profile (kept for future use and
                serialized context; planning reads needs, not profile fields).
            needs: Structured support needs to cover.
            schemes: Candidate schemes with verified support_options.
            eligibility_results: Status per scheme; absent schemes count
                as "unknown" (uncertain, never covered).

        Returns:
            SupportPlan with per-need mappings, gaps, and next actions.
        """
        _ = profile
        status_by_scheme = {item.scheme_id: item.status for item in (eligibility_results or [])}
        need_plans: list[NeedPlan] = []
        for need in needs:
            mappings: list[SupportMapping] = []
            seen_schemes: set[str] = set()
            for scheme in schemes:
                if scheme.id in seen_schemes:
                    continue
                seen_schemes.add(scheme.id)
                for support in scheme.support_options:
                    if need.need_type not in support.covers_need_types:
                        continue
                    mappings.append(self._map_need(need, scheme, support, status_by_scheme))
            # Deduplicate identical mappings; schemes evaluated once per need.
            unique: list[SupportMapping] = []
            seen_keys: set[tuple[str, str, str]] = set()
            for mapping in mappings:
                key = (mapping.scheme_id, mapping.support_type, str(mapping.max_amount))
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique.append(mapping)
            need_plans.append(self._plan_need(need, unique))

        total_need = sum(need.amount for need in needs if self._comparable(need))
        known_needs = [need for need in needs if self._comparable(need)]
        single_period = len({need.amount_period for need in known_needs}) <= 1
        bests = [plan.best_known_coverage for plan in need_plans]
        totals_fully_known = (
            bool(known_needs)
            and single_period
            and all(
                plan.best_known_coverage is not None
                for plan, need in zip(need_plans, needs)
                if self._comparable(need)
            )
        )
        total_best = sum(best for best in bests if best is not None)
        plan_reasons = [
            "No convergence data exists: schemes are assumed non-combinable; "
            "amounts from different schemes are never added together."
        ]
        if known_needs and not single_period:
            plan_reasons.append(
                "Needs span multiple periods: totals cannot mix periods, "
                "so plan-level totals are left unknown."
            )
        plan = SupportPlan(
            need_plans=need_plans,
            total_need_amount=float(total_need) if (known_needs and single_period) else None,
            total_best_known_coverage=float(total_best) if totals_fully_known else None,
            total_uncovered_amount=(
                float(total_need - total_best) if totals_fully_known else None
            ),
            totals_fully_known=totals_fully_known,
            convergence_unknown=True,
            reasons=plan_reasons,
            next_actions=self._next_actions(need_plans),
        )
        return plan

    @staticmethod
    def _comparable(need: SupportNeed) -> bool:
        """Only known amounts with a known period can enter arithmetic."""
        return need.amount is not None and need.amount_period != "unknown"

    @staticmethod
    def _periods_compatible(need: SupportNeed, support: SchemeSupport) -> bool:
        """Coverage needs equal, explicitly known periods (never converted)."""
        return (
            need.amount_period != "unknown"
            and support.max_amount_period != "unknown"
            and need.amount_period == support.max_amount_period
        )

    def _map_need(
        self,
        need: SupportNeed,
        scheme: Scheme,
        support: SchemeSupport,
        status_by_scheme: dict[str, str],
    ) -> SupportMapping:
        """Map one need to one verified support option with evidence."""
        status = status_by_scheme.get(scheme.id, "unknown")
        reasons = [
            f"Scheme '{scheme.id}' offers verified {support.support_type} support "
            f"covering '{need.need_type}'."
        ]
        if status != "eligible":
            if status == "not_eligible":
                reasons.append("User is not eligible for this scheme: no coverage counted.")
            elif status == "needs_information":
                reasons.append("Eligibility is unresolved: coverage cannot be confirmed.")
            else:
                reasons.append("No eligibility result for this scheme: coverage cannot be confirmed.")
            return SupportMapping(
                scheme_id=scheme.id,
                support_type=support.support_type,
                eligibility_status=status,
                max_amount=support.max_amount,
                max_amount_period=support.max_amount_period,
                coverage_known=False,
                reasons=reasons,
            )
        if not self._comparable(need):
            reasons.append("Need amount or period unknown: coverage cannot be calculated.")
            return SupportMapping(
                scheme_id=scheme.id,
                support_type=support.support_type,
                eligibility_status=status,
                max_amount=support.max_amount,
                max_amount_period=support.max_amount_period,
                coverage_known=False,
                reasons=reasons,
            )
        if support.max_amount is None:
            reasons.append("Scheme support amount unknown: coverage cannot be calculated.")
            return SupportMapping(
                scheme_id=scheme.id,
                support_type=support.support_type,
                eligibility_status=status,
                max_amount=None,
                max_amount_period=support.max_amount_period,
                coverage_known=False,
                reasons=reasons,
            )
        if not self._periods_compatible(need, support):
            reasons.append(
                f"Period mismatch (need {need.amount_period} vs support "
                f"{support.max_amount_period}): coverage cannot be calculated."
            )
            return SupportMapping(
                scheme_id=scheme.id,
                support_type=support.support_type,
                eligibility_status=status,
                max_amount=support.max_amount,
                max_amount_period=support.max_amount_period,
                coverage_known=False,
                reasons=reasons,
            )
        assert need.amount is not None
        coverage = min(need.amount, support.max_amount)
        uncovered = need.amount - coverage
        reasons.append(
            f"Potential coverage ₹{_format_money(coverage)} of ₹{_format_money(need.amount)} "
            f"needed ({support.support_type}, capped at ₹{_format_money(support.max_amount)})."
        )
        return SupportMapping(
            scheme_id=scheme.id,
            support_type=support.support_type,
            eligibility_status=status,
            max_amount=support.max_amount,
            max_amount_period=support.max_amount_period,
            potential_coverage=float(coverage),
            uncovered_amount=float(uncovered),
            coverage_known=True,
            reasons=reasons,
        )

    def _plan_need(self, need: SupportNeed, mappings: list[SupportMapping]) -> NeedPlan:
        """Summarize one need: best single-scheme coverage, never a sum."""
        known = [mapping.potential_coverage for mapping in mappings if mapping.coverage_known]
        best = max(known) if known else None
        uncovered = need.amount - best if (best is not None and self._comparable(need)) else None
        assert need.amount is None or uncovered is None or uncovered >= 0
        fully = uncovered == 0 if uncovered is not None else False
        reasons: list[str] = []
        if not mappings:
            reasons.append(f"No verified scheme support found for '{need.need_type}'.")
        elif best is None:
            reasons.append("Coverage unknown for all candidate supports.")
        elif fully:
            reasons.append("One scheme alone can cover the full need.")
        else:
            reasons.append(
                f"Best single-scheme coverage leaves ₹{_format_money(uncovered or 0)} uncovered."
            )
        return NeedPlan(
            need=need,
            mappings=mappings,
            best_known_coverage=float(best) if best is not None else None,
            uncovered_amount=float(uncovered) if uncovered is not None else None,
            fully_covered=fully,
            reasons=reasons,
        )

    def _next_actions(self, need_plans: list[NeedPlan]) -> list[str]:
        """Derive deterministic follow-ups from gaps and uncertainty."""
        actions: list[str] = []
        for plan in need_plans:
            label = plan.need.need_type
            if not plan.mappings:
                actions.append(f"Find verified scheme support for '{label}'.")
                continue
            for mapping in plan.mappings:
                if mapping.eligibility_status == "needs_information":
                    actions.append(f"Resolve eligibility inputs for scheme '{mapping.scheme_id}'.")
                elif mapping.eligibility_status == "unknown":
                    actions.append(f"Evaluate eligibility for scheme '{mapping.scheme_id}'.")
            if plan.uncovered_amount:
                actions.append(
                    f"Address uncovered ₹{_format_money(plan.uncovered_amount)} for '{label}'."
                )
            elif plan.best_known_coverage is None and plan.mappings:
                actions.append(f"Confirm support amounts covering '{label}'.")
        seen: set[str] = set()
        ordered: list[str] = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                ordered.append(action)
        return ordered
