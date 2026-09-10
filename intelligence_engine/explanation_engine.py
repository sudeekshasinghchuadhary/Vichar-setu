"""Explanation & decision-trace layer: explain, never decide.

Consumes structured outputs of the authoritative engines and restates
them as DecisionTrace objects with exactly one "decision" item plus
supporting evidence/calculation/uncertainty/action items.

Anti-invention rules:
- Verdicts come only from upstream status/flag fields (fixed templates).
- Evidence texts are verbatim upstream reasons/values, tagged with the
  source engine name. Nothing is paraphrased into new claims.
- Empty upstream reasons produce an explicit "information unavailable"
  uncertainty item instead of a guess.
- Uncertainty (missing/unknown) is listed, never turned into rejection.
- No eligibility/matching/financial recomputation, LLM, DB, or API.
"""

from intelligence_engine.schemas import (
    CoverageResult,
    DecisionTrace,
    EligibilityResult,
    ExplanationItem,
    MatchResult,
    NearMissResult,
    NeedAnalysisResult,
    PathwayResult,
    SupportPlan,
    WhatIfResult,
)


def _item(kind: str, text: str, source: str) -> ExplanationItem:
    """Build one trace item."""
    return ExplanationItem(kind=kind, text=text, source=source)  # type: ignore[arg-type]


def _evidence_or_unavailable(reasons: list[str], source: str) -> list[ExplanationItem]:
    """Verbatim reasons as evidence, or an explicit unavailable marker."""
    if reasons:
        return [_item("evidence", reason, source) for reason in reasons]
    return [_item("uncertainty", "Supporting detail unavailable from upstream result.", source)]


class ExplanationEngine:
    """Stateless consumer of engine outputs; holds no decision logic."""

    def explain_eligibility(self, result: EligibilityResult) -> DecisionTrace:
        """Trace an eligibility verdict with its evidence and gaps."""
        verdicts = {
            "eligible": f"Scheme '{result.scheme_id}' is eligible.",
            "not_eligible": f"Scheme '{result.scheme_id}' is not eligible.",
            "needs_information": f"Eligibility for scheme '{result.scheme_id}' cannot be determined yet.",
        }
        items = [_item("decision", verdicts[result.status], "EligibilityEngine")]
        items.extend(_evidence_or_unavailable(result.reasons, "EligibilityEngine"))
        for field in result.missing_information:
            items.append(_item("uncertainty", f"Missing information: {field}.", "EligibilityEngine"))
            items.append(_item("action", f"Provide {field} to complete the eligibility check.", "EligibilityEngine"))
        return DecisionTrace(subject=f"eligibility:{result.scheme_id}", items=items)

    def explain_matching(self, match: MatchResult) -> DecisionTrace:
        """Trace a ranking outcome with the score calculation shown."""
        items = [
            _item(
                "decision",
                f"Scheme '{match.scheme_id}' ranked #{match.rank} with match score {match.score}/100.",
                "MatchingEngine",
            ),
            _item(
                "calculation",
                f"Deterministic fit score {match.score} on a 0-100 scale (internal ranking signal).",
                "MatchingEngine",
            ),
        ]
        items.extend(_evidence_or_unavailable(match.reasons, "MatchingEngine"))
        return DecisionTrace(subject=f"matching:{match.scheme_id}", items=items)

    def explain_near_miss(self, result: NearMissResult) -> DecisionTrace:
        """Trace a near-miss analysis with structured failed criteria."""
        verdict = (
            f"Scheme '{result.scheme_id}' is a near miss."
            if result.is_near_miss
            else f"Scheme '{result.scheme_id}' is not a near miss."
        )
        items = [_item("decision", verdict, "NearMissEngine")]
        items.extend(_evidence_or_unavailable(result.reasons, "NearMissEngine"))
        for criterion in result.failed_criteria:
            items.append(
                _item(
                    "evidence",
                    f"Failed criterion '{criterion.criterion}': user value "
                    f"{criterion.user_value!r}, required {criterion.required}.",
                    "NearMissEngine",
                )
            )
            if criterion.difference is not None:
                items.append(
                    _item(
                        "calculation",
                        f"Miss distance for '{criterion.criterion}': {criterion.difference}.",
                        "NearMissEngine",
                    )
                )
        return DecisionTrace(subject=f"near-miss:{result.scheme_id}", items=items)

    def explain_needs(self, result: NeedAnalysisResult) -> DecisionTrace:
        """Trace extracted goals/needs with unknowns preserved."""
        items = [
            _item(
                "decision",
                f"Identified {len(result.needs)} support need(s)"
                + (f" for goal '{result.business_goal}'." if result.business_goal else "."),
                "NeedAnalyzer",
            )
        ]
        for need in result.needs:
            amount = need.amount if need.amount is not None else "unknown amount"
            items.append(
                _item(
                    "evidence",
                    f"Need '{need.need_type}': {amount} (period: {need.amount_period}).",
                    "NeedAnalyzer",
                )
            )
            if need.amount is None:
                items.append(
                    _item("uncertainty", f"Amount unknown for need '{need.need_type}'.", "NeedAnalyzer")
                )
        for note in result.notes:
            items.append(_item("uncertainty", note, "NeedAnalyzer"))
        if not result.needs:
            items.append(_item("uncertainty", "No support needs were identified.", "NeedAnalyzer"))
        return DecisionTrace(subject="need-analysis", items=items)

    def explain_support(self, plan: SupportPlan) -> DecisionTrace:
        """Trace support mappings, coverage, and gaps per need."""
        items = [
            _item(
                "decision",
                f"Support plan covers {len(plan.need_plans)} need(s).",
                "SupportPlanningEngine",
            )
        ]
        for need_plan in plan.need_plans:
            label = need_plan.need.need_type
            items.append(
                _item(
                    "evidence",
                    f"Need '{label}': {len(need_plan.mappings)} candidate support(s).",
                    "SupportPlanningEngine",
                )
            )
            for mapping in need_plan.mappings:
                items.extend(_evidence_or_unavailable(mapping.reasons, "SupportPlanningEngine"))
            if need_plan.best_known_coverage is not None:
                items.append(
                    _item(
                        "calculation",
                        f"Best single-scheme coverage for '{label}': {need_plan.best_known_coverage}.",
                        "SupportPlanningEngine",
                    )
                )
            else:
                items.append(
                    _item("uncertainty", f"Coverage unknown for need '{label}'.", "SupportPlanningEngine")
                )
            if need_plan.uncovered_amount:
                items.append(
                    _item(
                        "calculation",
                        f"Uncovered amount for '{label}': {need_plan.uncovered_amount}.",
                        "SupportPlanningEngine",
                    )
                )
        for action in plan.next_actions:
            items.append(_item("action", action, "SupportPlanningEngine"))
        if not plan.need_plans:
            items.append(_item("uncertainty", "No needs were planned.", "SupportPlanningEngine"))
        return DecisionTrace(subject="support-plan", items=items)

    def explain_financial(self, result: CoverageResult) -> DecisionTrace:
        """Trace coverage arithmetic with known/unknown clearly separated."""
        if not result.coverage_known:
            verdict = "Financial coverage cannot be determined from known values."
        elif result.fully_covered:
            verdict = "Requirement is fully covered by the best known option."
        else:
            verdict = "A financial gap remains after the best known option."
        items = [_item("decision", verdict, "FinancialIntelligence")]
        for reason in result.reasons:
            items.append(_item("evidence", reason, "FinancialIntelligence"))
        if result.coverage_known:
            items.append(
                _item(
                    "calculation",
                    f"Required {result.required}, covered {result.covered}, "
                    f"uncovered {result.uncovered}.",
                    "FinancialIntelligence",
                )
            )
        else:
            items.append(
                _item("uncertainty", "Coverage unknown: a needed value is missing.", "FinancialIntelligence")
            )
        return DecisionTrace(subject="financial-coverage", items=items)

    def explain_what_if(self, result: WhatIfResult) -> DecisionTrace:
        """Trace a hypothetical comparison without new decisions."""
        change = (
            "no changes applied" if not result.changes else f"{len(result.changes)} change(s) applied"
        )
        items = [
            _item(
                "decision",
                f"What-if for scheme '{result.scheme_id}' ({change}): "
                f"{result.current.status} -> {result.hypothetical.status}.",
                "WhatIfEngine",
            )
        ]
        for item in result.changes:
            items.append(
                _item(
                    "evidence",
                    f"{item.field}: {item.current_value!r} -> {item.hypothetical_value!r}.",
                    "WhatIfEngine",
                )
            )
        for criterion in result.remaining_failures:
            items.append(
                _item("uncertainty", f"Still failing in hypothetical run: {criterion}.", "WhatIfEngine")
            )
        items.extend(_evidence_or_unavailable(result.reasons, "WhatIfEngine"))
        return DecisionTrace(subject=f"what-if:{result.scheme_id}", items=items)

    def explain_pathway(self, result: PathwayResult) -> DecisionTrace:
        """Trace readiness, documents, steps, and next actions."""
        verdicts = {
            "ready": f"Application for scheme '{result.scheme_id}' is documentation-ready.",
            "not_ready": f"Application for scheme '{result.scheme_id}' is not ready.",
            "needs_information": f"Readiness for scheme '{result.scheme_id}' needs information.",
        }
        items = [_item("decision", verdicts[result.readiness], "PathwayEngine")]
        for check in result.documents:
            if check.status == "unknown":
                items.append(
                    _item(
                        "uncertainty",
                        f"Document '{check.document}': possession unknown.",
                        "PathwayEngine",
                    )
                )
            else:
                items.append(
                    _item(
                        "evidence",
                        f"Document '{check.document}': {check.status}.",
                        "PathwayEngine",
                    )
                )
        for step in result.steps:
            items.append(
                _item("evidence", f"Step [{step.step_type}]: {step.title}.", "PathwayEngine")
            )
        for action in result.next_actions:
            items.append(_item("action", action, "PathwayEngine"))
        items.extend(_evidence_or_unavailable(result.reasons, "PathwayEngine"))
        return DecisionTrace(subject=f"pathway:{result.scheme_id}", items=items)
