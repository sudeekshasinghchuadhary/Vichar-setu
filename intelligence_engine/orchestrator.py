"""Intelligence orchestrator: end-to-end flow with zero business rules.

Flow: raw text -> ProfileProcessor + NeedAnalyzer (parallel branches)
-> EligibilityEngine (every scheme) -> MatchingEngine (+ optional
SemanticMatcher hybrid) + NearMissEngine (not_eligible only)
-> SupportPlanner (needs + schemes + eligibility)
-> FinancialIntelligence (per-need eligible supports)
-> PathwayEngine (eligible schemes with pathway data)
-> ExplanationEngine (traces) -> ExplanationGenerator (optional wording)
-> IntelligenceResult.

Every decision, weight, calculation, and reason string comes from the
injected engines. This class only routes their inputs/outputs, assembles
SchemeInsights, and records stages_completed accurately. Component typed
errors propagate unchanged. What-If stays separate and never runs here.
No database, FastAPI, HTTP, parallelism, or persisted state.
"""

from typing import Any

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.explanation_engine import ExplanationEngine
from intelligence_engine.explanation_generator import ExplanationGenerator
from intelligence_engine.financial_intelligence import FinancialIntelligence
from intelligence_engine.matching_engine import MatchingEngine
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.pathway_engine import ApplicationPathwayEngine
from intelligence_engine.schemas import (
    DecisionTrace,
    GeneratedExplanation,
    IntelligenceResult,
    MatchResult,
    NeedAnalysisResult,
    Scheme,
    SchemeInsights,
    UserProfile,
)
from intelligence_engine.semantic_matcher import SemanticMatcher, combine_scores
from intelligence_engine.support_planner import SupportPlanningEngine


def _has_pathway_data(scheme: Scheme) -> bool:
    """True when the scheme defines any structured pathway material."""
    return bool(
        scheme.documents_required
        or scheme.application_steps
        or (scheme.application_link or "").strip()
        or scheme.application_channels
    )


class IntelligenceOrchestrator:
    """Coordinate all intelligence stages into one IntelligenceResult."""

    def __init__(
        self,
        profile_processor: Any,
        need_analyzer: Any = None,
        eligibility_engine: EligibilityEngine | None = None,
        matching_engine: MatchingEngine | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        semantic_weight: float = 0.0,
        near_miss_engine: NearMissEngine | None = None,
        support_planner: SupportPlanningEngine | None = None,
        financial: FinancialIntelligence | None = None,
        pathway_engine: ApplicationPathwayEngine | None = None,
        explanation_engine: ExplanationEngine | None = None,
        explanation_generator: ExplanationGenerator | None = None,
    ) -> None:
        """Store injected components (defaults are real engines, never vendors).

        Args:
            profile_processor: Required (needs an LLM backend; no default).
            semantic_weight: Hybrid weight in 0-1; 0 keeps pure
                deterministic ranking. Positive weight requires a
                semantic_matcher, else misconfiguration raises.
        """
        if semantic_weight and semantic_matcher is None:
            raise ValueError("semantic_weight > 0 requires a semantic_matcher.")
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be within 0-1.")
        self.profile_processor = profile_processor
        self.need_analyzer = need_analyzer
        self.eligibility_engine = eligibility_engine or EligibilityEngine()
        self.matching_engine = matching_engine or MatchingEngine()
        self.semantic_matcher = semantic_matcher
        self.semantic_weight = semantic_weight
        self.near_miss_engine = near_miss_engine or NearMissEngine()
        self.support_planner = support_planner or SupportPlanningEngine()
        self.financial = financial or FinancialIntelligence()
        self.pathway_engine = pathway_engine or ApplicationPathwayEngine()
        self.explanation_engine = explanation_engine or ExplanationEngine()
        self.explanation_generator = explanation_generator

    def run(
        self,
        user_text: str,
        schemes: list[Scheme],
        document_availability: dict[str, Any] | None = None,
    ) -> IntelligenceResult:
        """Run the full flow and assemble the product result.

        Args:
            user_text: Raw natural-language user description.
            schemes: Already-structured candidate schemes.
            document_availability: Optional doc-name -> availability map
                forwarded untouched to pathway planning.

        Returns:
            IntelligenceResult with accurately recorded stages_completed.
        """
        stages: list[str] = []
        profile = self.profile_processor.process_profile(user_text)
        stages.append("profile")

        needs = None
        if self.need_analyzer is not None:
            needs = self.need_analyzer.analyze(user_text)
            stages.append("needs")

        return self._execute(profile, needs, schemes, document_availability, stages)

    def run_from_profile(
        self,
        profile: UserProfile,
        schemes: list[Scheme],
        needs: NeedAnalysisResult | None = None,
        document_availability: dict[str, Any] | None = None,
    ) -> IntelligenceResult:
        """Run the same downstream flow from an already-validated profile.

        Args:
            profile: Validated UserProfile (e.g. built by FastAPI from
                stored/verified data instead of natural language).
            schemes: Already-structured candidate schemes.
            needs: Optional pre-analyzed needs; when omitted, the
                support/financial stages are skipped (the need analyzer
                needs raw text, which this path does not take).
            document_availability: Optional doc-name -> availability map
                forwarded untouched to pathway planning.

        Returns:
            IntelligenceResult with accurately recorded stages_completed
            ("profile" marked complete since a validated profile was
            supplied, not extracted).
        """
        return self._execute(profile, needs, schemes, document_availability, ["profile"])

    def _execute(
        self,
        profile: UserProfile,
        needs: NeedAnalysisResult | None,
        schemes: list[Scheme],
        document_availability: dict[str, Any] | None,
        stages: list[str],
    ) -> IntelligenceResult:
        """Shared downstream flow: eligibility -> result assembly."""
        eligibility = self.eligibility_engine.evaluate_many(profile, schemes)
        stages.append("eligibility")
        by_id = {item.scheme_id: item for item in eligibility}
        eligible = [s for s in schemes if by_id.get(s.id) and by_id[s.id].status == "eligible"]

        ranked = self._rank(profile, schemes, eligibility)
        stages.append("matching")

        near_by_id = {}
        for scheme in schemes:
            result = by_id.get(scheme.id)
            if result is not None and result.status == "not_eligible":
                near_by_id[scheme.id] = self.near_miss_engine.analyze(profile, scheme, result)
        if near_by_id:
            stages.append("near_miss")

        support_plan = None
        if needs is not None:
            support_plan = self.support_planner.plan(profile, needs.needs, schemes, eligibility)
            stages.append("support")

        coverage_results = []
        if needs is not None:
            for need in needs.needs:
                supports = [
                    support
                    for scheme in eligible
                    for support in scheme.support_options
                    if need.need_type in support.covers_need_types
                ]
                coverage_results.append(self.financial.coverage_for_need(need, supports))
            stages.append("financial")

        pathways = {}
        for scheme in eligible:
            if _has_pathway_data(scheme):
                pathways[scheme.id] = self.pathway_engine.plan(scheme, document_availability)
        if pathways:
            stages.append("pathway")

        insights: list[SchemeInsights] = []
        for scheme in schemes:
            result = by_id.get(scheme.id)
            match = next((m for m in ranked if m.scheme_id == scheme.id), None)
            traces = self._scheme_traces(result, match, near_by_id.get(scheme.id), pathways.get(scheme.id))
            insights.append(
                SchemeInsights(
                    scheme_id=scheme.id,
                    eligibility=result,
                    near_miss=near_by_id.get(scheme.id),
                    pathway=pathways.get(scheme.id),
                    traces=traces,
                    explanation=self._explain_first(traces) if self.explanation_generator else None,
                )
            )

        global_traces = self._global_traces(needs, support_plan, coverage_results)
        explanations = (
            [self.explanation_generator.generate(trace) for trace in self._all_traces(insights, global_traces)]
            if self.explanation_generator
            else []
        )
        stages.append("traces")
        if self.explanation_generator:
            stages.append("explanation")

        return IntelligenceResult(
            profile=profile,
            needs=needs,
            schemes=insights,
            ranked_matches=ranked,
            support_plan=support_plan,
            coverage_results=coverage_results,
            traces=global_traces,
            explanations=explanations,
            stages_completed=stages,
        )

    def _rank(self, profile: Any, schemes: list[Scheme], eligibility: list[Any]) -> list[MatchResult]:
        """Deterministic ranking, or hybrid when a semantic matcher is set."""
        if self.semantic_matcher is None or self.semantic_weight == 0:
            return self.matching_engine.rank_schemes(profile, schemes, eligibility)
        eligible_ids = {r.scheme_id for r in eligibility if r.status == "eligible"}
        scored: list[tuple[str, float, list[str]]] = []
        for scheme in schemes:
            if scheme.id not in eligible_ids:
                continue
            det_score, det_reasons = self.matching_engine.score_scheme(profile, scheme)
            sem_score = self.semantic_matcher.score(profile, scheme).score
            hybrid = combine_scores(det_score, sem_score, self.semantic_weight)
            scored.append(
                (
                    scheme.id,
                    hybrid,
                    [*det_reasons, f"Semantic fit {sem_score}/100 (weight {self.semantic_weight})."],
                )
            )
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            MatchResult(scheme_id=scheme_id, score=score, rank=rank, reasons=reasons)
            for rank, (scheme_id, score, reasons) in enumerate(scored, start=1)
        ]

    def _scheme_traces(
        self, eligibility: Any, match: Any, near_miss: Any, pathway: Any
    ) -> list[DecisionTrace]:
        """Build scheme-scoped traces in fixed order."""
        traces: list[DecisionTrace] = []
        if eligibility is not None:
            traces.append(self.explanation_engine.explain_eligibility(eligibility))
        if match is not None:
            traces.append(self.explanation_engine.explain_matching(match))
        if near_miss is not None:
            traces.append(self.explanation_engine.explain_near_miss(near_miss))
        if pathway is not None:
            traces.append(self.explanation_engine.explain_pathway(pathway))
        return traces

    def _global_traces(self, needs: Any, support_plan: Any, coverage_results: list[Any]) -> list[DecisionTrace]:
        """Build non-scheme traces in fixed order."""
        traces: list[DecisionTrace] = []
        if needs is not None:
            traces.append(self.explanation_engine.explain_needs(needs))
        if support_plan is not None:
            traces.append(self.explanation_engine.explain_support(support_plan))
        traces.extend(self.explanation_engine.explain_financial(item) for item in coverage_results)
        return traces

    @staticmethod
    def _all_traces(
        insights: list[SchemeInsights], global_traces: list[DecisionTrace]
    ) -> list[DecisionTrace]:
        """Every trace in stable order for optional LLM wording."""
        ordered: list[DecisionTrace] = []
        for insight in insights:
            ordered.extend(insight.traces)
        ordered.extend(global_traces)
        return ordered

    def _explain_first(self, traces: list[DecisionTrace]) -> GeneratedExplanation | None:
        """Word the scheme's primary (eligibility) trace when present."""
        if not traces or self.explanation_generator is None:
            return None
        primary = next((t for t in traces if t.subject.startswith("eligibility:")), traces[0])
        return self.explanation_generator.generate(primary)
