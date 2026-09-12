"""End-to-end audit tests: realistic tailoring scenario, no hand-built hops.

Scenario: "I run a small tailoring business in Uttar Pradesh. My annual
family income is Rs.2.8 lakh. I need Rs.5 lakh for machinery."
Scheme names are prototype-labeled; no real government claims.
"""

from typing import Any

import pytest

from intelligence_engine.eligibility_engine import EligibilityEngineError
from intelligence_engine.explanation_generator import ExplanationGenerator
from intelligence_engine.llm_client import LLMClient, LLMExplanationProvider, NeedExtractor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.schemas import (
    ApplicationStep,
    ChannelInfo,
    Scheme,
    SchemeSupport,
)
from intelligence_engine.semantic_matcher import EmbeddingProvider, SemanticMatcher

TEXT = (
    "I run a small tailoring business in Uttar Pradesh. "
    "My annual family income is Rs.2.8 lakh. I need Rs.5 lakh for machinery."
)


class TailorLLM(LLMClient):
    """Fixed tailor profile extraction."""

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return the scenario profile."""
        return {
            "age": 32,
            "annual_family_income": 280000.0,
            "occupation": "Tailor",
            "education_level": "Secondary",
            "social_category": "OBC",
            "state": "Uttar Pradesh",
            "purpose": "expand tailoring business",
            "project_type": "micro-enterprise",
        }


class TailorNeeds(NeedExtractor):
    """Fixed machinery need extraction."""

    def __init__(self, extra: list[dict[str, Any]] | None = None) -> None:
        """Optional extra needs."""
        self.extra = extra or []

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return the scenario needs."""
        return {
            "business_goal": "Expand tailoring business",
            "needs": [
                {"need_type": "machinery", "amount": 500000, "amount_period": "one_time"},
                *self.extra,
            ],
        }


class KeywordEmbeddings(EmbeddingProvider):
    """Deterministic keyword vectors."""

    VOCABULARY = ("tailoring", "business", "micro", "enterprise", "machinery", "loan")

    def embed(self, text: str) -> list[float]:
        """Count hits."""
        tokens = text.lower().split()
        return [float(tokens.count(word)) for word in self.VOCABULARY]


class EchoExplanations(LLMExplanationProvider):
    """Faithful echo with captured prompts."""

    def __init__(self) -> None:
        """Record prompts for grounding inspection."""
        self.prompts: list[str] = []

    def generate_explanation(self, prompt: str) -> str:
        """Echo facts; record the prompt."""
        self.prompts.append(prompt)
        return prompt.split("\n\nInstructions:")[0] + "\nStated with unknown missing."


class BadExplanations(LLMExplanationProvider):
    """Ungrounded draft."""

    def generate_explanation(self, prompt: str) -> str:
        """Return a guarantee."""
        return "Guaranteed approval for everyone."


def _support(**overrides) -> SchemeSupport:
    """One-time machinery loan of 5 lakh."""
    data = {
        "support_type": "loan",
        "covers_need_types": ["machinery"],
        "max_amount": 500000,
        "max_amount_period": "one_time",
    }
    data.update(overrides)
    return SchemeSupport(**data)


def _tailoring_support() -> Scheme:
    """Fully matching prototype scheme (eligible)."""
    return Scheme(
        id="proto-tailoring",
        name="Prototype tailoring enterprise support",
        description="Financial assistance for micro-enterprises expanding tailoring business.",
        supported_purposes=["expand tailoring business"],
        supported_project_types=["micro-enterprise"],
        eligibility_rules={
            "min_age": 18,
            "max_age": 60,
            "max_annual_income": 300000,
            "categories": ["OBC"],
            "states": ["Uttar Pradesh"],
        },
        support_options=[_support()],
        documents_required=["Aadhaar", "Income certificate"],
        application_link="https://example.com/apply",
        application_steps=[ApplicationStep(step_type="submit_application", title="Submit online")],
        application_channels=[ChannelInfo(name="District office", channel_type="offline")],
    )


def _strict_income_scheme() -> Scheme:
    """Fails only on income (2.8L > 2.5L): near miss."""
    return Scheme(
        id="proto-strict",
        name="Prototype strict income scheme",
        eligibility_rules={"min_age": 18, "max_annual_income": 250000},
    )


def _orchestrator(**overrides) -> IntelligenceOrchestrator:
    """Orchestrator wired to scenario fakes."""
    parts: dict[str, Any] = {
        "profile_processor": ProfileProcessor(TailorLLM()),
        "need_analyzer": NeedAnalyzer(TailorNeeds()),
    }
    parts.update(overrides)
    return IntelligenceOrchestrator(**parts)


def test_normal_successful_flow() -> None:
    """Profile, needs, eligibility, ranking, support, finance, pathway, traces."""
    result = _orchestrator().run(TEXT, [_tailoring_support(), _strict_income_scheme()], {"Aadhaar": True})
    assert result.profile.occupation == "Tailor"
    assert result.needs.total_requested == 500000
    assert result.ranked_matches[0].scheme_id == "proto-tailoring"
    assert result.support_plan.need_plans[0].fully_covered is True
    assert result.coverage_results[0].covered == 500000
    assert result.schemes[0].pathway.readiness == "needs_information"
    assert result.schemes[0].traces != []


def test_no_eligible_schemes() -> None:
    """Ranking empty, eligibility completed, near-miss analyzed but off."""
    result = _orchestrator().run(TEXT, [_strict_income_scheme()])
    assert result.ranked_matches == []
    assert "eligibility" in result.stages_completed
    assert result.schemes[0].near_miss.is_near_miss is False
    assert result.schemes[0].pathway is None


def test_mixed_eligibility() -> None:
    """Eligible ranked; strict excluded from ranking but near-miss flagged."""
    result = _orchestrator().run(TEXT, [_tailoring_support(), _strict_income_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["proto-tailoring"]
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["proto-strict"].near_miss.failed_criteria[0].criterion == "annual_family_income"
    assert by_id["proto-strict"].near_miss.failed_criteria[0].difference == 30000.0


def test_needs_information_eligibility() -> None:
    """Unknown profile fields keep needs_information out of ranking."""
    from intelligence_engine.profile_processor import ProfileProcessor as PP

    class SparseLLM(TailorLLM):
        def extract_profile_data(self, user_text: str) -> dict[str, Any]:
            """Profile missing state."""
            data = super().extract_profile_data(user_text)
            data.pop("state")
            return data

    scheme = Scheme(id="proto-unknown", name="U", eligibility_rules={"states": ["Uttar Pradesh"]})
    orch = IntelligenceOrchestrator(
        profile_processor=PP(SparseLLM()), need_analyzer=NeedAnalyzer(TailorNeeds())
    )
    result = orch.run(TEXT, [scheme])
    assert result.schemes[0].eligibility.status == "needs_information"
    assert result.ranked_matches == []
    near_miss = result.schemes[0].near_miss
    assert near_miss is not None
    assert near_miss.is_near_miss is False
    assert near_miss.failed_criteria == []
    assert near_miss.total_criteria == 1


def test_near_miss_scheme_end_to_end() -> None:
    """3+ satisfied, 1 failed (income): trace intact, verdict off under conservative-off."""
    result = _orchestrator().run(TEXT, [_strict_income_scheme()])
    near = result.schemes[0].near_miss
    assert near.is_near_miss is False
    assert any("maximum allowed income" in r for r in near.reasons)


def test_multiple_needs_end_to_end() -> None:
    """Two needs plan and finance independently without mixing."""
    orch = _orchestrator(need_analyzer=NeedAnalyzer(TailorNeeds([
        {"need_type": "working_capital", "amount": 50000, "amount_period": "one_time"},
    ])))
    result = orch.run(TEXT, [_tailoring_support()])
    assert len(result.support_plan.need_plans) == 2
    assert len(result.coverage_results) == 2
    assert result.coverage_results[1].coverage_known is False


def test_unknown_support_amount_end_to_end() -> None:
    """Support without a cap maps but never fabricates coverage."""
    scheme = _tailoring_support()
    scheme.support_options = [SchemeSupport(support_type="loan", covers_need_types=["machinery"])]
    result = _orchestrator().run(TEXT, [scheme])
    assert result.support_plan.need_plans[0].best_known_coverage is None
    assert result.coverage_results[0].coverage_known is False


def test_financial_period_mismatch_end_to_end() -> None:
    """Monthly support against a one-time need stays unknown."""
    scheme = _tailoring_support()
    scheme.support_options = [_support(max_amount=20000, max_amount_period="monthly")]
    result = _orchestrator().run(TEXT, [scheme])
    assert result.support_plan.need_plans[0].best_known_coverage is None


class AmountlessNeeds(NeedExtractor):
    """Machinery need without an amount (unknown stays unknown)."""

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return the scenario need minus its amount."""
        return {
            "business_goal": "Expand tailoring business",
            "needs": [{"need_type": "machinery"}],
        }


def test_financial_unknown_asks_then_reruns_known() -> None:
    """Unknown amount asks needs[0].amount; answered rerun covers; verdicts untouched."""
    unknown = _orchestrator(need_analyzer=NeedAnalyzer(AmountlessNeeds())).run(
        TEXT, [_tailoring_support()]
    )
    assert unknown.coverage_results[0].coverage_known is False
    assert unknown.coverage_results[0].covered is None
    assert unknown.clarification is not None
    assert "needs[0].amount" in unknown.clarification.missing_fields
    complete = _orchestrator().run(TEXT, [_tailoring_support()])
    assert [
        (insight.scheme_id, insight.eligibility.status) for insight in unknown.schemes
    ] == [
        (insight.scheme_id, insight.eligibility.status) for insight in complete.schemes
    ]
    assert [m.scheme_id for m in unknown.ranked_matches] == [
        m.scheme_id for m in complete.ranked_matches
    ]
    assert unknown.ranked_matches[0].score == complete.ranked_matches[0].score
    assert complete.coverage_results[0].coverage_known is True
    assert complete.coverage_results[0].covered == 500000


def test_missing_and_unknown_documents() -> None:
    """Explicitly missing blocks; unmentioned stays unknown (never missing)."""
    orch = _orchestrator()
    missing = orch.run(TEXT, [_tailoring_support()], {"Aadhaar": True, "Income certificate": False})
    assert missing.schemes[0].pathway.readiness == "not_ready"
    unknown = orch.run(TEXT, [_tailoring_support()], {"Aadhaar": True})
    checks = {c.document: c.status for c in unknown.schemes[0].pathway.documents}
    assert checks == {"Aadhaar": "available", "Income certificate": "unknown"}


def test_missing_application_link() -> None:
    """No link means no submit action, only a stated gap."""
    scheme = _tailoring_support()
    scheme.application_link = None
    result = _orchestrator().run(TEXT, [scheme], {"Aadhaar": True})
    assert not any("Submit the application" in a for a in result.schemes[0].pathway.next_actions)


def test_explanation_absent_keeps_determinism() -> None:
    """No generator: traces present, explanations empty, stage unmarked."""
    result = _orchestrator().run(TEXT, [_tailoring_support()])
    assert result.schemes[0].traces != []
    assert result.explanations == []
    assert "explanation" not in result.stages_completed
    assert "traces" in result.stages_completed


def test_explanation_fallback_keeps_intelligence() -> None:
    """Bad drafts fall back; deterministic results survive."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(BadExplanations()))
    result = orch.run(TEXT, [_tailoring_support()])
    assert all(e.fallback_used for e in result.explanations)
    assert result.ranked_matches[0].scheme_id == "proto-tailoring"


def test_malformed_scheme_typed_error() -> None:
    """Bad rule values raise; nothing is invented."""
    from intelligence_engine.eligibility_engine import EligibilityEngineError

    bad = Scheme(id="bad", name="Bad", eligibility_rules={"min_age": "eighteen"})
    with pytest.raises(EligibilityEngineError):
        _orchestrator().run(TEXT, [bad])


def test_empty_scheme_list() -> None:
    """No schemes yields a valid empty result with accurate stages."""
    result = _orchestrator().run(TEXT, [])
    assert result.schemes == [] and result.ranked_matches == []
    assert "near_miss" not in result.stages_completed
    assert "pathway" not in result.stages_completed
    assert result.stages_completed[:4] == ["profile", "needs", "eligibility", "matching"]


def test_repeated_execution_identical() -> None:
    """Identical runs produce identical end-to-end results."""
    orch = _orchestrator()
    args = (TEXT, [_tailoring_support(), _strict_income_scheme()])
    assert orch.run(*args) == orch.run(*args)


def test_input_immutability() -> None:
    """Schemes unchanged after the full flow."""
    schemes = [_tailoring_support(), _strict_income_scheme()]
    before = [s.model_dump() for s in schemes]
    _orchestrator().run(TEXT, schemes, {"Aadhaar": True})
    assert [s.model_dump() for s in schemes] == before


def test_scheme_association_integrity() -> None:
    """Every per-scheme output carries its own scheme_id; traces align."""
    provider = EchoExplanations()
    orch = _orchestrator(explanation_generator=ExplanationGenerator(provider))
    result = orch.run(TEXT, [_tailoring_support(), _strict_income_scheme()], {"Aadhaar": True})
    for insight in result.schemes:
        for named in (insight.eligibility, insight.near_miss, insight.pathway):
            if named is not None:
                assert named.scheme_id == insight.scheme_id
        for trace in insight.traces:
            assert insight.scheme_id in trace.subject
        if insight.explanation is not None:
            assert insight.scheme_id in insight.explanation.subject
    for match in result.ranked_matches:
        assert match.scheme_id in {s.scheme_id for s in result.schemes}


def test_stages_reflect_reality() -> None:
    """Skipped stages stay unmarked; executed stages marked."""
    bare = IntelligenceOrchestrator(profile_processor=ProfileProcessor(TailorLLM())).run(
        TEXT, [_tailoring_support()]
    )
    assert bare.stages_completed == ["profile", "eligibility", "matching", "pathway", "traces"]
    assert bare.needs is None and bare.support_plan is None


def test_grounding_prompt_contains_only_trace() -> None:
    """LLM prompts carry trace facts plus rules — no profile internals."""
    provider = EchoExplanations()
    orch = _orchestrator(explanation_generator=ExplanationGenerator(provider))
    orch.run(TEXT, [_tailoring_support()], {"Aadhaar": True})
    assert provider.prompts, "generator must have been called"
    for prompt in provider.prompts:
        assert "Use ONLY the facts" in prompt
        assert "- [decision]" in prompt
    joined = "\n".join(provider.prompts)
    assert "Tailor" not in joined  # profile-only wording never enters prompts


def test_semantic_weight_zero_is_deterministic_only() -> None:
    """Weight 0 ignores semantic signals entirely."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(KeywordEmbeddings()), semantic_weight=0.0
    )
    result = orch.run(TEXT, [_tailoring_support(), _strict_income_scheme()])
    assert not any("Semantic fit" in r for m in result.ranked_matches for r in m.reasons)


def test_semantic_hybrid_combines_scores() -> None:
    """Weight > 0 blends signals; ineligible stays excluded regardless."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(KeywordEmbeddings()), semantic_weight=0.5
    )
    result = orch.run(TEXT, [_tailoring_support(), _strict_income_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["proto-tailoring"]
    assert any("Semantic fit" in r for r in result.ranked_matches[0].reasons)


def test_semantic_misconfiguration_rejected() -> None:
    """Positive weight without a matcher is an explicit error."""
    with pytest.raises(ValueError):
        _orchestrator(semantic_weight=0.5)
