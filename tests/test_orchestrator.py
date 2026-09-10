"""Integration tests for the IntelligenceOrchestrator (fakes only, offline)."""

from typing import Any

import pytest

from intelligence_engine.eligibility_engine import EligibilityEngineError
from intelligence_engine.explanation_generator import ExplanationGenerator
from intelligence_engine.llm_client import LLMClient, LLMExplanationProvider, NeedExtractor
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.schemas import (
    ApplicationStep,
    ChannelInfo,
    IntelligenceResult,
    Scheme,
    SchemeSupport,
    UserProfile,
)
from intelligence_engine.semantic_matcher import EmbeddingProvider, SemanticMatcher


class FakeLLM(LLMClient):
    """Canned profile extraction."""

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return a fixed eligible profile payload."""
        return {
            "age": 25,
            "annual_family_income": 180000.0,
            "occupation": "Farmer",
            "education_level": "Graduate",
            "social_category": "OBC",
            "state": "Uttar Pradesh",
            "purpose": "business expansion",
            "project_type": "micro-enterprise",
        }


class FakeNeeds(NeedExtractor):
    """Canned need extraction."""

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return a fixed machinery need."""
        return {
            "business_goal": "Expand tailoring business",
            "needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}],
        }


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic keyword-count vectors."""

    VOCABULARY = ("food", "business", "micro", "enterprise", "farm", "loan")

    def embed(self, text: str) -> list[float]:
        """Count vocabulary hits."""
        tokens = text.lower().split()
        return [float(tokens.count(word)) for word in self.VOCABULARY]


class EchoExplanations(LLMExplanationProvider):
    """Faithful echo of trace facts."""

    def generate_explanation(self, prompt: str) -> str:
        """Render facts without instructions."""
        return prompt.split("\n\nInstructions:")[0] + "\nStated with unknown missing."


class BadExplanations(LLMExplanationProvider):
    """Guarantee-laden draft forcing fallback."""

    def generate_explanation(self, prompt: str) -> str:
        """Return an ungrounded draft."""
        return "Guaranteed approval for everyone."


def _support(**overrides) -> SchemeSupport:
    """One-time machinery loan support."""
    data = {
        "support_type": "loan",
        "covers_need_types": ["machinery"],
        "max_amount": 500000,
        "max_amount_period": "one_time",
    }
    data.update(overrides)
    return SchemeSupport(**data)


def _open_scheme() -> Scheme:
    """Eligible scheme with support, docs, steps, link, and channel."""
    return Scheme(
        id="scheme-open",
        name="Open support",
        description="Financial assistance for micro-enterprises expanding business.",
        supported_purposes=["business expansion"],
        supported_project_types=["micro-enterprise"],
        eligibility_rules={
            "min_age": 18,
            "max_annual_income": 300000,
            "occupations": ["Farmer"],
            "min_education": "12th",
            "categories": ["OBC"],
            "states": ["Uttar Pradesh"],
        },
        support_options=[_support()],
        documents_required=["Aadhaar", "Income certificate"],
        application_link="https://example.com/apply",
        application_steps=[ApplicationStep(step_type="submit_application", title="Submit online")],
        application_channels=[ChannelInfo(name="District office", channel_type="offline")],
    )


def _strict_scheme() -> Scheme:
    """Scheme failing only on income (320000 fake? no: 180000 income vs 100000 cap)."""
    return Scheme(
        id="scheme-strict",
        name="Strict support",
        eligibility_rules={"max_annual_income": 100000},
    )


def _orchestrator(**overrides) -> IntelligenceOrchestrator:
    """Orchestrator wired to fakes with overrideable parts."""
    parts: dict[str, Any] = {
        "profile_processor": ProfileProcessor(FakeLLM()),
        "need_analyzer": NeedAnalyzer(FakeNeeds()),
    }
    parts.update(overrides)
    return IntelligenceOrchestrator(**parts)


def test_profile_to_eligibility() -> None:
    """Fake profile flows into per-scheme eligibility verdicts."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.profile.age == 25
    assert {s.scheme_id: s.eligibility.status for s in result.schemes} == {
        "scheme-open": "eligible",
        "scheme-strict": "not_eligible",
    }


def test_profile_to_matching() -> None:
    """Only the eligible scheme enters confirmed ranking."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["scheme-open"]
    assert result.ranked_matches[0].rank == 1


def test_eligibility_to_near_miss() -> None:
    """The strict scheme's single income failure reads as a near miss."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-open"].near_miss is None
    assert by_id["scheme-strict"].near_miss.is_near_miss is True


def test_needs_to_support_planning() -> None:
    """Analyzed needs map to verified supports with best coverage."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.needs.total_requested == 500000
    need_plan = result.support_plan.need_plans[0]
    assert need_plan.best_known_coverage == 500000
    assert need_plan.fully_covered is True


def test_support_to_financial_analysis() -> None:
    """Coverage results reuse eligible supports with exact numbers."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert len(result.coverage_results) == 1
    assert result.coverage_results[0].covered == 500000
    assert result.coverage_results[0].coverage_known is True


def test_pathway_to_decision_trace() -> None:
    """Eligible scheme with pathway data gains pathway + traces."""
    result = _orchestrator().run(
        "tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True}
    )
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-open"].pathway.readiness == "needs_information"
    subjects = [t.subject for t in by_id["scheme-open"].traces]
    assert subjects == [
        "eligibility:scheme-open",
        "matching:scheme-open",
        "pathway:scheme-open",
    ]
    assert by_id["scheme-strict"].pathway is None


def test_trace_to_grounded_explanation() -> None:
    """Every trace gets grounded wording; per-scheme primary attached."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(EchoExplanations()))
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True})
    assert len(result.explanations) == len(
        [t for s in result.schemes for t in s.traces] + result.traces
    )
    assert all(e.grounded for e in result.explanations)
    assert result.schemes[0].explanation is not None


def test_semantic_hybrid_ranking() -> None:
    """Hybrid scores blend deterministic + semantic without new weights."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(FakeEmbeddings()), semantic_weight=0.3
    )
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["scheme-open"]
    assert any("Semantic fit" in reason for reason in result.ranked_matches[0].reasons)


def test_complete_result_and_stages() -> None:
    """Full run populates every section with accurate stage tracking."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(EchoExplanations()))
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True})
    assert isinstance(result, IntelligenceResult)
    assert result.stages_completed == [
        "profile", "needs", "eligibility", "matching", "near_miss",
        "support", "financial", "pathway", "traces", "explanation",
    ]
    assert result.profile is not None and result.support_plan is not None


def test_missing_llm_provider_keeps_deterministic_output() -> None:
    """No generator means traces stay, explanations stay empty."""
    result = _orchestrator().run("tailoring business", [_open_scheme()])
    assert result.schemes[0].traces != []
    assert result.explanations == []
    assert result.schemes[0].explanation is None
    assert "explanation" not in result.stages_completed
    assert "traces" in result.stages_completed


def test_llm_fallback_preserves_traces() -> None:
    """Ungrounded drafts fall back without losing deterministic traces."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(BadExplanations()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert all(e.fallback_used for e in result.explanations)
    assert result.schemes[0].traces != []


def test_malformed_scheme_raises_typed_error() -> None:
    """Bad rule formats surface as EligibilityEngineError, never invented."""
    bad = Scheme(id="bad", name="Bad", eligibility_rules={"min_age": "old"})
    with pytest.raises(EligibilityEngineError):
        _orchestrator().run("tailoring business", [bad])


def test_multiple_schemes_associated() -> None:
    """Each scheme keeps its own eligibility/near-miss/pathway slots."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [s.scheme_id for s in result.schemes] == ["scheme-open", "scheme-strict"]
    assert result.schemes[1].eligibility.status == "not_eligible"


def test_no_eligible_schemes() -> None:
    """Nothing ranked, near-miss still analyzed, plan stays uncertain."""
    result = _orchestrator().run("tailoring business", [_strict_scheme()])
    assert result.ranked_matches == []
    assert result.schemes[0].near_miss.is_near_miss is True
    assert result.schemes[0].pathway is None
    assert result.support_plan.need_plans[0].best_known_coverage is None


def test_needs_information_schemes() -> None:
    """Unknown profile fields keep needs_information out of ranking."""
    from intelligence_engine.profile_processor import ProfileProcessor as PP

    class SparseLLM(FakeLLM):
        def extract_profile_data(self, user_text: str) -> dict[str, Any]:
            """Profile missing occupation."""
            data = super().extract_profile_data(user_text)
            data.pop("occupation")
            return data

    orch = _orchestrator(profile_processor=PP(SparseLLM()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert result.schemes[0].eligibility.status == "needs_information"
    assert result.ranked_matches == []
    assert result.schemes[0].near_miss is None


def test_immutability() -> None:
    """Inputs are byte-identical after the run."""
    schemes = [_open_scheme(), _strict_scheme()]
    before = [s.model_dump() for s in schemes]
    _orchestrator().run("tailoring business", schemes)
    assert [s.model_dump() for s in schemes] == before


def test_deterministic_repeated_execution() -> None:
    """Same inputs always give identical IntelligenceResults."""
    orch = _orchestrator()
    args = ("tailoring business", [_open_scheme(), _strict_scheme()])
    assert orch.run(*args) == orch.run(*args)


def test_dependency_injection() -> None:
    """Injected processor output flows through untouched."""

    class FixedProcessor:
        def __init__(self) -> None:
            """Record calls."""
            self.calls = 0

        def process_profile(self, user_text: str) -> UserProfile:
            """Return a fixed profile."""
            self.calls += 1
            return UserProfile(age=40)

    processor = FixedProcessor()
    result = IntelligenceOrchestrator(profile_processor=processor).run("hi", [])
    assert processor.calls == 1
    assert result.profile.age == 40
    assert result.schemes == []


def test_what_if_remains_separate() -> None:
    """Orchestrator never runs or references What-If machinery."""
    orch = _orchestrator()
    assert not hasattr(orch, "what_if")
    import intelligence_engine.orchestrator as orchestrator_module

    assert "WhatIf" not in dir(orchestrator_module)


def test_no_database_or_api_access() -> None:
    """Orchestrator module has no database or API surface."""
    import intelligence_engine.orchestrator as orchestrator_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(orchestrator_module)
