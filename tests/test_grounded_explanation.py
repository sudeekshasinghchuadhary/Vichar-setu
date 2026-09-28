"""Tests for grounded LLM explanations (fake provider, offline)."""

from typing import Any

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.explanation_engine import ExplanationEngine
from intelligence_engine.explanation_generator import (
    ExplanationGenerator,
    build_prompt,
)
from intelligence_engine.financial_intelligence import FinancialIntelligence
from intelligence_engine.llm_client import LLMClient, LLMExplanationProvider
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.pathway_engine import ApplicationPathwayEngine
from intelligence_engine.schemas import (
    DecisionTrace,
    EligibilityResult,
    ExplanationItem,
    FinancialOption,
    MatchResult,
    NeedAnalysisResult,
    SupportNeed,
    UserProfile,
    WhatIfResult,
    AppliedChange,
)
from intelligence_engine.what_if_engine import WhatIfEngine
from tests.fixtures import make_scheme


class EchoProvider(LLMExplanationProvider):
    """Fake that restates trace facts faithfully (numbers + uncertainty kept)."""

    def generate_explanation(self, prompt: str) -> str:
        """Render the trace facts without echoing the instruction block."""
        facts = prompt.split("\n\nInstructions:")[0]
        return facts + "\nRendered faithfully with unknown details missing as stated."


class BadProvider(LLMExplanationProvider):
    """Fake returning a fixed bad draft."""

    def __init__(self, draft: Any) -> None:
        """Store the draft to return."""
        self.draft = draft

    def generate_explanation(self, prompt: str) -> str:
        """Return the bad draft regardless of prompt."""
        return self.draft


def _profile(**overrides) -> UserProfile:
    """Eligibility profile with overrides."""
    data = {
        "age": 25,
        "annual_family_income": 180000.0,
        "occupation": "Farmer",
        "education_level": "Graduate",
        "social_category": "OBC",
        "state": "Uttar Pradesh",
    }
    data.update(overrides)
    return UserProfile(**data)


def _traces() -> dict[str, DecisionTrace]:
    """One real trace per layer for coverage."""
    explainer = ExplanationEngine()
    scheme = make_scheme()
    eligible = EligibilityEngine().evaluate(_profile(), scheme)
    failing = _profile(annual_family_income=320000.0)
    not_eligible = EligibilityEngine().evaluate(failing, scheme)
    unknown = EligibilityEngine().evaluate(_profile(occupation=None), scheme)
    near = NearMissEngine().analyze(failing, scheme, not_eligible)
    what_if = WhatIfEngine().simulate(failing, scheme, {"annual_family_income": 280000.0})
    gap = FinancialIntelligence().coverage(800000, [FinancialOption(label="s", amount=500000)])
    pathway = ApplicationPathwayEngine().plan(scheme, {"Aadhaar": True})
    needs = NeedAnalysisResult(
        business_goal="Expand tailoring",
        needs=[SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
    )
    return {
        "eligible": explainer.explain_eligibility(eligible),
        "not_eligible": explainer.explain_eligibility(not_eligible),
        "needs_information": explainer.explain_eligibility(unknown),
        "matching": explainer.explain_matching(
            MatchResult(scheme_id="s", score=85.0, rank=1, reasons=["Purpose matches."])
        ),
        "near_miss": explainer.explain_near_miss(near),
        "support": explainer.explain_needs(needs),
        "financial": explainer.explain_financial(gap),
        "what_if": explainer.explain_what_if(what_if),
        "pathway": explainer.explain_pathway(pathway),
    }


def test_all_layers_grounded_with_faithful_provider() -> None:
    """Every layer's trace renders grounded when facts are preserved."""
    generator = ExplanationGenerator(EchoProvider())
    for name, trace in _traces().items():
        result = generator.generate(trace)
        assert result.grounded is True, name
        assert result.fallback_used is False
        assert result.subject == trace.subject


def test_exact_numeric_preservation() -> None:
    """Dropping a trace number (320000) fails the numbers check."""
    trace = _traces()["not_eligible"]
    assert any("320000" in item.text for item in trace.items)
    result = ExplanationGenerator(BadProvider("Not eligible. Some income too high, details unknown.")).generate(trace)
    assert result.grounded is False
    assert "numbers_preserved" in result.issues
    assert result.fallback_used is True


def test_uncertainty_preservation() -> None:
    """Stripping unknown-markers from a needs_information trace fails."""
    trace = _traces()["needs_information"]
    result = ExplanationGenerator(BadProvider("Eligibility cannot be determined. Income 180000.")).generate(trace)
    assert result.grounded is False
    assert "uncertainty_preserved" in result.issues


def test_unsupported_claim_detection() -> None:
    """Guarantee language triggers fallback even with numbers intact."""
    trace = _traces()["eligible"]
    numbers = " ".join(
        item.text for item in trace.items if any(char.isdigit() for char in item.text)
    )
    draft = f"Scheme is eligible. {numbers} Approval is guaranteed and definitely approved."
    result = ExplanationGenerator(BadProvider(draft)).generate(trace)
    assert result.grounded is False
    assert "no_guarantees" in result.issues


def test_verdict_flip_detection() -> None:
    """Upgrading not_eligible to eligible in prose is rejected."""
    trace = _traces()["not_eligible"]
    result = ExplanationGenerator(
        BadProvider("Scheme is not eligible per rules, but you are eligible, income 320000, unknown docs.")
    ).generate(trace)
    assert result.grounded is False
    assert "verdict_not_flipped" in result.issues


def test_malformed_and_empty_output_fallback() -> None:
    """Non-string and empty drafts both fall back safely."""
    trace = _traces()["matching"]
    malformed = ExplanationGenerator(BadProvider(42)).generate(trace)
    assert malformed.grounded is False and malformed.fallback_used is True
    empty = ExplanationGenerator(BadProvider("   ")).generate(trace)
    assert empty.grounded is False and empty.fallback_used is True
    assert "Subject: matching:s" in empty.text


def test_safe_fallback_uses_deterministic_wording() -> None:
    """Fallback text restates the trace decision verbatim."""
    trace = _traces()["financial"]
    result = ExplanationGenerator(BadProvider("guaranteed riches")).generate(trace)
    assert result.fallback_used is True
    assert "A financial gap remains" in result.text


def test_provider_error_falls_back() -> None:
    """A raising provider yields a marked fallback, never an exception."""

    class ExplodingProvider(LLMExplanationProvider):
        def generate_explanation(self, prompt: str) -> str:
            raise RuntimeError("boom")

    result = ExplanationGenerator(ExplodingProvider()).generate(_traces()["eligible"])
    assert result.grounded is False
    assert any("provider_error" in issue for issue in result.issues)


def test_trace_unchanged_and_prompt_grounded() -> None:
    """Trace untouched; prompt carries trace facts and grounding rules only."""
    trace = _traces()["what_if"]
    before = trace.model_dump()
    prompt = build_prompt(trace)
    ExplanationGenerator(EchoProvider()).generate(trace)
    assert trace.model_dump() == before
    assert "320000" in prompt and "280000" in prompt
    assert "Use ONLY the facts" in prompt


def test_provider_abstraction_without_touching_profile_contract() -> None:
    """New ABC stands apart; LLMClient keeps its single method."""
    assert issubclass(EchoProvider, LLMExplanationProvider)
    assert not issubclass(EchoProvider, LLMClient)
    assert set(LLMClient.__abstractmethods__) == {"extract_profile_data"}


def test_no_eligibility_logic_in_generation_layer() -> None:
    """Generator module holds no decision, scoring, or service machinery."""
    import intelligence_engine.explanation_generator as generator_module

    for forbidden in (
        "EligibilityEngine", "MatchingEngine", "NearMissEngine", "NeedAnalyzer",
        "SupportPlanningEngine", "FinancialIntelligence", "WhatIfEngine",
        "PathwayEngine", "LLMClient", "sqlite", "postgres", "requests", "fastapi",
    ):
        assert forbidden not in dir(generator_module)


def test_what_if_objects_flow_through() -> None:
    """What-if comparison structures survive trace generation intact."""
    result = WhatIfEngine().simulate(
        _profile(annual_family_income=320000.0), make_scheme(), {"annual_family_income": 280000.0}
    )
    assert isinstance(result.changes[0], AppliedChange)
    assert isinstance(_traces()["what_if"], DecisionTrace)
