"""Tests for explanation & decision traces (consumer-only, deterministic)."""

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.explanation_engine import ExplanationEngine
from intelligence_engine.financial_intelligence import FinancialIntelligence
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.pathway_engine import ApplicationPathwayEngine
from intelligence_engine.schemas import (
    CoverageResult,
    EligibilityResult,
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


def _engine() -> ExplanationEngine:
    """Fresh engine."""
    return ExplanationEngine()


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


def _decision(trace) -> str:
    """The single decision item's text."""
    decisions = [item.text for item in trace.items if item.kind == "decision"]
    assert len(decisions) == 1
    return decisions[0]


def test_eligible_explanation() -> None:
    """Eligible verdict traced with verbatim evidence."""
    result = EligibilityEngine().evaluate(_profile(), make_scheme())
    trace = _engine().explain_eligibility(result)
    assert _decision(trace) == "Scheme 'scheme-001' is eligible."
    assert any("Age 25 meets" in item.text for item in trace.items if item.kind == "evidence")


def test_not_eligible_explanation() -> None:
    """Rejection traced without softening or new claims."""
    profile = _profile(annual_family_income=320000.0)
    result = EligibilityEngine().evaluate(profile, make_scheme())
    trace = _engine().explain_eligibility(result)
    assert _decision(trace) == "Scheme 'scheme-001' is not eligible."
    assert any("exceeds the maximum" in item.text for item in trace.items)


def test_needs_information_explanation() -> None:
    """Gaps listed as uncertainty plus provide-actions, never rejection."""
    profile = _profile(occupation=None)
    result = EligibilityEngine().evaluate(profile, make_scheme())
    trace = _engine().explain_eligibility(result)
    assert "cannot be determined yet" in _decision(trace)
    assert any(item.kind == "uncertainty" and "occupation" in item.text for item in trace.items)
    assert any(item.kind == "action" and "Provide occupation" in item.text for item in trace.items)
    assert "not eligible" not in _decision(trace).lower().replace("cannot be determined yet", "")


def test_matching_explanation() -> None:
    """Rank/score restated with calculation shown."""
    trace = _engine().explain_matching(
        MatchResult(scheme_id="s", score=85.0, rank=1, reasons=["Purpose matches."])
    )
    assert _decision(trace) == "Scheme 's' ranked #1 with match score 85.0/100."
    assert any(item.kind == "calculation" for item in trace.items)
    assert any("Purpose matches." in item.text for item in trace.items)


def test_near_miss_explanation() -> None:
    """Near-miss verdict with structured failed-criterion evidence."""
    profile = _profile(annual_family_income=320000.0)
    scheme = make_scheme()
    near = NearMissEngine().analyze(profile, scheme, EligibilityEngine().evaluate(profile, scheme))
    trace = _engine().explain_near_miss(near)
    assert _decision(trace) == "Scheme 'scheme-001' is a near miss."
    assert any("annual_family_income" in item.text and "320000" in item.text for item in trace.items)
    assert any(item.kind == "calculation" and "20000" in item.text for item in trace.items)


def test_support_mapping_explanation() -> None:
    """Support plan traced per need with coverage calculations."""
    from intelligence_engine.schemas import SchemeSupport
    from intelligence_engine.support_planner import SupportPlanningEngine

    scheme = make_scheme()
    scheme.support_options = [
        SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=300000)
    ]
    plan = SupportPlanningEngine().plan(
        _profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [scheme],
        [EligibilityResult(scheme_id="scheme-001", status="eligible", reasons=["ok"])],
    )
    trace = _engine().explain_support(plan)
    assert "1 need(s)" in _decision(trace)
    assert any(item.kind == "calculation" and "300000" in item.text for item in trace.items)
    assert any(item.kind == "action" for item in trace.items)


def test_financial_gap_explanation() -> None:
    """Gap arithmetic traced; unknown coverage traced as uncertainty."""
    gap = FinancialIntelligence().coverage(800000, [FinancialOption(label="s", amount=500000)])
    trace = _engine().explain_financial(gap)
    assert "gap remains" in _decision(trace)
    assert any("300000" in item.text for item in trace.items if item.kind == "calculation")
    unknown = FinancialIntelligence().coverage(800000, [FinancialOption(label="s")])
    assert any(
        item.kind == "uncertainty" for item in _engine().explain_financial(unknown).items
    )


def test_what_if_explanation() -> None:
    """Hypothetical change traced with before/after and leftovers."""
    result = WhatIfEngine().simulate(
        _profile(annual_family_income=320000.0), make_scheme(), {"annual_family_income": 280000.0}
    )
    trace = _engine().explain_what_if(result)
    assert "not_eligible -> eligible" in _decision(trace)
    assert any("320000" in item.text and "280000" in item.text for item in trace.items)


def test_pathway_explanation() -> None:
    """Readiness traced with document states and next actions."""
    pathway = ApplicationPathwayEngine().plan(make_scheme(), {"Aadhaar": True})
    trace = _engine().explain_pathway(pathway)
    assert "needs information" in _decision(trace)
    assert any(item.kind == "uncertainty" and "Income certificate" in item.text for item in trace.items)
    assert any(item.kind == "action" for item in trace.items)


def test_unknown_and_empty_upstream_data() -> None:
    """Empty reasons yield explicit unavailable markers, never guesses."""
    trace = _engine().explain_eligibility(
        EligibilityResult(scheme_id="s", status="eligible", reasons=[])
    )
    assert any("unavailable" in item.text for item in trace.items if item.kind == "uncertainty")
    needs = NeedAnalysisResult()
    assert any(
        item.kind == "uncertainty" for item in _engine().explain_needs(needs).items
    )


def test_source_reasons_preserved_verbatim() -> None:
    """Every upstream reason appears word-for-word in the trace."""
    result = EligibilityResult(scheme_id="s", status="not_eligible", reasons=["Custom reason XYZ."])
    trace = _engine().explain_eligibility(result)
    assert any(item.text == "Custom reason XYZ." and item.source == "EligibilityEngine" for item in trace.items)


def test_no_contradictory_explanation() -> None:
    """Decision templates match statuses exactly across all three states."""
    for status, fragment in [
        ("eligible", "is eligible."),
        ("not_eligible", "is not eligible."),
        ("needs_information", "cannot be determined yet."),
    ]:
        trace = _engine().explain_eligibility(
            EligibilityResult(scheme_id="s", status=status, reasons=["r"])
        )
        assert _decision(trace).endswith(fragment)


def test_deterministic_output() -> None:
    """Identical inputs give identical traces."""
    result = EligibilityEngine().evaluate(_profile(), make_scheme())
    assert _engine().explain_eligibility(result) == _engine().explain_eligibility(result)


def test_sources_unchanged() -> None:
    """Upstream result objects are unchanged by explanation."""
    result = EligibilityEngine().evaluate(_profile(), make_scheme())
    before = result.model_dump()
    _engine().explain_eligibility(result)
    assert result.model_dump() == before


def test_what_if_result_untouched() -> None:
    """What-if comparison objects survive explanation intact."""
    result = WhatIfEngine().simulate(
        _profile(annual_family_income=320000.0), make_scheme(), {"annual_family_income": 280000.0}
    )
    before = result.model_dump()
    _engine().explain_what_if(result)
    assert result.model_dump() == before
    assert isinstance(result.changes[0], AppliedChange)


def test_no_llm_or_recomputation() -> None:
    """Explanation module imports no engines, LLMs, or services."""
    import intelligence_engine.explanation_engine as explanation_module

    for forbidden in (
        "LLMClient", "llm_client", "EligibilityEngine", "MatchingEngine",
        "NearMissEngine", "NeedAnalyzer", "SupportPlanningEngine",
        "FinancialIntelligence", "WhatIfEngine", "PathwayEngine",
        "EmbeddingProvider", "cosine", "sqlite", "postgres", "requests", "fastapi",
    ):
        assert forbidden not in dir(explanation_module)
