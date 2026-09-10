"""Data-contract regression tests (Step 16A): API-boundary safety.

Covers Pydantic serialization of public results, internal-only
RuleOutcome behavior, Scheme state/category authority, and the
estimated_project_cost vs total_requested distinction.
"""

from dataclasses import is_dataclass

from pydantic import BaseModel

from intelligence_engine.eligibility_engine import (
    RuleOutcome,
    evaluate_rule_outcomes,
)
from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.llm_client import NeedExtractor
from intelligence_engine.schemas import (
    CoverageResult,
    DecisionTrace,
    EligibilityResult,
    ExplanationItem,
    FailedCriterion,
    GeneratedExplanation,
    MatchResult,
    NearMissResult,
    NeedAnalysisResult,
    PathwayResult,
    PipelineResult,
    Scheme,
    SupportNeed,
    SupportPlan,
    UserProfile,
    WhatIfResult,
)
from intelligence_engine.semantic_matcher import SemanticScore
from tests.fixtures import make_scheme


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


def test_public_results_serialize_cleanly() -> None:
    """Every API-boundary result round-trips through Pydantic JSON."""
    samples = [
        EligibilityResult(scheme_id="s", status="eligible", reasons=["ok"]),
        MatchResult(scheme_id="s", score=50.0, rank=1),
        PipelineResult(profile=_profile()),
        NearMissResult(
            scheme_id="s",
            is_near_miss=True,
            failed_criteria=[FailedCriterion(criterion="age", user_value=17, required=">= 18", difference=1.0)],
            satisfied_criteria=["state"],
            total_criteria=2,
        ),
        NeedAnalysisResult(
            business_goal="goal",
            needs=[SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
            total_requested=500000,
        ),
        SupportPlan(business_goal="goal"),
        CoverageResult(required=800000, covered=500000, uncovered=300000, own_contribution=300000, coverage_known=True),
        PathwayResult(scheme_id="s", readiness="ready"),
        DecisionTrace(
            subject="eligibility:s",
            items=[ExplanationItem(kind="decision", text="eligible", source="EligibilityEngine")],
        ),
        GeneratedExplanation(subject="s", text="words", grounded=True),
        SemanticScore(score=80.0, cosine=0.8, user_text="a", scheme_text="b"),
        WhatIfResult(
            scheme_id="s",
            current=EligibilityResult(scheme_id="s", status="not_eligible", reasons=["r"]),
            hypothetical=EligibilityResult(scheme_id="s", status="eligible", reasons=["r"]),
            hypothetical_profile=_profile(),
            status_changed=True,
        ),
        make_scheme(),
        _profile(),
    ]
    for sample in samples:
        assert isinstance(sample, BaseModel), type(sample)
        assert type(sample).model_validate_json(sample.model_dump_json()) == sample


def test_rule_outcome_stays_internal() -> None:
    """RuleOutcome remains a plain dataclass (not a public API model)."""
    assert is_dataclass(RuleOutcome)
    assert not (isinstance(RuleOutcome, type) and issubclass(RuleOutcome, BaseModel))
    outcomes = evaluate_rule_outcomes(_profile(), make_scheme().eligibility_rules)
    assert all(isinstance(outcome, RuleOutcome) for outcome in outcomes)
    assert {outcome.rule for outcome in outcomes} >= {"min_age", "max_annual_income"}


def test_rule_metadata_never_overrides_rules() -> None:
    """Conflicting Scheme.state/category lose to eligibility_rules both ways."""
    engine = EligibilityEngine()
    base = make_scheme().model_dump()
    # Display says Bihar, rules say UP, user in UP -> eligible (rules win).
    assert engine.evaluate(
        _profile(), Scheme(**{**base, "state": "Bihar"})
    ).status == "eligible"
    # Display says UP, rules say Bihar, user in UP -> not eligible (rules win).
    rules = dict(base["eligibility_rules"], states=["Bihar"])
    assert engine.evaluate(
        _profile(), Scheme(**{**base, "state": "Uttar Pradesh", "eligibility_rules": rules})
    ).status == "not_eligible"
    # Same both ways for category/categories with an OBC user.
    assert engine.evaluate(
        _profile(), Scheme(**{**base, "category": "General"})
    ).status == "eligible"
    cat_rules = dict(base["eligibility_rules"], categories=["General"])
    assert engine.evaluate(
        _profile(), Scheme(**{**base, "category": "OBC", "eligibility_rules": cat_rules})
    ).status == "not_eligible"


class _FakeNeeds(NeedExtractor):
    """Canned need data regardless of input text."""

    def extract_need_data(self, user_text: str) -> dict:
        """Return fixed needs."""
        return {
            "business_goal": "Expand tailoring",
            "needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}],
        }


def test_cost_estimate_distinct_from_analyzed_total() -> None:
    """Need analysis totals needs only; profile estimate never leaks in."""
    profile = _profile()
    profile.estimated_project_cost = 999999.0
    result = NeedAnalyzer(_FakeNeeds()).analyze("need 5 lakh for machines")
    assert result.total_requested == 500000
    assert result.total_requested != profile.estimated_project_cost
    assert profile.estimated_project_cost == 999999.0
