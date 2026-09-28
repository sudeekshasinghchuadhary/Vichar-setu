"""Tests for the product-level IntelligenceResult container (no logic)."""

from intelligence_engine.schemas import (
    CoverageResult,
    DecisionTrace,
    EligibilityResult,
    ExplanationItem,
    GeneratedExplanation,
    IntelligenceResult,
    MatchResult,
    NearMissResult,
    NeedAnalysisResult,
    PathwayResult,
    PipelineResult,
    SchemeInsights,
    SupportNeed,
    SupportPlan,
    UserProfile,
    WhatIfResult,
)
from tests.fixtures import make_pipeline_result, make_user_profile


def _full_result() -> IntelligenceResult:
    """Fully populated container across two schemes."""
    return IntelligenceResult(
        profile=make_user_profile(),
        needs=NeedAnalysisResult(
            business_goal="Expand tailoring",
            needs=[SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
            total_requested=500000,
        ),
        schemes=[
            SchemeInsights(
                scheme_id="scheme-001",
                eligibility=EligibilityResult(scheme_id="scheme-001", status="eligible", reasons=["ok"]),
                near_miss=NearMissResult(
                    scheme_id="scheme-001", is_near_miss=False, total_criteria=2,
                    satisfied_criteria=["age", "state"],
                ),
                pathway=PathwayResult(scheme_id="scheme-001", readiness="ready"),
                traces=[
                    DecisionTrace(
                        subject="eligibility:scheme-001",
                        items=[ExplanationItem(kind="decision", text="eligible", source="EligibilityEngine")],
                    )
                ],
                explanation=GeneratedExplanation(subject="eligibility:scheme-001", text="words", grounded=True),
            ),
            SchemeInsights(
                scheme_id="scheme-002",
                eligibility=EligibilityResult(scheme_id="scheme-002", status="not_eligible", reasons=["no"]),
            ),
        ],
        ranked_matches=[MatchResult(scheme_id="scheme-001", score=85.0, rank=1, reasons=["fit"])],
        support_plan=SupportPlan(business_goal="Expand tailoring"),
        coverage_results=[CoverageResult(required=500000, covered=300000, uncovered=200000, own_contribution=200000, coverage_known=True)],
        stages_completed=["profile", "needs", "eligibility", "matching", "support"],
    )


def test_complete_result_construction() -> None:
    """All ten product sections coexist with stable scheme association."""
    result = _full_result()
    assert result.profile.age == 28
    assert result.needs.total_requested == 500000
    assert [scheme.scheme_id for scheme in result.schemes] == ["scheme-001", "scheme-002"]
    assert result.schemes[0].eligibility.status == "eligible"
    assert result.schemes[0].traces[0].subject == "eligibility:scheme-001"
    assert result.schemes[0].explanation.grounded is True
    assert result.ranked_matches[0].scheme_id == "scheme-001"
    assert result.coverage_results[0].uncovered == 200000


def test_minimal_result_construction() -> None:
    """An empty container is valid: nothing executed, nothing faked."""
    result = IntelligenceResult()
    assert result.profile is None
    assert result.schemes == []
    assert result.stages_completed == []


def test_serialization_round_trip() -> None:
    """Full container serializes and deserializes losslessly."""
    result = _full_result()
    assert IntelligenceResult.model_validate_json(result.model_dump_json()) == result


def test_pipeline_result_compatibility() -> None:
    """Legacy PipelineResult maps into the container; old contract intact."""
    legacy = make_pipeline_result()
    assert isinstance(legacy, PipelineResult)
    migrated = IntelligenceResult.from_pipeline_result(legacy)
    assert migrated.profile == legacy.profile
    assert [scheme.scheme_id for scheme in migrated.schemes] == ["scheme-001"]
    assert migrated.schemes[0].eligibility == legacy.eligibility_results[0]
    assert migrated.ranked_matches == legacy.ranked_matches
    assert migrated.stages_completed == ["profile", "eligibility", "matching"]
    assert migrated.support_plan is None


def test_optional_explanation_absence() -> None:
    """Missing LLM wording leaves deterministic outputs untouched."""
    result = _full_result()
    result.schemes[0].explanation = None
    assert result.schemes[0].traces != []
    assert result.schemes[0].eligibility.status == "eligible"


def test_empty_optional_collections() -> None:
    """Unexecuted stages read as empty, not as verdicts."""
    insights = SchemeInsights(scheme_id="s")
    assert insights.eligibility is None
    assert insights.traces == []
    assert insights.explanation is None


def test_what_if_stays_separate() -> None:
    """WhatIfResult is not a field of the container."""
    assert "what_if" not in IntelligenceResult.model_fields
    assert "whatif" not in "".join(IntelligenceResult.model_fields).lower()


def test_container_holds_no_business_logic() -> None:
    """Container module surface is data-only (no engines/methods)."""
    assert not hasattr(IntelligenceResult, "evaluate")
    assert not hasattr(IntelligenceResult, "score")
    assert not hasattr(SchemeInsights, "rank")


def test_inputs_stable_and_immutable_shape() -> None:
    """Construction never mutates sub-objects; dumps are stable."""
    profile = make_user_profile()
    before = profile.model_dump()
    result = IntelligenceResult(profile=profile, stages_completed=["profile"])
    assert profile.model_dump() == before
    assert result.model_dump() == IntelligenceResult.model_validate(result.model_dump()).model_dump()
