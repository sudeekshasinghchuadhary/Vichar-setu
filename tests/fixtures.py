"""Shared test fixtures for intelligence-engine tests (schemas only, no logic)."""

from intelligence_engine.schemas import (
    EligibilityResult,
    MatchResult,
    PipelineResult,
    Scheme,
    UserProfile,
)


def make_user_profile() -> UserProfile:
    """Return a valid UserProfile with representative fields set."""
    return UserProfile(
        age=28,
        gender="female",
        social_category="OBC",
        state="Uttar Pradesh",
        district="Lucknow",
        occupation="tailor",
        annual_family_income=180000.0,
        purpose="business expansion",
        project_type="micro-enterprise",
        estimated_project_cost=200000.0,
        education_level="secondary",
    )


def make_scheme() -> Scheme:
    """Return a valid prototype Scheme mapped from DB-style fields."""
    return Scheme(
        id="scheme-001",
        name="Prototype Micro-Enterprise Support",
        description="Prototype scheme for testing only.",
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
        source="prototype",
        ministry="Prototype Ministry",
        state="Uttar Pradesh",
        category="micro-enterprise",
        benefit="Prototype benefit description.",
        documents_required=["Aadhaar", "Income certificate"],
        application_link="https://example.com/apply",
    )


def make_eligibility_result() -> EligibilityResult:
    """Return a valid EligibilityResult."""
    return EligibilityResult(
        scheme_id="scheme-001",
        status="eligible",
        reasons=["Age 28 meets min_age 18."],
    )


def make_match_result() -> MatchResult:
    """Return a valid MatchResult."""
    return MatchResult(
        scheme_id="scheme-001",
        score=85.0,
        rank=1,
        reasons=["Purpose matches supported_purposes."],
    )


def make_pipeline_result() -> PipelineResult:
    """Return a valid PipelineResult combining the other models."""
    return PipelineResult(
        profile=make_user_profile(),
        eligibility_results=[make_eligibility_result()],
        ranked_matches=[make_match_result()],
    )
