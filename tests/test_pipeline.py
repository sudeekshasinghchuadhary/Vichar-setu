"""Tests for the pipeline orchestrator (mocked LLM, offline)."""

from typing import Any

import pytest

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.llm_client import LLMClient
from intelligence_engine.matching_engine import MatchingEngine
from intelligence_engine.pipeline import IntelligencePipeline
from intelligence_engine.profile_processor import ProfileProcessor, ProfileProcessingError
from intelligence_engine.schemas import (
    EligibilityResult,
    MatchResult,
    PipelineResult,
    UserProfile,
)
from tests.fixtures import make_pipeline_result, make_scheme


class FakeLLMClient(LLMClient):
    """Fake LLM client returning a canned extraction dict."""

    def __init__(self, payload: Any) -> None:
        """Store the canned payload."""
        self.payload = payload

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return the canned payload regardless of input."""
        return self.payload


SATISFYING_PAYLOAD: dict[str, Any] = {
    "age": 28,
    "occupation": "Farmer",
    "education_level": "Graduate",
    "social_category": "OBC",
    "state": "Uttar Pradesh",
    "annual_family_income": 180000.0,
    "purpose": "business expansion",
    "project_type": "micro-enterprise",
}


def _pipeline(payload: Any) -> IntelligencePipeline:
    """Build a pipeline wired to a fake LLM (real engines)."""
    return IntelligencePipeline(
        profile_processor=ProfileProcessor(FakeLLMClient(payload)),
        eligibility_engine=EligibilityEngine(),
        matching_engine=MatchingEngine(),
    )


def test_pipeline_result_combines_models() -> None:
    """PipelineResult can combine profile, eligibility, and matches."""
    result = make_pipeline_result()
    assert result.profile.age == 28
    assert len(result.eligibility_results) == 1
    assert len(result.ranked_matches) == 1
    assert result.ranked_matches[0].rank == 1


def test_valid_input_flows_end_to_end() -> None:
    """NL input flows Processor -> Eligibility -> Matching with all outputs."""
    result = _pipeline(dict(SATISFYING_PAYLOAD)).run("I am a 28-year-old Farmer.", [make_scheme()])
    assert isinstance(result, PipelineResult)
    assert result.profile.age == 28
    assert result.eligibility_results[0].status == "eligible"
    assert len(result.ranked_matches) == 1
    assert result.ranked_matches[0].scheme_id == "scheme-001"


def test_pipeline_preserves_validated_profile() -> None:
    """Returned profile is the validated, normalized UserProfile."""
    result = _pipeline(dict(SATISFYING_PAYLOAD)).run("hello", [make_scheme()])
    assert isinstance(result.profile, UserProfile)
    assert result.profile.occupation == "Farmer"
    assert result.profile.state == "Uttar Pradesh"


def test_eligibility_results_preserved_verbatim() -> None:
    """Every scheme keeps its own three-state eligibility outcome."""
    strict = make_scheme().model_copy(
        update={"id": "scheme-002", "name": "Strict", "eligibility_rules": {"max_annual_income": 100000}}
    )
    result = _pipeline(dict(SATISFYING_PAYLOAD)).run("hello", [make_scheme(), strict])
    by_id = {item.scheme_id: item.status for item in result.eligibility_results}
    assert by_id == {"scheme-001": "eligible", "scheme-002": "not_eligible"}


def test_ranked_matches_contain_only_eligible() -> None:
    """not_eligible schemes are excluded from ranked matches."""
    strict = make_scheme().model_copy(
        update={"id": "scheme-002", "name": "Strict", "eligibility_rules": {"max_annual_income": 100000}}
    )
    result = _pipeline(dict(SATISFYING_PAYLOAD)).run("hello", [make_scheme(), strict])
    assert [match.scheme_id for match in result.ranked_matches] == ["scheme-001"]


def test_needs_information_not_presented_as_match() -> None:
    """Unresolved schemes stay out of ranked_matches (kept in eligibility)."""
    payload = {key: value for key, value in SATISFYING_PAYLOAD.items() if key != "age"}
    result = _pipeline(payload).run("hello", [make_scheme()])
    assert result.eligibility_results[0].status == "needs_information"
    assert result.ranked_matches == []


def test_processor_validation_errors_propagate() -> None:
    """Malformed LLM output surfaces as ProfileProcessingError, not valid data."""
    with pytest.raises(ProfileProcessingError):
        _pipeline("not-a-dict").run("hello", [make_scheme()])


def test_dependency_injection_with_fake_components() -> None:
    """Pipeline works with stub components sharing the same method shapes."""
    profile = UserProfile(age=30)

    class StubProcessor:
        def process_profile(self, user_text: str) -> UserProfile:
            assert user_text == "hello"
            return profile

    class StubEligibility:
        def evaluate_many(self, profile: UserProfile, schemes: list) -> list[EligibilityResult]:
            return [EligibilityResult(scheme_id=scheme.id, status="eligible", reasons=["stub"]) for scheme in schemes]

    class StubMatching:
        def rank_schemes(self, profile: UserProfile, schemes: list, eligibility_results: list) -> list[MatchResult]:
            assert len(eligibility_results) == len(schemes)
            return [MatchResult(scheme_id=schemes[0].id, score=50.0, rank=1, reasons=["stub"])]

    result = IntelligencePipeline(StubProcessor(), StubEligibility(), StubMatching()).run("hello", [make_scheme()])
    assert result.profile is profile
    assert result.ranked_matches[0].score == 50.0


def test_pipeline_does_not_call_llm_directly() -> None:
    """Pipeline talks to ProfileProcessor only; no LLM attribute or import."""
    import intelligence_engine.pipeline as pipeline_module

    pipeline = _pipeline(dict(SATISFYING_PAYLOAD))
    assert not hasattr(pipeline, "llm_client")
    assert "LLMClient" not in dir(pipeline_module)
    assert "llm_client" not in dir(pipeline_module)


def test_pipeline_has_no_database_or_api_access() -> None:
    """Pipeline module depends on engines/schemas only."""
    import intelligence_engine.pipeline as pipeline_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(pipeline_module)


def test_pipeline_is_deterministic() -> None:
    """Same inputs produce identical pipeline outputs."""
    first = _pipeline(dict(SATISFYING_PAYLOAD)).run("hello", [make_scheme()]).model_dump()
    second = _pipeline(dict(SATISFYING_PAYLOAD)).run("hello", [make_scheme()]).model_dump()
    assert first == second
