"""Tests for profile processing (mocked LLM client, no network/key)."""

from typing import Any

import pytest

from intelligence_engine.llm_client import LLMClient
from intelligence_engine.profile_processor import (
    ProfileProcessor,
    ProfileProcessingError,
)
from intelligence_engine.schemas import UserProfile
from tests.fixtures import make_user_profile


class FakeLLMClient(LLMClient):
    """Fake LLM client returning canned extraction results for tests."""

    def __init__(self, payload: Any) -> None:
        """Store the canned payload to return."""
        self.payload = payload

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return the canned payload regardless of input."""
        return self.payload


def test_valid_user_profile_can_be_created() -> None:
    """A valid UserProfile can be created."""
    profile = make_user_profile()
    assert profile.age == 28
    assert profile.annual_family_income == 180000.0


def test_missing_optional_profile_info_is_none() -> None:
    """Missing optional profile information is represented as None."""
    profile = UserProfile()
    assert profile.age is None
    assert profile.gender is None
    assert profile.social_category is None
    assert profile.state is None
    assert profile.district is None
    assert profile.occupation is None
    assert profile.annual_family_income is None
    assert profile.purpose is None
    assert profile.project_type is None
    assert profile.estimated_project_cost is None
    assert profile.education_level is None


def test_natural_language_profile_converted_via_mock_llm() -> None:
    """Natural-language input converts to UserProfile via the mocked LLM."""
    client = FakeLLMClient({"age": 28, "occupation": "tailor", "state": "Uttar Pradesh"})
    profile = ProfileProcessor(client).process_profile("I am 28, a tailor from Uttar Pradesh.")
    assert isinstance(profile, UserProfile)
    assert profile.age == 28
    assert profile.occupation == "tailor"


def test_missing_information_remains_none() -> None:
    """Fields the LLM did not extract remain None (no invention)."""
    client = FakeLLMClient({"age": 30})
    profile = ProfileProcessor(client).process_profile("I am 30.")
    assert profile.age == 30
    assert profile.occupation is None
    assert profile.state is None
    assert profile.annual_family_income is None


def test_normalization_of_obvious_equivalents() -> None:
    """Normalization maps UP -> Uttar Pradesh and Scheduled Caste -> SC."""
    client = FakeLLMClient({"state": "UP", "social_category": "Scheduled Caste"})
    profile = ProfileProcessor(client).process_profile("from UP, scheduled caste")
    assert profile.state == "Uttar Pradesh"
    assert profile.social_category == "SC"
    client2 = FakeLLMClient({"state": "  uttar pradesh  ", "social_category": "sc"})
    profile2 = ProfileProcessor(client2).process_profile("from uttar pradesh, sc")
    assert profile2.state == "Uttar Pradesh"
    assert profile2.social_category == "SC"


def test_invalid_llm_output_rejected_cleanly() -> None:
    """Malformed LLM output and invalid values raise without fabricated defaults."""
    with pytest.raises(ProfileProcessingError):
        ProfileProcessor(FakeLLMClient("not-a-dict")).process_profile("hello")  # type: ignore[arg-type]
    with pytest.raises(ProfileProcessingError):
        ProfileProcessor(FakeLLMClient(["age", 28])).process_profile("hello")  # type: ignore[arg-type]
    with pytest.raises(ProfileProcessingError):
        ProfileProcessor(FakeLLMClient({"age": "twenty-eight"})).process_profile("hello")


def test_processor_makes_no_eligibility_decisions() -> None:
    """ProfileProcessor exposes no eligibility/score/recommendation behavior."""
    processor = ProfileProcessor(FakeLLMClient({}))
    for attr in ("eligible", "eligibility", "score", "recommend", "match", "scheme"):
        assert not hasattr(processor, attr)
        assert attr not in dir(processor)
