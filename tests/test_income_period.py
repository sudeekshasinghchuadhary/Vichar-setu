"""Focused tests for income_period (annual/monthly/unknown + compat)."""

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.profile_processor import ProfileProcessor, normalize_profile_data
from intelligence_engine.llm_client import LLMClient
from intelligence_engine.schemas import UserProfile
from tests.fixtures import make_scheme


class _FakeLLM(LLMClient):
    """Canned extraction payload."""

    def __init__(self, payload):
        """Store the payload."""
        self.payload = payload

    def extract_profile_data(self, user_text):
        """Return the canned payload."""
        return self.payload


def _profile(**overrides):
    """Eligible-style profile with overrides."""
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


def test_annual_behaves_as_before() -> None:
    """Explicit annual income follows the exact legacy verdicts."""
    engine = EligibilityEngine()
    assert engine.evaluate(_profile(), make_scheme()).status == "eligible"
    over = engine.evaluate(
        _profile(annual_family_income=320000.0), make_scheme()
    )
    assert over.status == "not_eligible"
    assert any("maximum allowed income" in r for r in over.reasons)


def test_monthly_converted_to_annual_in_normalization() -> None:
    """20000 monthly normalizes deterministically to 240000 annual."""
    out = normalize_profile_data({"annual_family_income": 20000, "income_period": "monthly"})
    assert out == {"annual_family_income": 240000, "income_period": "annual"}
    assert normalize_profile_data({"annual_family_income": 20000, "income_period": "monthly"}) == out


def test_monthly_end_to_end_through_processor() -> None:
    """Monthly speech becomes an eligible annual profile without guessing."""
    profile = ProfileProcessor(
        _FakeLLM(
            {
                "age": 25,
                "annual_family_income": 20000,
                "income_period": "monthly",
                "occupation": "Farmer",
                "education_level": "Graduate",
                "social_category": "OBC",
                "state": "Uttar Pradesh",
            }
        )
    ).process_profile("I earn twenty thousand a month")
    assert profile.annual_family_income == 240000
    assert profile.income_period == "annual"
    assert EligibilityEngine().evaluate(profile, make_scheme()).status == "eligible"


def test_unknown_period_yields_needs_information() -> None:
    """Known amount with unknown period cannot satisfy an annual rule."""
    result = EligibilityEngine().evaluate(
        _profile(annual_family_income=280000.0, income_period="unknown"), make_scheme()
    )
    assert result.status == "needs_information"
    assert "annual_family_income" in result.missing_information
    assert any("period" in r for r in result.reasons)


def test_backward_compatibility_default_is_annual() -> None:
    """Profiles without the field behave exactly like legacy annual data."""
    assert UserProfile(age=25).income_period == "annual"
    assert _profile().income_period == "annual"
    engine = EligibilityEngine()
    assert engine.evaluate(_profile(), make_scheme()).status == "eligible"
    assert engine.evaluate(
        _profile(annual_family_income=320000.0), make_scheme()
    ).status == "not_eligible"
