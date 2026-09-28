"""Tests for eligibility_engine contracts (schemas only, no logic yet)."""

import pytest
from pydantic import ValidationError

from intelligence_engine.eligibility_engine import (
    EligibilityEngine,
    EligibilityEngineError,
)
from intelligence_engine.schemas import EligibilityResult, Scheme, UserProfile
from tests.fixtures import make_eligibility_result, make_scheme, make_user_profile


def test_valid_scheme_can_be_created() -> None:
    """A valid Scheme can be created."""
    scheme = make_scheme()
    assert scheme.id == "scheme-001"
    assert isinstance(scheme.eligibility_rules, dict)


def test_eligibility_result_accepts_all_three_valid_states() -> None:
    """EligibilityResult accepts all three explicit status values."""
    eligible = EligibilityResult(
        scheme_id="scheme-001",
        status="eligible",
        reasons=["Age 28 meets min_age 18."],
    )
    not_eligible = EligibilityResult(
        scheme_id="scheme-002",
        status="not_eligible",
        reasons=["Income exceeds the maximum allowed."],
    )
    needs_information = EligibilityResult(
        scheme_id="scheme-003",
        status="needs_information",
        reasons=["Disability status is required to evaluate this scheme."],
        missing_information=["disability_status"],
    )

    assert eligible.status == "eligible"
    assert not_eligible.status == "not_eligible"
    assert needs_information.status == "needs_information"
    assert needs_information.missing_information == ["disability_status"]


def test_invalid_eligibility_status_values_are_rejected() -> None:
    """Only the three explicit states are valid."""
    with pytest.raises(ValidationError):
        EligibilityResult(scheme_id="scheme-001", status="maybe", reasons=["bad"])


def test_needs_information_can_carry_missing_information() -> None:
    """needs_information includes missing required fields without inventing values."""
    result = EligibilityResult(
        scheme_id="scheme-001",
        status="needs_information",
        reasons=["Disability status is required to evaluate this scheme."],
        missing_information=["disability_status", "annual_family_income"],
    )
    assert result.status == "needs_information"
    assert result.missing_information == ["disability_status", "annual_family_income"]


def test_eligible_and_not_eligible_results_still_carry_reasons() -> None:
    """Eligible and not_eligible results continue to include human-readable reasons."""
    eligible = make_eligibility_result()
    not_eligible = EligibilityResult(
        scheme_id="scheme-002",
        status="not_eligible",
        reasons=["Age is below the minimum requirement."],
    )

    assert eligible.status == "eligible"
    assert len(eligible.reasons) == 1
    assert not_eligible.status == "not_eligible"
    assert len(not_eligible.reasons) == 1


def test_status_distinguishes_not_eligible_from_needs_information() -> None:
    """The contract distinguishes a known rejection from insufficient information."""
    not_eligible = EligibilityResult(
        scheme_id="scheme-002",
        status="not_eligible",
        reasons=["Income exceeds the maximum allowed."],
    )
    needs_information = EligibilityResult(
        scheme_id="scheme-003",
        status="needs_information",
        reasons=["Disability status is required to evaluate this scheme."],
        missing_information=["disability_status"],
    )

    assert not_eligible.status == "not_eligible"
    assert not_eligible.missing_information == []
    assert needs_information.status == "needs_information"
    assert needs_information.missing_information == ["disability_status"]


def test_scheme_can_represent_eligibility_rules() -> None:
    """Scheme can represent DB-style eligibility info (age, income, etc.)."""
    scheme = make_scheme()
    rules = scheme.eligibility_rules
    assert rules["min_age"] == 18
    assert rules["max_annual_income"] == 300000
    assert "Farmer" in rules["occupations"]
    assert rules["min_education"] == "12th"
    assert "OBC" in rules["categories"]
    assert "Uttar Pradesh" in rules["states"]


def test_scheme_import_not_shadowed() -> None:
    """Keep Scheme import used so contract stays explicit."""
    assert Scheme is not None


def _satisfied_profile() -> UserProfile:
    """Profile satisfying every rule in make_scheme()."""
    return UserProfile(
        age=28,
        annual_family_income=180000.0,
        occupation="Farmer",
        education_level="Graduate",
        social_category="OBC",
        state="Uttar Pradesh",
    )


def test_fully_satisfied_criteria_is_eligible() -> None:
    """Fully satisfied mandatory criteria -> eligible."""
    result = EligibilityEngine().evaluate(_satisfied_profile(), make_scheme())
    assert result.status == "eligible"
    assert result.missing_information == []
    assert len(result.reasons) >= 1


def test_known_age_failure_is_not_eligible() -> None:
    """Age below min_age -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"age": 17})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"
    assert any("minimum age" in reason for reason in result.reasons)


def test_known_income_failure_is_not_eligible() -> None:
    """Income above max_annual_income -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"annual_family_income": 320000.0})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"
    assert any("maximum allowed income" in reason for reason in result.reasons)


def test_known_occupation_failure_is_not_eligible() -> None:
    """Occupation outside occupations -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"occupation": "Student"})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"
    assert any("Occupation" in reason for reason in result.reasons)


def test_known_education_failure_is_not_eligible() -> None:
    """Education below min_education -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"education_level": "Primary"})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"
    assert any("education" in reason.lower() for reason in result.reasons)


def test_known_category_failure_is_not_eligible() -> None:
    """Social category outside categories -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"social_category": "General"})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"


def test_known_state_failure_is_not_eligible() -> None:
    """State outside states -> not_eligible."""
    profile = _satisfied_profile().model_copy(update={"state": "Bihar"})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"


def test_missing_one_required_field_needs_information() -> None:
    """One missing mandatory value -> needs_information, never not_eligible."""
    profile = _satisfied_profile().model_copy(update={"age": None})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "needs_information"
    assert result.missing_information == ["age"]
    assert any("Age" in reason for reason in result.reasons)


def test_missing_multiple_fields_reports_all() -> None:
    """Several missing values are all collected, each with a reason."""
    profile = _satisfied_profile().model_copy(
        update={"social_category": None, "education_level": None}
    )
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "needs_information"
    assert result.missing_information == ["education_level", "social_category"]
    assert len(result.reasons) == 2


def test_known_failure_plus_missing_field_is_not_eligible() -> None:
    """A known failure wins over missing info; both are still reported."""
    profile = _satisfied_profile().model_copy(update={"age": 17, "occupation": None})
    result = EligibilityEngine().evaluate(profile, make_scheme())
    assert result.status == "not_eligible"
    assert "occupation" in result.missing_information
    assert any("minimum age" in reason for reason in result.reasons)


def test_scheme_without_rule_does_not_require_field() -> None:
    """A dimension the scheme does not define is not a restriction."""
    scheme = Scheme(id="s-no-rules", name="Open scheme")
    profile = UserProfile()
    result = EligibilityEngine().evaluate(profile, scheme)
    assert result.status == "eligible"
    assert result.missing_information == []


def test_reasons_are_deterministic() -> None:
    """Same inputs always produce identical reasons."""
    first = EligibilityEngine().evaluate(_satisfied_profile(), make_scheme())
    second = EligibilityEngine().evaluate(_satisfied_profile(), make_scheme())
    assert first.reasons == second.reasons
    assert first.status == second.status == "eligible"


def test_malformed_rule_values_fail_clearly() -> None:
    """Malformed or unsupported rules raise instead of silent eligibility."""
    engine = EligibilityEngine()
    bad_age = make_scheme().model_copy(update={"eligibility_rules": {"min_age": "eighteen"}})
    with pytest.raises(EligibilityEngineError):
        engine.evaluate(_satisfied_profile(), bad_age)
    bad_list = make_scheme().model_copy(update={"eligibility_rules": {"occupations": "Farmer"}})
    with pytest.raises(EligibilityEngineError):
        engine.evaluate(_satisfied_profile(), bad_list)
    unknown = make_scheme().model_copy(update={"eligibility_rules": {"magic": True}})
    with pytest.raises(EligibilityEngineError):
        engine.evaluate(_satisfied_profile(), unknown)


def test_engine_makes_no_llm_calls() -> None:
    """EligibilityEngine must not depend on any LLM client."""
    import intelligence_engine.eligibility_engine as engine_module

    assert not hasattr(EligibilityEngine, "llm_client")
    assert "LLMClient" not in dir(engine_module)
    assert "llm_client" not in dir(engine_module)
