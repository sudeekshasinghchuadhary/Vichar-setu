"""Tests for what-if simulation (deterministic, engine-authoritative, offline)."""

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.schemas import UserProfile, WhatIfChange
from intelligence_engine.what_if_engine import WhatIfEngine, WhatIfError
from tests.fixtures import make_scheme

import pytest


def _income_failing_profile() -> UserProfile:
    """Profile failing only the income rule (320000 > 300000)."""
    return UserProfile(
        age=25,
        annual_family_income=320000.0,
        occupation="Farmer",
        education_level="Graduate",
        social_category="OBC",
        state="Uttar Pradesh",
    )


def test_one_change_can_flip_to_eligible() -> None:
    """'What if income were 2.8 lakh?' turns the verdict with full comparison."""
    result = WhatIfEngine().simulate(
        _income_failing_profile(), make_scheme(), {"annual_family_income": 280000.0}
    )
    assert result.current.status == "not_eligible"
    assert result.hypothetical.status == "eligible"
    assert result.status_changed is True
    assert result.changed_criteria == ["annual_family_income"]
    assert result.remaining_failures == []
    assert result.changes[0].current_value == 320000.0
    assert result.hypothetical_profile.annual_family_income == 280000.0


def test_multiple_changes_applied_together() -> None:
    """Two failing fields fixed at once yield eligibility."""
    profile = _income_failing_profile().model_copy(update={"age": 17})
    result = WhatIfEngine().simulate(
        profile, make_scheme(), {"age": 25, "annual_family_income": 280000.0}
    )
    assert result.hypothetical.status == "eligible"
    assert sorted(result.changed_criteria) == ["age", "annual_family_income"]


def test_remaining_failure_explained() -> None:
    """A still-failing run lists exactly what remains failed."""
    result = WhatIfEngine().simulate(
        _income_failing_profile(), make_scheme(), {"annual_family_income": 310000.0}
    )
    assert result.hypothetical.status == "not_eligible"
    assert result.status_changed is False
    assert result.remaining_failures == ["annual_family_income"]
    assert any("Still failing" in reason for reason in result.reasons)


def test_needs_information_preserved() -> None:
    """Missing required info stays needs_information, never upgraded silently."""
    profile = _income_failing_profile().model_copy(
        update={"annual_family_income": 180000.0, "occupation": None}
    )
    result = WhatIfEngine().simulate(profile, make_scheme(), {"age": 30})
    assert result.current.status == "needs_information"
    assert result.hypothetical.status == "needs_information"
    assert result.status_changed is False


def test_boundary_values_follow_rules() -> None:
    """Exact bounds (income 300000, age 18) satisfy inclusive rules."""
    result = WhatIfEngine().simulate(
        _income_failing_profile(), make_scheme(), {"annual_family_income": 300000.0}
    )
    assert result.hypothetical.status == "eligible"
    young = _income_failing_profile().model_copy(update={"age": 17, "annual_family_income": 180000.0})
    assert WhatIfEngine().simulate(young, make_scheme(), {"age": 18}).hypothetical.status == "eligible"


def test_invalid_changes_rejected() -> None:
    """Negative income and wrong types raise without guessing."""
    with pytest.raises(WhatIfError):
        WhatIfEngine().simulate(_income_failing_profile(), make_scheme(), {"annual_family_income": -5})
    with pytest.raises(WhatIfError):
        WhatIfEngine().simulate(_income_failing_profile(), make_scheme(), {"age": "old"})


def test_unknown_fields_rejected() -> None:
    """Fields outside UserProfile are rejected, never added."""
    with pytest.raises(WhatIfError):
        WhatIfEngine().simulate(_income_failing_profile(), make_scheme(), {"caste_certificate": True})


def test_unchanged_values_report_no_change() -> None:
    """Setting the same value changes nothing and says so structurally."""
    result = WhatIfEngine().simulate(
        _income_failing_profile(), make_scheme(), {"annual_family_income": 320000.0}
    )
    assert result.status_changed is False
    assert result.changed_criteria == []
    assert result.hypothetical.status == "not_eligible"


def test_original_profile_never_mutated() -> None:
    """The real profile is byte-identical after simulation."""
    profile = _income_failing_profile()
    before = profile.model_dump()
    WhatIfEngine().simulate(profile, make_scheme(), {"annual_family_income": 100000.0})
    assert profile.model_dump() == before
    assert profile.annual_family_income == 320000.0


def test_deterministic_repeated_execution() -> None:
    """Same inputs always give identical what-if results."""
    engine = WhatIfEngine()
    args = (_income_failing_profile(), make_scheme(), {"annual_family_income": 280000.0})
    assert engine.simulate(*args) == engine.simulate(*args)


def test_accepts_change_objects_and_current_result() -> None:
    """WhatIfChange list plus a precomputed current result both work."""
    profile = _income_failing_profile()
    current = EligibilityEngine().evaluate(profile, make_scheme())
    result = WhatIfEngine().simulate(
        profile, make_scheme(), [WhatIfChange(field="annual_family_income", hypothetical_value=200000.0)], current
    )
    assert result.current is current
    assert result.hypothetical.status == "eligible"


def test_no_llm_matching_or_semantic_dependency() -> None:
    """What-if module touches only eligibility machinery."""
    import intelligence_engine.what_if_engine as what_if_module

    for forbidden in (
        "LLMClient", "llm_client", "MatchingEngine", "MatchResult",
        "EmbeddingProvider", "cosine", "SemanticMatcher", "NearMissEngine",
    ):
        assert forbidden not in dir(what_if_module)


def test_no_database_or_api_calls() -> None:
    """What-if module has no database or API surface."""
    import intelligence_engine.what_if_engine as what_if_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(what_if_module)
