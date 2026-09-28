"""Tests for near-miss analysis (deterministic, offline, no scores/embeddings)."""

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.schemas import Scheme, UserProfile
from tests.fixtures import make_scheme


def _satisfied_profile() -> UserProfile:
    """Profile satisfying every rule in make_scheme()."""
    return UserProfile(
        age=25,
        annual_family_income=180000.0,
        occupation="Farmer",
        education_level="Graduate",
        social_category="OBC",
        state="Uttar Pradesh",
    )


def test_all_passing_scheme_is_not_near_miss() -> None:
    """An eligible outcome is not a near miss."""
    scheme = make_scheme()
    eligibility = EligibilityEngine().evaluate(_satisfied_profile(), scheme)
    result = NearMissEngine().analyze(_satisfied_profile(), scheme, eligibility)
    assert eligibility.status == "eligible"
    assert result.is_near_miss is False
    assert result.failed_criteria == []
    assert result.total_criteria == 6


def test_single_failure_conservative_off() -> None:
    """CONSERVATIVE-OFF: one failure no longer classifies; evidence preserved."""
    profile = _satisfied_profile().model_copy(update={"annual_family_income": 320000.0})
    scheme = make_scheme()
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "not_eligible"
    assert result.is_near_miss is False
    assert [criterion.criterion for criterion in result.failed_criteria] == ["annual_family_income"]
    assert len(result.satisfied_criteria) == 5
    assert result.total_criteria == 6


def test_multiple_failures_handled() -> None:
    """Two failed criteria exceed the default closeness definition."""
    profile = _satisfied_profile().model_copy(
        update={"annual_family_income": 320000.0, "state": "Bihar"}
    )
    scheme = make_scheme()
    result = NearMissEngine().analyze(
        profile, scheme, EligibilityEngine().evaluate(profile, scheme)
    )
    assert result.is_near_miss is False
    assert len(result.failed_criteria) == 2
    assert len(result.satisfied_criteria) == 4


def test_numeric_failures_expose_difference() -> None:
    """Income/age misses carry deterministic miss distances."""
    income_profile = _satisfied_profile().model_copy(update={"annual_family_income": 320000.0})
    income_result = NearMissEngine().analyze(
        income_profile, make_scheme(), EligibilityEngine().evaluate(income_profile, make_scheme())
    )
    income_failed = income_result.failed_criteria[0]
    assert income_failed.user_value == 320000.0
    assert income_failed.required == "<= 300000"
    assert income_failed.difference == 20000.0

    age_profile = _satisfied_profile().model_copy(update={"age": 17})
    age_result = NearMissEngine().analyze(
        age_profile, make_scheme(), EligibilityEngine().evaluate(age_profile, make_scheme())
    )
    age_failed = age_result.failed_criteria[0]
    assert age_failed.criterion == "age"
    assert age_failed.difference == 1.0


def test_categorical_failures_have_no_numeric_distance() -> None:
    """State/category mismatches carry no invented numeric distance."""
    profile = _satisfied_profile().model_copy(update={"state": "Bihar"})
    result = NearMissEngine().analyze(
        profile, make_scheme(), EligibilityEngine().evaluate(profile, make_scheme())
    )
    assert result.is_near_miss is False
    assert result.failed_criteria[0].difference is None
    assert "Bihar" in str(result.failed_criteria[0].user_value)


def test_missing_information_is_not_a_failure() -> None:
    """Unknown values are reported as missing, never as failed criteria."""
    profile = _satisfied_profile().model_copy(update={"occupation": None})
    scheme = make_scheme()
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "needs_information"
    assert result.is_near_miss is False
    assert result.failed_criteria == []
    assert "occupation" not in result.satisfied_criteria


def test_needs_information_not_treated_as_not_eligible() -> None:
    """Unresolved outcomes are never labeled near misses of a rejection."""
    profile = _satisfied_profile().model_copy(update={"occupation": None})
    scheme = make_scheme()
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert not any("not eligible" in reason.lower() for reason in result.reasons)
    assert result.is_near_miss is False


def test_near_miss_never_overrides_status() -> None:
    """Analysis leaves the original eligibility result untouched."""
    profile = _satisfied_profile().model_copy(update={"annual_family_income": 320000.0})
    eligibility = EligibilityEngine().evaluate(profile, make_scheme())
    NearMissEngine().analyze(profile, make_scheme(), eligibility)
    assert eligibility.status == "not_eligible"


def test_no_semantic_embeddings_used() -> None:
    """Near-miss module has no embedding machinery."""
    import intelligence_engine.near_miss_engine as near_miss_module

    for forbidden in ("EmbeddingProvider", "embed", "cosine", "SemanticMatcher", "encode"):
        assert forbidden not in dir(near_miss_module)


def test_no_match_score_used() -> None:
    """Near-miss module has no matching/scoring machinery."""
    import intelligence_engine.near_miss_engine as near_miss_module

    for forbidden in ("MatchingEngine", "MatchResult", "score_scheme", "rank_schemes"):
        assert forbidden not in dir(near_miss_module)


def test_no_llm_calls() -> None:
    """Near-miss module never touches the LLM layer."""
    import intelligence_engine.near_miss_engine as near_miss_module

    assert "LLMClient" not in dir(near_miss_module)
    assert "llm_client" not in dir(near_miss_module)


def test_no_database_or_api_calls() -> None:
    """Near-miss module has no database or API surface."""
    import intelligence_engine.near_miss_engine as near_miss_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(near_miss_module)


def test_repeated_execution_is_identical() -> None:
    """Same inputs always produce identical near-miss results."""
    profile = _satisfied_profile().model_copy(update={"annual_family_income": 320000.0})
    scheme = make_scheme()
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    engine = NearMissEngine()
    assert engine.analyze(profile, scheme, eligibility) == engine.analyze(profile, scheme, eligibility)


def test_scheme_without_rules_is_not_near_miss() -> None:
    """A rule-less scheme has nothing to be close to."""
    scheme = Scheme(id="s-open", name="Open")
    profile = UserProfile()
    result = NearMissEngine().analyze(
        profile, scheme, EligibilityEngine().evaluate(profile, scheme)
    )
    assert result.total_criteria == 0
    assert result.is_near_miss is False


def test_categorical_single_failure_conservative_off() -> None:
    """CONSERVATIVE-OFF: categorical failures stay off; trace intact for policy work."""
    profile = UserProfile(occupation="Teacher")
    scheme = Scheme(id="s-cat", name="Category-gated", eligibility_rules={"occupations": ["Farmer"]})
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "not_eligible"
    assert len(result.failed_criteria) == 1
    failed = result.failed_criteria[0]
    assert failed.criterion == "occupation"
    assert failed.user_value == "Teacher"
    assert failed.required == "one of: Farmer"
    assert failed.difference is None
    assert result.is_near_miss is False


def test_small_quantitative_failure_conservative_off() -> None:
    """CONSERVATIVE-OFF: numeric distance alone never classifies; trace intact."""
    profile = UserProfile(annual_family_income=550000)
    scheme = Scheme(id="s-small", name="Income-gated", eligibility_rules={"max_annual_income": 500000})
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "not_eligible"
    assert len(result.failed_criteria) == 1
    failed = result.failed_criteria[0]
    assert failed.criterion == "annual_family_income"
    assert failed.user_value == 550000.0
    assert failed.required == "<= 500000"
    assert failed.difference == 50000.0
    assert result.is_near_miss is False


def test_large_quantitative_failure_conservative_off() -> None:
    """CONSERVATIVE-OFF: large misses stay off; trace intact for future policy."""
    profile = UserProfile(annual_family_income=900000)
    scheme = Scheme(id="s-large", name="Income-gated", eligibility_rules={"max_annual_income": 500000})
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "not_eligible"
    assert len(result.failed_criteria) == 1
    failed = result.failed_criteria[0]
    assert failed.criterion == "annual_family_income"
    assert failed.user_value == 900000.0
    assert failed.required == "<= 500000"
    assert failed.difference == 400000.0
    assert result.is_near_miss is False


def test_two_failures_currently_return_not_near_miss() -> None:
    """CURRENT BEHAVIOR BASELINE: two failed dimensions are not a near miss."""
    profile = UserProfile(occupation="Teacher", annual_family_income=900000)
    scheme = Scheme(
        id="s-multi",
        name="Multi-gated",
        eligibility_rules={"occupations": ["Farmer"], "max_annual_income": 500000},
    )
    eligibility = EligibilityEngine().evaluate(profile, scheme)
    result = NearMissEngine().analyze(profile, scheme, eligibility)
    assert eligibility.status == "not_eligible"
    assert len(result.failed_criteria) == 2
    assert result.total_criteria == 2
    assert result.is_near_miss is False
