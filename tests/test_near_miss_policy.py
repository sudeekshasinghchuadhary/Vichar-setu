"""Near-Miss policy invariants (locked).

Interim policy is CONSERVATIVE-OFF: no criterion currently has
domain-validated closeness support, so no NOT_ELIGIBLE scheme may be
classified as a near miss. These tests lock that posture plus the
structural invariants that any future policy must preserve.
"""
import pytest

from intelligence_engine.eligibility_engine import EligibilityEngine, EligibilityEngineError
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.schemas import Scheme, UserProfile


def _analyze(profile, rules):
    scheme = Scheme(id="s", name="S", eligibility_rules=rules)
    engine = EligibilityEngine()
    return engine, NearMissEngine().analyze(profile, scheme, engine.evaluate(profile, scheme))


def test_close_income_plus_failing_occupation_is_not_near_miss() -> None:
    """One close criterion cannot compensate an unsupported failure."""
    _, result = _analyze(
        UserProfile(annual_family_income=305000, occupation="Teacher"),
        {"max_annual_income": 300000, "occupations": ["Farmer"]},
    )
    assert result.is_near_miss is False
    assert [c.criterion for c in result.failed_criteria] == ["annual_family_income", "occupation"]


def test_two_independently_close_failures_still_not_near_miss() -> None:
    """Without validated per-criterion support, even close+close stays off."""
    _, result = _analyze(
        UserProfile(age=17, annual_family_income=301000),
        {"min_age": 18, "max_annual_income": 300000},
    )
    assert result.is_near_miss is False


def test_passing_criteria_neither_help_nor_hurt() -> None:
    """Only failed criteria participate; extra passes change nothing."""
    _, one_rule = _analyze(
        UserProfile(annual_family_income=320000),
        {"max_annual_income": 300000},
    )
    _, five_rules = _analyze(
        UserProfile(age=25, annual_family_income=320000, occupation="Farmer",
                    education_level="Graduate", social_category="OBC", state="Uttar Pradesh"),
        {"min_age": 18, "max_annual_income": 300000, "occupations": ["Farmer"],
         "min_education": "12th", "categories": ["OBC"], "states": ["Uttar Pradesh"]},
    )
    assert one_rule.is_near_miss is False
    assert five_rules.is_near_miss is False
    assert [c.criterion for c in five_rules.failed_criteria] == ["annual_family_income"]


def test_conservative_off_across_failure_shapes() -> None:
    """No failure shape yields a positive classification under conservative-off."""
    cases = [
        (UserProfile(annual_family_income=305000), {"max_annual_income": 300000}),
        (UserProfile(annual_family_income=700000), {"max_annual_income": 300000}),
        (UserProfile(occupation="Teacher"), {"occupations": ["Farmer"]}),
        (UserProfile(age=17), {"min_age": 18}),
        (UserProfile(age=41), {"max_age": 40}),
    ]
    for profile, rules in cases:
        _, result = _analyze(profile, rules)
        assert result.is_near_miss is False


def test_numeric_distance_alone_is_never_sufficient() -> None:
    """Small and large differences alike stay off without validated policy."""
    _, small = _analyze(UserProfile(annual_family_income=305000), {"max_annual_income": 300000})
    _, large = _analyze(UserProfile(annual_family_income=700000), {"max_annual_income": 300000})
    assert small.failed_criteria[0].difference == 5000.0
    assert large.failed_criteria[0].difference == 400000.0
    assert small.is_near_miss is False
    assert large.is_near_miss is False


def test_categorical_mismatch_is_never_sufficient() -> None:
    """Membership failures carry no closeness measure and stay off."""
    for rules, profile in [
        ({"occupations": ["Farmer"]}, UserProfile(occupation="Teacher")),
        ({"categories": ["OBC"]}, UserProfile(social_category="General")),
        ({"states": ["Uttar Pradesh"]}, UserProfile(state="Bihar")),
    ]:
        _, result = _analyze(profile, rules)
        assert result.failed_criteria[0].difference is None
        assert result.is_near_miss is False


def test_multiple_failures_stay_off() -> None:
    """Count-based closeness is gone; two failures stay off."""
    _, result = _analyze(
        UserProfile(annual_family_income=320000, state="Bihar"),
        {"max_annual_income": 300000, "states": ["Uttar Pradesh"]},
    )
    assert len(result.failed_criteria) == 2
    assert result.is_near_miss is False


def test_needs_information_is_never_near_miss() -> None:
    """Unresolved schemes go to clarification, never to near-miss."""
    _, result = _analyze(UserProfile(), {"min_age": 18})
    assert result.is_near_miss is False
    assert result.failed_criteria == []


def test_malformed_rule_raises_typed_error() -> None:
    """Scheme A: first malformed criterion aborts with a typed error."""
    scheme = Scheme(id="s", name="S", eligibility_rules={
        "min_age": "eighteen", "max_age": 65, "max_annual_income": 500000})
    with pytest.raises(EligibilityEngineError, match="Rule 'min_age' must be a number"):
        EligibilityEngine().evaluate(UserProfile(age=25, annual_family_income=300000), scheme)


def test_multiple_malformed_rules_report_first_only() -> None:
    """Scheme C: evaluation stops at the first malformed criterion."""
    scheme = Scheme(id="s", name="S", eligibility_rules={
        "min_age": "eighteen", "max_annual_income": "five lakh", "max_age": 65})
    with pytest.raises(EligibilityEngineError, match="Rule 'min_age' must be a number"):
        EligibilityEngine().evaluate(UserProfile(age=25, annual_family_income=300000), scheme)


def test_malformed_scheme_batch_boundary() -> None:
    """Individual evaluations are independent; a batch call aborts on malformed input.

    CURRENT BEHAVIOR (verified, not endorsed): evaluate_many() does not
    isolate per scheme — one malformed scheme aborts the whole batch.
    Callers needing isolation must evaluate per scheme.
    """
    engine = EligibilityEngine()
    profile = UserProfile(age=25, annual_family_income=300000)
    good = Scheme(id="good", name="G", eligibility_rules={
        "min_age": 18, "max_age": 65, "max_annual_income": 500000})
    bad = Scheme(id="bad", name="B", eligibility_rules={"min_age": "eighteen"})
    assert engine.evaluate(profile, good).status == "eligible"
    with pytest.raises(EligibilityEngineError):
        engine.evaluate(profile, bad)
    with pytest.raises(EligibilityEngineError):
        engine.evaluate_many(profile, [good, bad])
    assert engine.evaluate(profile, good).status == "eligible"


def _analyzed(profile, rules, supported):
    """Analyze with a test-only injected validation set (monkeypatched)."""
    import intelligence_engine.near_miss_engine as engine_module

    original = engine_module._VALIDATED_CLOSENESS_CRITERIA
    engine_module._VALIDATED_CLOSENESS_CRITERIA = frozenset(supported)
    try:
        scheme = Scheme(id="s", name="S", eligibility_rules=rules)
        engine = EligibilityEngine()
        return NearMissEngine().analyze(profile, scheme, engine.evaluate(profile, scheme))
    finally:
        engine_module._VALIDATED_CLOSENESS_CRITERIA = original


def test_zero_validated_criteria_any_failures_off() -> None:
    """With an empty support set, any failure shape stays off."""
    assert _analyzed(
        UserProfile(annual_family_income=320000), {"max_annual_income": 300000}, set()
    ).is_near_miss is False
    assert _analyzed(
        UserProfile(age=17, annual_family_income=320000, occupation="Teacher"),
        {"min_age": 18, "max_annual_income": 300000, "occupations": ["Farmer"]},
        set(),
    ).is_near_miss is False


def test_one_supported_close_failure_on() -> None:
    """A single supported failure qualifies with no count involved."""
    result = _analyzed(
        UserProfile(annual_family_income=305000), {"max_annual_income": 300000},
        {"annual_family_income"},
    )
    assert result.is_near_miss is True
    assert [c.criterion for c in result.failed_criteria] == ["annual_family_income"]


def test_two_supported_close_failures_on() -> None:
    """Two independently supported failures qualify; no count cap applies."""
    result = _analyzed(
        UserProfile(age=17, annual_family_income=301000),
        {"min_age": 18, "max_annual_income": 300000},
        {"age", "annual_family_income"},
    )
    assert result.is_near_miss is True


def test_three_supported_close_failures_on() -> None:
    """Three independently supported failures qualify; still no count cap."""
    result = _analyzed(
        UserProfile(age=17, annual_family_income=301000, state="Bihar"),
        {"min_age": 18, "max_annual_income": 300000, "states": ["Uttar Pradesh"]},
        {"age", "annual_family_income", "state"},
    )
    assert result.is_near_miss is True
    assert len(result.failed_criteria) == 3


def test_supported_plus_unsupported_failure_off() -> None:
    """One unsupported failure blocks, even beside a supported one."""
    result = _analyzed(
        UserProfile(annual_family_income=305000, occupation="Teacher"),
        {"max_annual_income": 300000, "occupations": ["Farmer"]},
        {"annual_family_income"},
    )
    assert result.is_near_miss is False
    assert [c.criterion for c in result.failed_criteria] == ["annual_family_income", "occupation"]


def test_passing_criteria_ignored_with_support() -> None:
    """Passes never gate the verdict, supported or not."""
    supported = _analyzed(
        UserProfile(age=25, annual_family_income=305000, state="Uttar Pradesh"),
        {"min_age": 18, "max_annual_income": 300000, "states": ["Uttar Pradesh"]},
        {"annual_family_income"},
    )
    assert supported.is_near_miss is True
    assert [c.criterion for c in supported.failed_criteria] == ["annual_family_income"]


def test_needs_information_never_near_miss_with_support() -> None:
    """Unresolved schemes stay out even when other criteria have support."""
    result = _analyzed(
        UserProfile(), {"min_age": 18, "max_annual_income": 300000},
        {"age", "annual_family_income"},
    )
    assert result.is_near_miss is False
    assert result.failed_criteria == []
