"""Tests for the canonical need vocabulary and financial-period contract."""

from intelligence_engine.financial_intelligence import FinancialIntelligence
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.need_vocabulary import (
    CANONICAL_NEED_TYPES,
    is_canonical_need_type,
    normalize_need_type,
)
from intelligence_engine.schemas import (
    EligibilityResult,
    Scheme,
    SchemeSupport,
    SupportNeed,
)
from intelligence_engine.support_planner import SupportPlanningEngine
from tests.fixtures import make_user_profile


def _support(**overrides) -> SchemeSupport:
    """Support option with explicit period by default."""
    data = {
        "support_type": "loan",
        "covers_need_types": ["machinery"],
        "max_amount": 300000,
        "max_amount_period": "one_time",
    }
    data.update(overrides)
    return SchemeSupport(**data)


def _scheme(scheme_id: str, support: SchemeSupport) -> Scheme:
    """Scheme carrying one support option."""
    return Scheme(id=scheme_id, name=scheme_id, support_options=[support])


def _eligible(scheme_id: str) -> EligibilityResult:
    """Eligible result for a scheme."""
    return EligibilityResult(scheme_id=scheme_id, status="eligible", reasons=["ok"])


def test_canonical_need_types() -> None:
    """The canonical set is small and stable."""
    assert set(CANONICAL_NEED_TYPES) == {
        "machinery",
        "working_capital",
        "training",
        "marketing",
        "infrastructure",
    }


def test_aliases_collapse_to_canonical() -> None:
    """Equivalent aliases resolve to one canonical type."""
    assert normalize_need_type("machines") == ("machinery", True)
    assert normalize_need_type("equipment") == ("machinery", True)
    assert normalize_need_type("Working Capital") == ("working_capital", True)
    assert normalize_need_type("skill training") == ("training", True)


def test_canonical_type_maps_to_itself() -> None:
    """Canonical inputs are recognized and unchanged."""
    for canonical in CANONICAL_NEED_TYPES:
        assert normalize_need_type(canonical) == (canonical, True)
        assert is_canonical_need_type(canonical) is True


def test_case_and_whitespace_variants_unify() -> None:
    """Casing/whitespace never create duplicate meanings."""
    assert normalize_need_type("  MACHINES  ") == ("machinery", True)
    assert normalize_need_type("Working-Capital") == ("working_capital", True)
    assert SupportNeed(need_type="  Machines ").need_type == "machinery"


def test_unknown_types_preserved_safely() -> None:
    """Unknown wording is kept as cleaned text, never invented."""
    assert normalize_need_type("raw material") == ("raw material", False)
    assert is_canonical_need_type("raw material") is False
    assert SupportNeed(need_type="raw material").need_type == "raw material"


def test_scheme_support_uses_shared_vocabulary() -> None:
    """covers_need_types normalize through the same vocabulary."""
    support = SchemeSupport(support_type="loan", covers_need_types=["Machines ", "raw material"])
    assert support.covers_need_types == ["machinery", "raw material"]


def test_one_time_to_one_time_coverage() -> None:
    """Equal known one-time periods allow numeric coverage."""
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme("s", _support())],
        [_eligible("s")],
    )
    assert plan.need_plans[0].best_known_coverage == 300000


def test_monthly_to_monthly_coverage() -> None:
    """Equal known monthly periods allow numeric coverage."""
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=20000, amount_period="monthly")],
        [_scheme("s", _support(max_amount=15000, max_amount_period="monthly"))],
        [_eligible("s")],
    )
    assert plan.need_plans[0].best_known_coverage == 15000


def test_annual_to_annual_coverage() -> None:
    """Equal known annual periods allow numeric coverage."""
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=240000, amount_period="annual")],
        [_scheme("s", _support(max_amount=200000, max_amount_period="annual"))],
        [_eligible("s")],
    )
    assert plan.need_plans[0].best_known_coverage == 200000


def test_incompatible_periods_stay_unknown() -> None:
    """Monthly need vs one-time support never converts: unknown, not zero."""
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=20000, amount_period="monthly")],
        [_scheme("s", _support())],
        [_eligible("s")],
    )
    mapping = plan.need_plans[0].mappings[0]
    assert mapping.coverage_known is False
    assert mapping.potential_coverage is None
    assert any("Period mismatch" in reason for reason in mapping.reasons)


def test_unknown_periods_stay_unknown() -> None:
    """Unknown need or support periods block coverage without guessing."""
    unknown_support = _support()
    unknown_support.max_amount_period = "unknown"
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme("s", unknown_support)],
        [_eligible("s")],
    )
    assert plan.need_plans[0].best_known_coverage is None


def test_legacy_max_amount_defaults_to_unknown_period() -> None:
    """Entries with only max_amount construct fine but assume no period."""
    support = SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount=500000)
    assert support.max_amount_period == "unknown"
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme("s", support)],
        [_eligible("s")],
    )
    assert plan.need_plans[0].best_known_coverage is None


def test_no_silent_monthly_annual_conversion() -> None:
    """12x monthly is never annualized; annual never split into months."""
    engine = FinancialIntelligence()
    monthly_need = SupportNeed(need_type="machinery", amount=20000, amount_period="monthly")
    annual_support = [_support(max_amount=240000, max_amount_period="annual")]
    assert engine.coverage_for_need(monthly_need, annual_support).coverage_known is False
    annual_need = SupportNeed(need_type="machinery", amount=240000, amount_period="annual")
    monthly_support = [_support(max_amount=20000, max_amount_period="monthly")]
    assert engine.coverage_for_need(annual_need, monthly_support).coverage_known is False


def test_mixed_period_totals_not_mixed() -> None:
    """Plan totals stay unknown when comparable needs span periods."""
    plan = SupportPlanningEngine().plan(
        make_user_profile(),
        [
            SupportNeed(need_type="machinery", amount=500000, amount_period="one_time"),
            SupportNeed(need_type="working_capital", amount=20000, amount_period="monthly"),
        ],
        [
            _scheme("a", _support(max_amount=500000, max_amount_period="one_time")),
            _scheme(
                "b",
                SchemeSupport(
                    support_type="subsidy",
                    covers_need_types=["working_capital"],
                    max_amount=20000,
                    max_amount_period="monthly",
                ),
            ),
        ],
        [_eligible("a"), _eligible("b")],
    )
    assert plan.need_plans[0].best_known_coverage == 500000
    assert plan.need_plans[1].best_known_coverage == 20000
    assert plan.totals_fully_known is False
    assert plan.total_need_amount is None


def test_analyzer_still_uses_shared_vocabulary() -> None:
    """NeedAnalyzer normalization flows through the central vocabulary."""
    from tests.test_need_analyzer import FakeNeedExtractor

    result = NeedAnalyzer(
        FakeNeedExtractor({"needs": [{"need_type": "MACHINES", "amount": 1000}]})
    ).analyze("machines")
    assert result.needs[0].need_type == "machinery"


def test_deterministic_and_immutable() -> None:
    """Vocabulary and period logic are stable and side-effect free."""
    assert normalize_need_type("machines") == normalize_need_type("  MACHINES ")
    support = _support()
    before = support.model_dump()
    SupportPlanningEngine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme("s", support)],
        [_eligible("s")],
    )
    assert support.model_dump() == before
