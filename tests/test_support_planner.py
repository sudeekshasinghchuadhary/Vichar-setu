"""Tests for support planning (deterministic, offline, no invented values)."""

from intelligence_engine.schemas import (
    EligibilityResult,
    Scheme,
    SchemeSupport,
    SupportNeed,
    UserProfile,
)
from intelligence_engine.support_planner import SupportPlanningEngine
from tests.fixtures import make_user_profile


def _scheme(
    scheme_id: str = "scheme-001",
    supports: list[SchemeSupport] | None = None,
) -> Scheme:
    """Scheme with the given verified support options."""
    return Scheme(
        id=scheme_id,
        name=f"Scheme {scheme_id}",
        support_options=supports
        if supports is not None
        else [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=500000)],
    )


def _eligible(scheme_id: str = "scheme-001") -> EligibilityResult:
    """Eligible result for a scheme."""
    return EligibilityResult(scheme_id=scheme_id, status="eligible", reasons=["ok"])


def _engine() -> SupportPlanningEngine:
    """Fresh engine."""
    return SupportPlanningEngine()


def test_one_need_one_support_exact_coverage() -> None:
    """Need 500000 with max 500000 is fully covered."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme()],
        [_eligible()],
    )
    need_plan = plan.need_plans[0]
    assert need_plan.best_known_coverage == 500000
    assert need_plan.uncovered_amount == 0
    assert need_plan.fully_covered is True


def test_one_need_multiple_schemes_best_wins() -> None:
    """Two candidates report separately; best single coverage stands."""
    schemes = [
        _scheme("scheme-001", [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=200000)]),
        _scheme("scheme-002", [SchemeSupport(support_type="grant", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=300000)]),
    ]
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        schemes,
        [_eligible("scheme-001"), _eligible("scheme-002")],
    )
    need_plan = plan.need_plans[0]
    assert len(need_plan.mappings) == 2
    assert need_plan.best_known_coverage == 300000
    assert need_plan.uncovered_amount == 200000


def test_multiple_needs_planned_separately() -> None:
    """Machinery + working capital each get their own mappings."""
    schemes = [
        _scheme("scheme-001", [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=500000)]),
        _scheme("scheme-002", [SchemeSupport(support_type="subsidy", covers_need_types=["working_capital"], max_amount_period="one_time", max_amount=100000)]),
    ]
    plan = _engine().plan(
        make_user_profile(),
        [
            SupportNeed(need_type="machinery", amount=500000, amount_period="one_time"),
            SupportNeed(need_type="working_capital", amount=300000, amount_period="one_time"),
        ],
        schemes,
        [_eligible("scheme-001"), _eligible("scheme-002")],
    )
    assert len(plan.need_plans) == 2
    assert plan.need_plans[1].best_known_coverage == 100000
    assert plan.need_plans[1].uncovered_amount == 200000


def test_partial_coverage_numbers() -> None:
    """Need 800000 against max 500000 leaves 300000 uncovered."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=800000, amount_period="one_time")],
        [_scheme()],
        [_eligible()],
    )
    mapping = plan.need_plans[0].mappings[0]
    assert mapping.potential_coverage == 500000
    assert mapping.uncovered_amount == 300000
    assert mapping.coverage_known is True


def test_unknown_support_amount_preserves_uncertainty() -> None:
    """Unknown max is not zero and yields no coverage numbers."""
    schemes = [_scheme("scheme-001", [SchemeSupport(support_type="loan", covers_need_types=["machinery"])])]
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        schemes,
        [_eligible()],
    )
    mapping = plan.need_plans[0].mappings[0]
    assert mapping.max_amount is None
    assert mapping.coverage_known is False
    assert mapping.potential_coverage is None
    assert plan.need_plans[0].best_known_coverage is None


def test_missing_need_amount_no_coverage() -> None:
    """Amount-less needs map for relevance but never gain numbers."""
    plan = _engine().plan(
        make_user_profile(), [SupportNeed(need_type="machinery")], [_scheme()], [_eligible()]
    )
    mapping = plan.need_plans[0].mappings[0]
    assert mapping.coverage_known is False
    assert plan.need_plans[0].uncovered_amount is None


def test_ineligible_scheme_counts_nothing() -> None:
    """not_eligible mappings carry status and zero coverage."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme()],
        [EligibilityResult(scheme_id="scheme-001", status="not_eligible", reasons=["failed"])],
    )
    mapping = plan.need_plans[0].mappings[0]
    assert mapping.eligibility_status == "not_eligible"
    assert mapping.coverage_known is False
    assert plan.need_plans[0].best_known_coverage is None


def test_needs_information_scheme_counts_nothing() -> None:
    """Unresolved eligibility stays uncertain, never covered."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme()],
        [EligibilityResult(scheme_id="scheme-001", status="needs_information", reasons=["missing"])],
    )
    assert plan.need_plans[0].mappings[0].coverage_known is False
    assert plan.need_plans[0].best_known_coverage is None
    assert any("Resolve eligibility" in action for action in plan.next_actions)


def test_overlap_without_convergence_never_summed() -> None:
    """Two 300000 supports against a 500000 need give best 300000, not 600000."""
    schemes = [
        _scheme("scheme-001", [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=300000)]),
        _scheme("scheme-002", [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=300000)]),
    ]
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        schemes,
        [_eligible("scheme-001"), _eligible("scheme-002")],
    )
    assert plan.need_plans[0].best_known_coverage == 300000
    assert plan.convergence_unknown is True


def test_duplicate_scheme_input_mapped_once() -> None:
    """Passing the same scheme twice does not duplicate mappings."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme(), _scheme()],
        [_eligible()],
    )
    assert len(plan.need_plans[0].mappings) == 1


def test_no_invented_support_or_values() -> None:
    """Schemes without support_options yield no mappings and no numbers."""
    plan = _engine().plan(
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [Scheme(id="scheme-009", name="No support data", benefit="Large benefits available")],
        [_eligible("scheme-009")],
    )
    assert plan.need_plans[0].mappings == []
    assert plan.need_plans[0].best_known_coverage is None
    assert any("No verified scheme support" in reason for reason in plan.need_plans[0].reasons)


def test_deterministic_output() -> None:
    """Identical inputs give identical plans."""
    args = (
        make_user_profile(),
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [_scheme()],
        [_eligible()],
    )
    assert _engine().plan(*args).model_dump() == _engine().plan(*args).model_dump()


def test_no_llm_calls() -> None:
    """Planner module never touches LLM machinery."""
    import intelligence_engine.support_planner as planner_module

    for forbidden in ("LLMClient", "llm_client", "NeedExtractor", "embed", "cosine"):
        assert forbidden not in dir(planner_module)


def test_no_database_or_api_calls() -> None:
    """Planner module has no database or API surface."""
    import intelligence_engine.support_planner as planner_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(planner_module)


def test_inputs_not_mutated() -> None:
    """Profile, needs, and schemes are unchanged by planning."""
    profile = make_user_profile()
    needs = [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")]
    schemes = [_scheme()]
    before = (
        profile.model_dump(),
        [need.model_dump() for need in needs],
        [scheme.model_dump() for scheme in schemes],
    )
    _engine().plan(profile, needs, schemes, [_eligible()])
    assert profile.model_dump() == before[0]
    assert [need.model_dump() for need in needs] == before[1]
    assert [scheme.model_dump() for scheme in schemes] == before[2]
    assert isinstance(profile, UserProfile)
