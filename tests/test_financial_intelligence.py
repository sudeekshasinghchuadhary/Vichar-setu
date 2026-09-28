"""Tests for financial intelligence (pure arithmetic, offline, deterministic)."""

import pytest
from pydantic import ValidationError

from intelligence_engine.financial_intelligence import FinancialError, FinancialIntelligence
from intelligence_engine.schemas import (
    FinancialOption,
    LoanTerms,
    SchemeSupport,
    SupportNeed,
)


def _engine() -> FinancialIntelligence:
    """Fresh engine."""
    return FinancialIntelligence()


def _opt(label: str, amount: float | None, kind: str | None = None) -> FinancialOption:
    """Build an option."""
    return FinancialOption(label=label, kind=kind, amount=amount)


def test_exact_requirement_coverage() -> None:
    """500000 need against 500000 support is fully covered."""
    result = _engine().coverage(500000, [_opt("s", 500000)])
    assert result.coverage_known is True
    assert result.covered == 500000
    assert result.uncovered == 0
    assert result.own_contribution == 0
    assert result.fully_covered is True


def test_partial_coverage_gap() -> None:
    """800000 need against 500000 support leaves a 300000 gap."""
    result = _engine().coverage(800000, [_opt("s", 500000)])
    assert result.covered == 500000
    assert result.uncovered == 300000
    assert result.own_contribution == 300000
    assert result.fully_covered is False


def test_zero_coverage_known() -> None:
    """A known-zero support covers nothing but stays known (not unknown)."""
    result = _engine().coverage(800000, [_opt("s", 0)])
    assert result.coverage_known is True
    assert result.covered == 0
    assert result.uncovered == 800000


def test_support_greater_than_requirement_capped() -> None:
    """600000 support against 500000 need covers 500000, never more."""
    result = _engine().coverage(500000, [_opt("s", 600000)])
    assert result.covered == 500000
    assert result.uncovered == 0
    assert result.fully_covered is True


def test_multiple_options_best_single_wins() -> None:
    """Options are never summed: best of 200000/300000 covers 300000."""
    result = _engine().coverage(500000, [_opt("a", 200000), _opt("b", 300000)])
    assert result.covered == 300000
    assert result.uncovered == 200000


def test_duplicate_options_merged() -> None:
    """Identical options collapse to one (no double counting)."""
    result = _engine().coverage(500000, [_opt("a", 300000), _opt("a", 300000)])
    assert result.covered == 300000


def test_missing_values_stay_unknown() -> None:
    """Unknown requirement or all-unknown options yield unknown coverage."""
    assert _engine().coverage(None, [_opt("s", 500000)]).coverage_known is False
    unknown = _engine().coverage(500000, [_opt("s", None)])
    assert unknown.coverage_known is False
    assert unknown.covered is None
    assert unknown.uncovered is None


def test_unknown_never_becomes_zero() -> None:
    """Unknown coverage reports None, not 0."""
    result = _engine().coverage(500000, [])
    assert result.coverage_known is False
    assert result.covered is None
    assert result.fully_covered is False


def test_invalid_and_negative_values_rejected() -> None:
    """Negatives fail validation; bad required types raise FinancialError."""
    with pytest.raises(ValidationError):
        FinancialOption(label="s", amount=-5)
    with pytest.raises(ValidationError):
        LoanTerms(principal=-1000, annual_rate_percent=5, tenure_months=12)
    with pytest.raises(ValidationError):
        LoanTerms(principal=1000, annual_rate_percent=-1, tenure_months=12)
    with pytest.raises(ValidationError):
        LoanTerms(principal=1000, annual_rate_percent=5, tenure_months=0)
    with pytest.raises(FinancialError):
        _engine().coverage(-100, [_opt("s", 500000)])
    with pytest.raises(FinancialError):
        _engine().coverage("lots", [_opt("s", 500000)])  # type: ignore[arg-type]


def test_zero_interest_emi_exact() -> None:
    """120000 at 0% over 12 months is exactly 10000/month, no interest."""
    result = _engine().loan_schedule(LoanTerms(principal=120000, annual_rate_percent=0, tenure_months=12))
    assert result.monthly_emi == 10000
    assert result.total_payable == 120000
    assert result.total_interest == 0


def test_normal_emi_matches_formula() -> None:
    """100000 at 12% over 12 months matches the standard amortizing EMI."""
    result = _engine().loan_schedule(LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=12))
    assert result.monthly_emi == pytest.approx(8884.88, abs=0.01)
    assert result.total_payable == pytest.approx(result.monthly_emi * 12, abs=0.01)
    assert result.total_interest == pytest.approx(result.total_payable - 100000, abs=0.01)


def test_compare_ranks_known_only() -> None:
    """Known amounts rank highest-first; unknowns stay listed but unranked."""
    result = _engine().compare([_opt("b", 200000), _opt("a", 300000), _opt("c", None)])
    assert result.ranked_labels == ["a", "b"]
    assert result.unknown_labels == ["c"]
    assert result.best_amount == 300000


def test_coverage_for_need_bridge() -> None:
    """SupportNeed + SchemeSupport flow into arithmetic without eligibility checks."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")
    supports = [SchemeSupport(support_type="loan", covers_need_types=["machinery"], max_amount_period="one_time", max_amount=300000)]
    result = _engine().coverage_for_need(need, supports)
    assert result.covered == 300000
    assert result.uncovered == 200000
    monthly = SupportNeed(need_type="machinery", amount=5000, amount_period="monthly")
    assert _engine().coverage_for_need(monthly, supports).coverage_known is False


def test_deterministic_results() -> None:
    """Same inputs always give identical outputs."""
    engine = _engine()
    assert engine.coverage(800000, [_opt("s", 500000)]) == engine.coverage(800000, [_opt("s", 500000)])
    terms = LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=12)
    assert engine.loan_schedule(terms) == engine.loan_schedule(terms)


def test_inputs_unchanged() -> None:
    """Options and terms are unchanged by calculations."""
    options = [_opt("a", 300000), _opt("b", None)]
    terms = LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=12)
    before = ([option.model_dump() for option in options], terms.model_dump())
    _engine().coverage(800000, options)
    _engine().loan_schedule(terms)
    _engine().compare(options)
    assert [option.model_dump() for option in options] == before[0]
    assert terms.model_dump() == before[1]


def test_no_llm_or_eligibility_duplication() -> None:
    """Financial module holds only arithmetic machinery."""
    import intelligence_engine.financial_intelligence as finance_module

    for forbidden in (
        "LLMClient", "llm_client", "EligibilityEngine", "evaluate",
        "MatchingEngine", "EmbeddingProvider", "cosine", "requests", "fastapi",
        "sqlite", "postgres",
    ):
        assert forbidden not in dir(finance_module)
