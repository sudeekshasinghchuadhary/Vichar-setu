"""Tests: Haversine distance, 4-tier partner matching, moratorium EMI."""

import pytest
from pydantic import ValidationError

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.financial_intelligence import FinancialError, FinancialIntelligence
from intelligence_engine.partner_matcher import PartnerMatcher, haversine_distance_km
from intelligence_engine.schemas import (
    ChannelPartner,
    LoanTerms,
    Scheme,
    UserProfile,
)


def _partner(pid, ptype, state=None, district=None, lat=None, lon=None):
    return ChannelPartner(
        id=pid, name=f"Partner {pid}", partner_type=ptype,
        state=state, district=district, latitude=lat, longitude=lon,
        source="test", source_date="2026-01-01",
    )


# --- Haversine ---

def test_haversine_known_distances():
    # Delhi (28.6139, 77.2090) -> Mumbai (19.0760, 72.8777) ≈ 1148 km
    d = haversine_distance_km(28.6139, 77.2090, 19.0760, 72.8777)
    assert d == pytest.approx(1148, abs=15)
    # Same point is zero
    assert haversine_distance_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)
    # 1 degree of longitude at equator ≈ 111.19 km
    assert haversine_distance_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.5)


def test_haversine_symmetry():
    a = haversine_distance_km(12.97, 77.59, 13.08, 80.27)
    b = haversine_distance_km(13.08, 80.27, 12.97, 77.59)
    assert a == pytest.approx(b)


# --- Tiers ---

def test_tier_nearest_by_distance_sorted_with_values():
    partners = [
        _partner("far", "PSB", "Karnataka", "Bengaluru", 13.5, 78.5),
        _partner("near", "PSB", "Karnataka", "Bengaluru", 12.98, 77.60),
    ]
    result = PartnerMatcher().match("PSB", partners, "Karnataka", "Bengaluru", 12.9716, 77.5946)
    assert result.tier == "NEAREST_BY_DISTANCE"
    assert [m.partner.id for m in result.matches] == ["near", "far"]
    assert all(m.distance_km is not None and m.distance_km >= 0 for m in result.matches)
    assert result.matches[0].distance_km < result.matches[1].distance_km


def test_tier_exact_district():
    partners = [_partner("d1", "RRB", "Bihar", "Patna")]
    result = PartnerMatcher().match("RRB", partners, "Bihar", "Patna")
    assert result.tier == "EXACT_DISTRICT"
    assert len(result.matches) == 1
    assert all(m.distance_km is None for m in result.matches)


def test_tier_state_level():
    partners = [_partner("s1", "SCA", "Odisha", "Cuttack")]
    result = PartnerMatcher().match("SCA", partners, "Odisha", "Puri")
    assert result.tier == "STATE_LEVEL"
    assert len(result.matches) == 1


def test_tier_type_only_no_listing():
    result = PartnerMatcher().match("NBFC_MFI", [], "Kerala", "Kochi")
    assert result.tier == "TYPE_ONLY_NO_LISTING"
    assert result.matches == []
    assert result.required_partner_type == "NBFC_MFI"
    assert any("NBFC_MFI" in r for r in result.reasons)


def test_missing_user_coords_falls_back_not_error():
    partners = [_partner("d1", "PSB", "Karnataka", "Mysuru")]
    result = PartnerMatcher().match("PSB", partners, "Karnataka", "Mysuru")
    assert result.tier == "EXACT_DISTRICT"


def test_missing_partner_coords_never_nearest():
    partners = [_partner("n1", "PSB", "Karnataka", "Bengaluru")]  # no coords
    result = PartnerMatcher().match(
        "PSB", partners, "Karnataka", "Bengaluru", 12.9716, 77.5946
    )
    assert result.tier != "NEAREST_BY_DISTANCE"
    assert all(m.distance_km is None for m in result.matches)
    assert "nearest" not in " ".join(result.reasons).lower() or result.tier == "NEAREST_BY_DISTANCE"


def test_wrong_type_partners_ignored():
    partners = [_partner("x", "PSB", "Bihar", "Patna")]
    result = PartnerMatcher().match("RRB", partners, "Bihar", "Patna")
    assert result.tier == "TYPE_ONLY_NO_LISTING"


def test_partner_matching_does_not_alter_eligibility():
    profile = UserProfile(age=30, state="Bihar")
    scheme = Scheme(id="s1", name="S", eligibility_rules={"min_age": 18, "states": ["Bihar"]})
    before = EligibilityEngine().evaluate(profile, scheme)
    PartnerMatcher().match("PSB", [], "Bihar", "Patna", 25.6, 85.1)
    after = EligibilityEngine().evaluate(profile, scheme)
    assert before == after
    assert after.status == "eligible"


def test_partner_fields_optional_no_fabrication():
    p = ChannelPartner(id="p1", name="P", partner_type="OTHER")
    assert p.latitude is None and p.district is None and p.contact is None


# --- Moratorium EMI ---

def test_moratorium_zero_matches_legacy():
    engine = FinancialIntelligence()
    terms = LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=12)
    result = engine.loan_schedule(terms)
    assert result.moratorium_months == 0
    assert result.post_moratorium_emi == result.monthly_emi
    assert result.repayment_months == 12
    assert result.monthly_emi == pytest.approx(8884.88, abs=0.01)
    assert result.total_payable == pytest.approx(result.monthly_emi * 12, abs=0.01)
    assert result.total_interest == pytest.approx(result.total_payable - 100000, abs=0.01)


def test_moratorium_zero_interest_with_holiday():
    engine = FinancialIntelligence()
    terms = LoanTerms(
        principal=120000, annual_rate_percent=0, tenure_months=12,
        moratorium_months=3, moratorium_interest_accrues=True,
    )
    result = engine.loan_schedule(terms)
    assert result.repayment_months == 9
    assert result.post_moratorium_emi == pytest.approx(120000 / 9, abs=0.01)
    assert result.total_interest == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("m", [3, 6])
def test_moratorium_accruing_raises_emi(m):
    engine = FinancialIntelligence()
    base = engine.loan_schedule(LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=24))
    hol = engine.loan_schedule(LoanTerms(
        principal=100000, annual_rate_percent=12, tenure_months=24,
        moratorium_months=m, moratorium_interest_accrues=True,
    ))
    assert hol.repayment_months == 24 - m
    assert hol.moratorium_months == m
    assert hol.post_moratorium_emi == hol.monthly_emi
    assert hol.total_interest > base.total_interest


def test_moratorium_12_months_accruing():
    engine = FinancialIntelligence()
    result = engine.loan_schedule(LoanTerms(
        principal=100000, annual_rate_percent=12, tenure_months=24,
        moratorium_months=12, moratorium_interest_accrues=True,
    ))
    r = 12 / 1200.0
    grown = 100000 * ((1 + r) ** 12)
    factor = (1 + r) ** 12
    expected_emi = round(grown * r * factor / (factor - 1), 2)
    assert result.post_moratorium_emi == pytest.approx(expected_emi, abs=0.01)
    assert result.repayment_months == 12


def test_moratorium_non_accruing_vs_accruing():
    engine = FinancialIntelligence()
    kw = dict(principal=100000, annual_rate_percent=12, tenure_months=24, moratorium_months=6)
    acc = engine.loan_schedule(LoanTerms(**kw, moratorium_interest_accrues=True))
    waive = engine.loan_schedule(LoanTerms(**kw, moratorium_interest_accrues=False))
    assert waive.post_moratorium_emi < acc.post_moratorium_emi
    assert waive.total_interest < acc.total_interest
    # Non-accruing: same EMI as amortizing original principal over 18 months
    plain18 = engine.loan_schedule(LoanTerms(principal=100000, annual_rate_percent=12, tenure_months=18))
    assert waive.post_moratorium_emi == pytest.approx(plain18.monthly_emi, abs=0.01)


def test_moratorium_validation():
    engine = FinancialIntelligence()
    with pytest.raises(ValidationError):
        LoanTerms(principal=1000, annual_rate_percent=5, tenure_months=12, moratorium_months=13)
    with pytest.raises(FinancialError):
        engine.loan_schedule(LoanTerms(principal=1000, annual_rate_percent=5, tenure_months=12),
                             moratorium_months=12)
    with pytest.raises(FinancialError):
        engine.loan_schedule(LoanTerms(principal=1000, annual_rate_percent=5, tenure_months=12),
                             moratorium_months=-1)
