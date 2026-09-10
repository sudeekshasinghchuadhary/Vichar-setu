"""Adapter tests: backend dicts in, FastAPI-shaped dicts out (offline).

No PostgreSQL, no LLM, no network. Backend shapes mirror
routes.scheme_dict() and ProfileBase.model_dump().
"""

import pytest
from pydantic import ValidationError

from ai_scheme_backend.backend.intelligence.adapter import (
    _to_scheme,
    _to_user_profile,
    check_eligibility,
    match_schemes,
)
from tests.fixtures import make_user_profile


def _profile(**overrides) -> dict:
    """Backend-style profile dict derived from the shared fixture."""
    data = make_user_profile().model_dump()
    data["name"] = "Test User"
    data.update(overrides)
    return data


def _scheme_dict(scheme_id: int = 1, **overrides) -> dict:
    """Backend-style flat scheme dict like routes.scheme_dict()."""
    data = {
        "id": scheme_id,
        "scheme_name": "Prototype support",
        "description": "Prototype scheme for testing.",
        "min_age": 18,
        "max_age": None,
        "max_income": 300000,
        "occupation": "Farmer",
        "education": "12th",
        "category": "OBC",
        "state": "Uttar Pradesh",
        "benefit": "Prototype benefit.",
    }
    data.update(overrides)
    return data


def _farmer_profile() -> dict:
    """Profile satisfying the default scheme dict."""
    return _profile(occupation="Farmer", education_level="Graduate")


def test_eligible_through_real_engine() -> None:
    """Satisfying profile returns eligible with engine reasons."""
    result = check_eligibility(_farmer_profile(), _scheme_dict())
    assert result["eligible"] is True
    assert len(result["reasons"]) >= 1
    assert result["status"] == "VERIFIED"


def test_not_eligible_through_real_engine() -> None:
    """Failing income returns not eligible with reasons."""
    result = check_eligibility(_farmer_profile(), _scheme_dict(max_income=100000))
    assert result["eligible"] is False
    assert any("income" in reason.lower() for reason in result["reasons"])


def test_needs_information_maps_to_not_eligible() -> None:
    """Missing required info is conservative: eligible False, reasons kept."""
    profile = _farmer_profile()
    profile["occupation"] = None
    result = check_eligibility(profile, _scheme_dict())
    assert result["eligible"] is False
    assert any("required" in reason.lower() for reason in result["reasons"])


def test_profile_mapping_keeps_only_present_known_fields() -> None:
    """Backend extras (name) excluded; explicit nulls stay null."""
    user = _to_user_profile({"name": "X", "unknown_key": 1, "age": 30, "occupation": None})
    assert user.age == 30
    assert user.occupation is None
    assert not hasattr(user, "name")
    assert user.model_dump().get("unknown_key", "absent") == "absent"


def test_scheme_mapping_translates_flat_rules() -> None:
    """Flat DB fields become structured rules only when present."""
    scheme = _to_scheme(_scheme_dict(7))
    assert scheme.id == "7"
    assert scheme.name == "Prototype support"
    assert scheme.eligibility_rules["min_age"] == 18
    assert "max_age" not in scheme.eligibility_rules
    assert scheme.eligibility_rules["max_annual_income"] == 300000
    assert scheme.eligibility_rules["occupations"] == ["Farmer"]
    assert scheme.eligibility_rules["min_education"] == "12th"


def test_blank_scheme_strings_add_no_rules() -> None:
    """Empty occupation/education strings invent no requirements."""
    scheme = _to_scheme(_scheme_dict(occupation="  ", education=None, category="", state=" "))
    assert "occupations" not in scheme.eligibility_rules
    assert "min_education" not in scheme.eligibility_rules
    assert "categories" not in scheme.eligibility_rules
    assert "states" not in scheme.eligibility_rules


def test_match_ordering_shape_and_ranks() -> None:
    """Eligible ranked first with engine score; others 0.0, ranks sequential."""
    schemes = [_scheme_dict(1), _scheme_dict(2, max_income=100000)]
    results = match_schemes(_farmer_profile(), schemes)
    assert [item["scheme_id"] for item in results] == [1, 2]
    assert results[0]["eligible"] is True
    assert results[0]["match_score"] > 0
    assert results[1] == {
        "scheme_id": 2,
        "match_score": 0.0,
        "reasons": results[1]["reasons"],
        "eligible": False,
        "rank": 2,
    }
    assert results[0]["rank"] == 1
    assert all(isinstance(item["scheme_id"], int) for item in results)


def test_invalid_profile_rejected() -> None:
    """Bad backend values fail validation instead of coercing."""
    with pytest.raises(ValidationError):
        check_eligibility(_profile(age="old"), _scheme_dict())


def test_deterministic_and_offline() -> None:
    """Repeated calls agree; adapter needs no env keys or services."""
    import ai_scheme_backend.backend.intelligence.adapter as adapter_module

    assert "ENGINE_MODE" not in dir(adapter_module)
    first = match_schemes(_farmer_profile(), [_scheme_dict(1), _scheme_dict(2)])
    assert match_schemes(_farmer_profile(), [_scheme_dict(1), _scheme_dict(2)]) == first
