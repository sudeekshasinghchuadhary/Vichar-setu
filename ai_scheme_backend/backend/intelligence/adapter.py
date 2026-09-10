"""Adapter for Ananya's intelligence engine.

Expected external module: intelligence_engine.py
Expected functions:
  check_eligibility(profile: dict, scheme: dict) -> dict
  match_schemes(profile: dict, schemes: list[dict]) -> list[dict]

The fallback below is intentionally a tiny demo stub so the FastAPI APIs can be
run before Ananya's engine is merged. Replace it by setting ENGINE_MODE=real and
installing/importing intelligence_engine.py.
"""
import os
from typing import Any


def _real_engine():
    from intelligence_engine import check_eligibility, match_schemes
    return check_eligibility, match_schemes


def check_eligibility(profile: dict[str, Any], scheme: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("ENGINE_MODE", "mock").lower() == "real":
        fn, _ = _real_engine()
        return fn(profile, scheme)

    reasons: list[str] = []
    age = profile.get("age")
    income = profile.get("annual_family_income")
    occupation = (profile.get("occupation") or "").strip().lower()
    eligible = True

    if age is not None and scheme.get("min_age") is not None and age < scheme["min_age"]:
        eligible = False; reasons.append(f"Age is below minimum {scheme['min_age']} years.")
    if age is not None and scheme.get("max_age") is not None and age > scheme["max_age"]:
        eligible = False; reasons.append(f"Age is above maximum {scheme['max_age']} years.")
    if income is not None and scheme.get("max_income") is not None and income > scheme["max_income"]:
        eligible = False; reasons.append("Annual family income exceeds the scheme limit.")
    required_occ = (scheme.get("occupation") or "").strip().lower()
    if required_occ and required_occ not in occupation and occupation not in required_occ:
        eligible = False; reasons.append(f"Occupation does not match required occupation: {scheme['occupation']}.")

    if eligible:
        reasons.append("Profile satisfies the available database eligibility rules.")
    return {"eligible": eligible, "reasons": reasons, "status": "VERIFIED"}


def match_schemes(profile: dict[str, Any], schemes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if os.getenv("ENGINE_MODE", "mock").lower() == "real":
        _, fn = _real_engine()
        return fn(profile, schemes)

    scored = []
    for s in schemes:
        result = check_eligibility(profile, s)
        score = 100.0 if result["eligible"] else 25.0
        scored.append({"scheme_id": s["id"], "match_score": score, "reasons": result["reasons"], "eligible": result["eligible"]})
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    for i, item in enumerate(scored, start=1):
        item["rank"] = i
    return scored
