"""Adapter between Raj/Anusha's FastAPI backend and Ananya's intelligence engine.

Uses the real intelligence_engine package through
IntelligenceOrchestrator.run_from_profile():

- Backend dict inputs are validated into UserProfile / Scheme with
  Pydantic. Only fields actually present are mapped; missing data is
  never invented (absent keys stay absent, explicit nulls stay null).
- Flat backend rule fields map to structured eligibility_rules:
    min_age -> min_age, max_age -> max_age, max_income -> max_annual_income,
    occupation -> occupations [single value],
    education -> min_education,
    category -> categories [single value],
    state -> states [single value].
  A rule is added only when its source value is present and non-empty.
- No LLM is required: run_from_profile() never touches the profile
  processor, so no API key or network is needed.
- All eligibility/matching decisions stay inside intelligence_engine;
  this module only translates shapes in and out.
- Response shapes match the existing FastAPI contracts:
    check_eligibility -> {"eligible", "reasons", "status"}
    match_schemes -> [{"scheme_id", "match_score", "reasons", "eligible", "rank"}]
  eligible is True only for status "eligible" (needs_information and
  not_eligible both map to False, conservatively). Non-ranked schemes
  keep score 0.0 with their eligibility reasons (never invented scores).
"""
from typing import Any

from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.schemas import Scheme, UserProfile

_PROFILE_FIELDS = set(UserProfile.model_fields)

# profile_processor=None: run_from_profile() never uses it, so the
# structured-profile path needs no LLM backend. Engines are stateless.
_ORCHESTRATOR = IntelligenceOrchestrator(profile_processor=None)


def _to_user_profile(profile: dict[str, Any]) -> UserProfile:
    """Validate backend profile dict; keep only known intelligence fields."""
    return UserProfile.model_validate({k: v for k, v in profile.items() if k in _PROFILE_FIELDS})


def _non_empty_str(value: Any) -> str | None:
    """Stripped string or None when absent/blank/non-string."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _to_scheme(scheme: dict[str, Any]) -> Scheme:
    """Translate a flat backend scheme dict into the intelligence Scheme."""
    rules: dict[str, Any] = {}
    if scheme.get("min_age") is not None:
        rules["min_age"] = scheme["min_age"]
    if scheme.get("max_age") is not None:
        rules["max_age"] = scheme["max_age"]
    if scheme.get("max_income") is not None:
        rules["max_annual_income"] = scheme["max_income"]
    occupation = _non_empty_str(scheme.get("occupation"))
    if occupation is not None:
        rules["occupations"] = [occupation]
    education = _non_empty_str(scheme.get("education"))
    if education is not None:
        rules["min_education"] = education
    category = _non_empty_str(scheme.get("category"))
    if category is not None:
        rules["categories"] = [category]
    state = _non_empty_str(scheme.get("state"))
    if state is not None:
        rules["states"] = [state]
    return Scheme(
        id=str(scheme["id"]),
        name=scheme.get("scheme_name") or "",
        description=scheme.get("description") or "",
        eligibility_rules=rules,
        state=state,
        category=category,
        benefit=_non_empty_str(scheme.get("benefit")),
    )


def check_eligibility(profile: dict[str, Any], scheme: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one backend profile dict against one backend scheme dict."""
    user = _to_user_profile(profile)
    target = _to_scheme(scheme)
    result = _ORCHESTRATOR.run_from_profile(user, [target])
    decision = result.schemes[0].eligibility
    return {
        "eligible": decision.status == "eligible",
        "reasons": list(decision.reasons),
        "status": "VERIFIED",
    }


def match_schemes(profile: dict[str, Any], schemes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank backend scheme dicts; ranked items carry engine scores and ranks."""
    user = _to_user_profile(profile)
    targets = [_to_scheme(item) for item in schemes]
    result = _ORCHESTRATOR.run_from_profile(user, targets)
    ranked = {match.scheme_id: match for match in result.ranked_matches}
    decided = {insight.scheme_id: insight.eligibility for insight in result.schemes}
    ordered: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for target, raw in zip(targets, schemes):
        match = ranked.get(target.id)
        if match is not None:
            ordered.append({
                "scheme_id": raw["id"],
                "match_score": match.score,
                "reasons": list(match.reasons),
                "eligible": True,
                "rank": match.rank,
            })
        else:
            decision = decided.get(target.id)
            reasons = list(decision.reasons) if decision is not None else ["No eligibility result."]
            leftovers.append({
                "scheme_id": raw["id"],
                "match_score": 0.0,
                "reasons": reasons,
                "eligible": False,
                "rank": 0,
            })
    ordered.sort(key=lambda item: item["rank"])
    items = ordered + leftovers
    for position, item in enumerate(items, start=1):
        item["rank"] = position
    return items
