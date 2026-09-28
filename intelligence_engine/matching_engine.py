"""Matching engine: deterministic fit scoring and ranking.

Answers "How strongly does this scheme fit the person's needs?"
It does NOT answer eligibility ("Can this person satisfy the
mandatory requirements?") — that belongs to EligibilityEngine.

Scoring model (total 100, weights are class constants):
- purpose match (supported_purposes): 35
- project_type match (supported_project_types): 25
- occupation alignment (eligibility_rules occupations): 15
- state applicability (eligibility_rules states): 10
- social-category alignment (eligibility_rules categories): 10
- education alignment (eligibility_rules min_education): 5

A dimension counts toward the score only when BOTH hold:
(a) the scheme defines it (non-empty list / non-empty rule), and
(b) the user provided a value for it.
Otherwise the dimension is skipped (not penalized) and the reason
says why it was not used. Final score = 100 * earned / applicable,
rounded to 2 decimals. No applicable dimensions -> 0.0.

Eligibility separation: score_scheme() measures pure fit and never
overrides eligibility. rank_schemes() ranks ONLY schemes whose
EligibilityResult status is "eligible". "not_eligible" schemes are
excluded (never a confirmed recommendation); "needs_information"
schemes are excluded too — a provisional score would need a status
field MatchResult does not have, so unresolved schemes stay out of
ranked_matches and remain visible via eligibility_results instead.

The score is an internal ranking signal, NOT a probability of
approval, benefit receipt, government score, or "percentage
eligibility". No LLM, database, FastAPI, embeddings, or randomness.
"""

from intelligence_engine.schemas import (
    EligibilityResult,
    MatchResult,
    Scheme,
    UserProfile,
)


class MatchingEngine:
    """Deterministic fit scorer and ranker for eligible schemes."""

    PURPOSE_WEIGHT: float = 35.0
    PROJECT_TYPE_WEIGHT: float = 25.0
    OCCUPATION_WEIGHT: float = 15.0
    STATE_WEIGHT: float = 10.0
    CATEGORY_WEIGHT: float = 10.0
    EDUCATION_WEIGHT: float = 5.0

    def score_scheme(self, profile: UserProfile, scheme: Scheme) -> tuple[float, list[str]]:
        """Compute the 0-100 fit score and reasons for one scheme.

        Args:
            profile: Validated user profile (missing fields are None).
            scheme: Scheme with supported lists and eligibility_rules.

        Returns:
            Tuple of (score, reasons). Pure fit signal; ignores eligibility.
        """
        earned = 0.0
        applicable = 0.0
        reasons: list[str] = []

        # Purpose (most important: what the citizen needs support for).
        if scheme.supported_purposes and profile.purpose is not None:
            applicable += self.PURPOSE_WEIGHT
            if self._matches(profile.purpose, scheme.supported_purposes):
                earned += self.PURPOSE_WEIGHT
                reasons.append(
                    f"Purpose '{profile.purpose}' matches the scheme's supported purposes: "
                    f"{', '.join(scheme.supported_purposes)}."
                )
            else:
                reasons.append(
                    f"Purpose '{profile.purpose}' does not match the scheme's supported purposes: "
                    f"{', '.join(scheme.supported_purposes)}."
                )
        else:
            reasons.append(self._skipped_reason("Purpose", scheme.supported_purposes, profile.purpose))

        # Project type.
        if scheme.supported_project_types and profile.project_type is not None:
            applicable += self.PROJECT_TYPE_WEIGHT
            if self._matches(profile.project_type, scheme.supported_project_types):
                earned += self.PROJECT_TYPE_WEIGHT
                reasons.append(
                    f"Project type '{profile.project_type}' matches the scheme's supported project types: "
                    f"{', '.join(scheme.supported_project_types)}."
                )
            else:
                reasons.append(
                    f"Project type '{profile.project_type}' does not match the scheme's supported project types: "
                    f"{', '.join(scheme.supported_project_types)}."
                )
        else:
            reasons.append(
                self._skipped_reason("Project type", scheme.supported_project_types, profile.project_type)
            )

        # Occupation (only when the scheme defines occupations).
        occupations = scheme.eligibility_rules.get("occupations")
        if self._is_non_empty_str_list(occupations) and profile.occupation is not None:
            applicable += self.OCCUPATION_WEIGHT
            assert isinstance(occupations, list)
            if self._matches(profile.occupation, occupations):
                earned += self.OCCUPATION_WEIGHT
                reasons.append(
                    f"Occupation '{profile.occupation}' aligns with the scheme's supported occupations: "
                    f"{', '.join(occupations)}."
                )
            else:
                reasons.append(
                    f"Occupation '{profile.occupation}' does not align with the scheme's supported occupations: "
                    f"{', '.join(occupations)}."
                )
        else:
            reasons.append(self._skipped_reason("Occupation", occupations, profile.occupation))

        # State (only when the scheme defines states).
        states = scheme.eligibility_rules.get("states")
        if self._is_non_empty_str_list(states) and profile.state is not None:
            applicable += self.STATE_WEIGHT
            assert isinstance(states, list)
            if self._matches(profile.state, states):
                earned += self.STATE_WEIGHT
                reasons.append(
                    f"State '{profile.state}' matches the scheme's applicable states: {', '.join(states)}."
                )
            else:
                reasons.append(
                    f"State '{profile.state}' does not match the scheme's applicable states: {', '.join(states)}."
                )
        else:
            reasons.append(self._skipped_reason("State", states, profile.state))

        # Social category (only when the scheme defines categories).
        categories = scheme.eligibility_rules.get("categories")
        if self._is_non_empty_str_list(categories) and profile.social_category is not None:
            applicable += self.CATEGORY_WEIGHT
            assert isinstance(categories, list)
            if self._matches(profile.social_category, categories):
                earned += self.CATEGORY_WEIGHT
                reasons.append(
                    f"Social category '{profile.social_category}' aligns with the scheme's supported categories: "
                    f"{', '.join(categories)}."
                )
            else:
                reasons.append(
                    f"Social category '{profile.social_category}' does not align with the scheme's supported "
                    f"categories: {', '.join(categories)}."
                )
        else:
            reasons.append(self._skipped_reason("Social category", categories, profile.social_category))

        # Education (only when the scheme defines min_education).
        min_education = scheme.eligibility_rules.get("min_education")
        if (
            isinstance(min_education, str)
            and min_education.strip()
            and profile.education_level is not None
        ):
            applicable += self.EDUCATION_WEIGHT
            if profile.education_level.strip().lower() == min_education.strip().lower():
                earned += self.EDUCATION_WEIGHT
                reasons.append(
                    f"Education '{profile.education_level}' aligns with the scheme's education criterion "
                    f"'{min_education}'."
                )
            else:
                reasons.append(
                    f"Education '{profile.education_level}' does not align with the scheme's education criterion "
                    f"'{min_education}'."
                )
        else:
            reasons.append(self._skipped_reason("Education", min_education, profile.education_level))

        if applicable == 0:
            reasons.append("No matching dimensions apply: the scheme defines no comparable criteria.")
            return 0.0, reasons
        return round(100.0 * earned / applicable, 2), reasons

    def rank_schemes(
        self,
        profile: UserProfile,
        schemes: list[Scheme],
        eligibility_results: list[EligibilityResult] | None = None,
    ) -> list[MatchResult]:
        """Rank schemes by fit score, restricted to confirmed-eligible ones.

        Args:
            profile: Validated user profile.
            schemes: Candidate schemes.
            eligibility_results: Eligibility outcomes keyed by scheme_id.
                When provided, only status "eligible" schemes are ranked.
                "not_eligible" and "needs_information" schemes are excluded
                (never presented as confirmed recommendations). When None,
                all schemes are scored and ranked (caller takes responsibility).

        Returns:
            Ranked MatchResult list (rank 1..N). Ties broken by scheme_id.
        """
        eligible_ids: set[str] | None = None
        if eligibility_results is not None:
            eligible_ids = {r.scheme_id for r in eligibility_results if r.status == "eligible"}

        scored: list[tuple[str, float, list[str]]] = []
        for scheme in schemes:
            if eligible_ids is not None and scheme.id not in eligible_ids:
                continue
            score, reasons = self.score_scheme(profile, scheme)
            scored.append((scheme.id, score, reasons))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            MatchResult(scheme_id=scheme_id, score=score, rank=rank, reasons=reasons)
            for rank, (scheme_id, score, reasons) in enumerate(scored, start=1)
        ]

    @staticmethod
    def _matches(user_value: str, allowed: list[str]) -> bool:
        """Case-insensitive membership check."""
        return user_value.strip().lower() in [v.strip().lower() for v in allowed]

    @staticmethod
    def _is_non_empty_str_list(value: object) -> bool:
        """True when value is a non-empty list of strings."""
        return isinstance(value, list) and len(value) > 0 and all(isinstance(v, str) for v in value)

    @staticmethod
    def _skipped_reason(label: str, scheme_value: object, user_value: object) -> str:
        """Explain why a dimension was not used (no penalty either way)."""
        if isinstance(scheme_value, list) and len(scheme_value) == 0:
            return f"{label} was not used because the scheme does not specify it."
        if scheme_value is None or (isinstance(scheme_value, str) and not scheme_value.strip()):
            return f"{label} was not used because the scheme does not specify it."
        return f"{label} was not used because user information is missing."
