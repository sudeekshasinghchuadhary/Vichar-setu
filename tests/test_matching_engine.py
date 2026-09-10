"""Tests for the deterministic matching engine (offline, no LLM/DB/API)."""

import pytest
from pydantic import ValidationError

from intelligence_engine.matching_engine import MatchingEngine
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.schemas import EligibilityResult, MatchResult, Scheme, UserProfile
from intelligence_engine.semantic_matcher import EmbeddingProvider, SemanticMatcher
from tests.fixtures import make_match_result, make_scheme


def test_match_result_accepts_scores_0_to_100() -> None:
    """MatchResult accepts scores from 0 to 100."""
    assert make_match_result().score == 85.0
    assert MatchResult(scheme_id="s", score=0, rank=1).score == 0
    assert MatchResult(scheme_id="s", score=100, rank=1).score == 100


def test_invalid_match_scores_are_rejected() -> None:
    """Invalid match scores outside 0-100 are rejected."""
    with pytest.raises(ValidationError):
        MatchResult(scheme_id="s", score=-1, rank=1)
    with pytest.raises(ValidationError):
        MatchResult(scheme_id="s", score=101, rank=1)


def _profile_for(scheme: Scheme) -> UserProfile:
    """Profile matching every fit dimension of the fixture scheme."""
    return UserProfile(
        purpose="business expansion",
        project_type="micro-enterprise",
        occupation="Farmer",
        state="Uttar Pradesh",
        social_category="OBC",
        education_level="12th",
    )


def test_strong_match_scores_higher_than_weak_match() -> None:
    """Strong purpose/project match outscores a weak/no match."""
    engine = MatchingEngine()
    strong, _ = engine.score_scheme(_profile_for(make_scheme()), make_scheme())
    weak_profile = UserProfile(purpose="unrelated need", project_type="other")
    weak, _ = engine.score_scheme(weak_profile, make_scheme())
    assert strong > weak
    assert strong == 100.0


def test_matching_is_deterministic() -> None:
    """Identical inputs always give identical scores and reasons."""
    engine = MatchingEngine()
    first = engine.score_scheme(_profile_for(make_scheme()), make_scheme())
    second = engine.score_scheme(_profile_for(make_scheme()), make_scheme())
    assert first == second


def test_scores_always_within_0_to_100() -> None:
    """Scores stay inside the 0-100 contract across varied inputs."""
    engine = MatchingEngine()
    profiles = [
        _profile_for(make_scheme()),
        UserProfile(),
        UserProfile(purpose="business expansion"),
        UserProfile(purpose="other", occupation="Student", state="Bihar"),
    ]
    for profile in profiles:
        score, _ = engine.score_scheme(profile, make_scheme())
        assert 0 <= score <= 100


def test_reasons_correspond_to_actual_conditions() -> None:
    """Reasons mention the dimensions actually evaluated."""
    score, reasons = MatchingEngine().score_scheme(_profile_for(make_scheme()), make_scheme())
    assert score == 100.0
    assert any("Purpose" in reason and "matches" in reason for reason in reasons)
    assert any("Project type" in reason and "matches" in reason for reason in reasons)


def test_undefined_dimension_does_not_penalize() -> None:
    """A scheme without an occupation rule scores full marks on purpose alone."""
    engine = MatchingEngine()
    minimal = Scheme(id="s-min", name="Minimal", supported_purposes=["business expansion"])
    score, reasons = engine.score_scheme(UserProfile(purpose="business expansion"), minimal)
    assert score == 100.0
    assert any("not used because the scheme does not specify it" in reason for reason in reasons)


def test_ineligible_schemes_are_not_ranked() -> None:
    """not_eligible schemes never appear as confirmed recommendations."""
    engine = MatchingEngine()
    profile = _profile_for(make_scheme())
    results = [
        EligibilityResult(scheme_id="scheme-001", status="eligible", reasons=["ok"]),
        EligibilityResult(scheme_id="scheme-002", status="not_eligible", reasons=["failed"]),
    ]
    schemes = [
        make_scheme(),
        make_scheme().model_copy(update={"id": "scheme-002", "name": "Other"}),
    ]
    ranked = engine.rank_schemes(profile, schemes, results)
    assert [match.scheme_id for match in ranked] == ["scheme-001"]
    assert ranked[0].rank == 1


def test_needs_information_is_not_mislabeled_eligible() -> None:
    """Unresolved schemes stay out of ranked_matches (visible via eligibility instead)."""
    engine = MatchingEngine()
    profile = _profile_for(make_scheme())
    results = [
        EligibilityResult(
            scheme_id="scheme-001",
            status="needs_information",
            reasons=["Age required."],
            missing_information=["age"],
        )
    ]
    ranked = engine.rank_schemes(profile, [make_scheme()], results)
    assert ranked == []


def test_deterministic_tie_break_by_scheme_id() -> None:
    """Equal scores order by scheme_id with sequential ranks."""
    engine = MatchingEngine()
    profile = UserProfile(purpose="business expansion")
    schemes = [
        Scheme(id="scheme-b", name="B", supported_purposes=["business expansion"]),
        Scheme(id="scheme-a", name="A", supported_purposes=["business expansion"]),
    ]
    results = [
        EligibilityResult(scheme_id="scheme-a", status="eligible", reasons=["ok"]),
        EligibilityResult(scheme_id="scheme-b", status="eligible", reasons=["ok"]),
    ]
    ranked = engine.rank_schemes(profile, schemes, results)
    assert [(match.scheme_id, match.rank) for match in ranked] == [
        ("scheme-a", 1),
        ("scheme-b", 2),
    ]
    assert ranked[0].score == ranked[1].score


def test_engine_has_no_llm_database_or_api_calls() -> None:
    """MatchingEngine depends on schemas only."""
    import intelligence_engine.matching_engine as engine_module

    assert "LLMClient" not in dir(engine_module)
    assert "llm_client" not in dir(engine_module)
    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi", "openai"):
        assert forbidden not in dir(engine_module)


class _HitEmbeddings(EmbeddingProvider):
    """Shared-vocabulary vectors: identical direction, cosine 1.0."""

    def embed(self, text: str) -> list[float]:
        """Return a fixed non-zero vector regardless of input."""
        return [1.0, 2.0, 3.0]


class _MissEmbeddings(EmbeddingProvider):
    """Empty-vocabulary vectors: zero overlap, semantic 0.0."""

    def embed(self, text: str) -> list[float]:
        """Return an all-zero vector regardless of input."""
        return [0.0, 0.0, 0.0]


def _full_match_scheme() -> Scheme:
    """Scheme matching every fit dimension for a full 100.0 deterministic score."""
    return Scheme(
        id="scheme-full",
        name="Full",
        description="support",
        supported_purposes=["business expansion"],
        supported_project_types=["micro-enterprise"],
        eligibility_rules={
            "occupations": ["Farmer"],
            "states": ["Uttar Pradesh"],
            "categories": ["OBC"],
            "min_education": "12th",
        },
    )


def _full_match_profile() -> UserProfile:
    """Profile matching every fit dimension of the full-match scheme."""
    return UserProfile(
        purpose="business expansion",
        project_type="micro-enterprise",
        occupation="Farmer",
        state="Uttar Pradesh",
        social_category="OBC",
        education_level="12th",
    )


def _orchestrator(**overrides) -> IntelligenceOrchestrator:
    """Orchestrator with stub processor (structured path needs none of it)."""
    parts = {"profile_processor": None}
    parts.update(overrides)
    return IntelligenceOrchestrator(**parts)


def test_deterministic_only_populates_deterministic_component() -> None:
    """Deterministic ranking carries deterministic_score, semantic stays None."""
    orch = _orchestrator()
    profile = _full_match_profile()
    result = orch.run_from_profile(profile, [_full_match_scheme()])
    assert len(result.ranked_matches) == 1
    match = result.ranked_matches[0]
    assert match.score == 100.0
    assert match.deterministic_score == 100.0
    assert match.semantic_score is None


def test_hybrid_populates_both_components_exact() -> None:
    """det 100 + sem 100 at weight 0.25 blends to exactly 100 with both kept."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(_HitEmbeddings()), semantic_weight=0.25
    )
    result = orch.run_from_profile(_full_match_profile(), [_full_match_scheme()])
    match = result.ranked_matches[0]
    assert match.deterministic_score == 100.0
    assert match.semantic_score == 100.0
    assert match.score == 100.0


def test_hybrid_partial_semantic_exact() -> None:
    """det 100 + sem 0 at weight 0.25 blends to exactly 75 with both kept."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(_MissEmbeddings()), semantic_weight=0.25
    )
    result = orch.run_from_profile(_full_match_profile(), [_full_match_scheme()])
    match = result.ranked_matches[0]
    assert match.deterministic_score == 100.0
    assert match.semantic_score == 0.0
    assert match.score == 75.0


def test_components_do_not_change_ranking() -> None:
    """Hybrid ordering matches deterministic ordering on the same inputs."""
    weak = Scheme(id="scheme-weak", name="Weak", supported_purposes=["other purpose"])
    profile = _full_match_profile()
    schemes = [_full_match_scheme(), weak]
    det_order = [
        m.scheme_id for m in _orchestrator().run_from_profile(profile, schemes).ranked_matches
    ]
    hybrid_order = [
        m.scheme_id
        for m in _orchestrator(
            semantic_matcher=SemanticMatcher(_HitEmbeddings()), semantic_weight=0.25
        ).run_from_profile(profile, schemes).ranked_matches
    ]
    assert det_order == hybrid_order == ["scheme-full", "scheme-weak"]
