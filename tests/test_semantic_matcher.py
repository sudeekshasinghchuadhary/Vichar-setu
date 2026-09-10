"""Tests for the semantic matching layer (deterministic fake, offline)."""

import math

import pytest

from intelligence_engine.schemas import Scheme, UserProfile
from intelligence_engine.semantic_matcher import (
    EmbeddingProvider,
    SemanticMatcher,
    SemanticMatcherError,
    build_scheme_text,
    build_user_text,
    combine_scores,
    cosine_similarity,
)
from tests.fixtures import make_scheme


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic test-only provider: keyword counts over a fixed vocabulary.

    Related texts sharing vocabulary produce similar vectors; unrelated
    texts do not. Implements EmbeddingProvider so a real backend can
    replace it without changing SemanticMatcher.
    """

    VOCABULARY: tuple[str, ...] = (
        "food",
        "processing",
        "business",
        "micro",
        "enterprise",
        "farm",
        "loan",
    )

    def embed(self, text: str) -> list[float]:
        """Return keyword-count vector for the text (deterministic)."""
        tokens = [token.strip(".,!?;:()\"'").lower() for token in text.lower().split()]
        return [float(tokens.count(word)) for word in self.VOCABULARY]


def test_identical_vectors_maximum_similarity() -> None:
    """Identical vectors produce cosine 1.0."""
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_neutral_similarity() -> None:
    """Orthogonal vectors produce cosine 0.0."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_zero_vectors_handled_safely() -> None:
    """Zero vectors raise instead of returning a misleading score."""
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([1.0, 2.0], [0.0, 0.0])


def test_different_dimensions_rejected() -> None:
    """Vectors with different dimensions are rejected."""
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_malformed_vectors_rejected() -> None:
    """Empty, non-numeric, or non-finite vectors are rejected."""
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([], [1.0])
    with pytest.raises(SemanticMatcherError):
        cosine_similarity("ab", [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([1.0, "x"], [1.0, 2.0])  # type: ignore[list-item]
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([1.0, math.nan], [1.0, 2.0])
    with pytest.raises(SemanticMatcherError):
        cosine_similarity([1.0, math.inf], [1.0, 2.0])


def test_fake_embeddings_are_deterministic() -> None:
    """Same text always yields the same fake vector and semantic score."""
    provider = FakeEmbeddingProvider()
    assert provider.embed("food processing business") == provider.embed("food processing business")
    matcher = SemanticMatcher(provider)
    profile = UserProfile(purpose="food processing business")
    first = matcher.score(profile, make_scheme())
    second = matcher.score(profile, make_scheme())
    assert first == second


def test_similar_text_scores_stronger_than_unrelated() -> None:
    """Related need language outscores unrelated text under the fake provider."""
    matcher = SemanticMatcher(FakeEmbeddingProvider())
    scheme = Scheme(
        id="s-food",
        name="Food scheme",
        description="Financial assistance for micro-enterprises involved in food processing.",
        supported_purposes=["food processing business"],
        supported_project_types=["micro-enterprise"],
    )
    related = matcher.score(UserProfile(purpose="start a small food processing business"), scheme)
    unrelated = matcher.score(UserProfile(purpose="stellar astronomy research"), scheme)
    assert related.score > unrelated.score
    assert 0 <= unrelated.score <= related.score <= 100


def test_embedded_text_excludes_hard_gates() -> None:
    """Age/income never enter the embedded text on either side."""
    user_text = build_user_text(UserProfile(age=40, annual_family_income=999999, purpose="tailoring"))
    scheme = make_scheme()
    scheme_text = build_scheme_text(scheme)
    assert "40" not in user_text and "999999" not in user_text
    assert "300000" not in scheme_text and "18" not in scheme_text.split()


def test_hybrid_combines_without_breaking_deterministic_scale() -> None:
    """Hybrid stays 0-100; weight 0 keeps deterministic, 1 keeps semantic."""
    assert combine_scores(80.0, 60.0, semantic_weight=0.0) == 80.0
    assert combine_scores(80.0, 60.0, semantic_weight=1.0) == 60.0
    assert combine_scores(80.0, 60.0) == pytest.approx(74.0)
    with pytest.raises(SemanticMatcherError):
        combine_scores(120.0, 60.0)


def test_semantic_layer_separate_from_eligibility() -> None:
    """Semantic matcher holds no eligibility logic and imports no LLM/DB/API."""
    import intelligence_engine.semantic_matcher as semantic_module

    matcher = SemanticMatcher(FakeEmbeddingProvider())
    assert not hasattr(matcher, "eligibility")
    assert "EligibilityEngine" not in dir(semantic_module)
    assert "LLMClient" not in dir(semantic_module)
    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi", "openai"):
        assert forbidden not in dir(semantic_module)


def test_no_external_embedding_provider_required() -> None:
    """Tests run on the fake provider; the abstraction needs no network or key."""
    assert issubclass(FakeEmbeddingProvider, EmbeddingProvider)
    assert FakeEmbeddingProvider().embed("food business") != [0.0] * len(
        FakeEmbeddingProvider.VOCABULARY
    )
