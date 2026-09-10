"""Unit tests for LocalEmbeddingProvider (fully offline, no model download).

Nothing here imports sentence-transformers or touches the network.
The real model is exercised only by tests/test_real_embeddings.py,
which is skipped unless RUN_EMBEDDING_INTEGRATION=1.
"""

import sys

import pytest

from intelligence_engine.local_embedding_provider import (
    DEFAULT_MODEL_NAME,
    LocalEmbeddingProvider,
    validate_embedding,
)
from intelligence_engine.semantic_matcher import (
    EmbeddingProvider,
    SemanticMatcher,
    SemanticMatcherError,
)


def test_provider_implements_abstraction() -> None:
    """Concrete provider subclasses EmbeddingProvider with embed()."""
    assert issubclass(LocalEmbeddingProvider, EmbeddingProvider)
    assert callable(LocalEmbeddingProvider.embed)


def test_provider_returns_expected_shape_contract() -> None:
    """validate_embedding() enforces the non-empty finite-float contract."""
    assert validate_embedding([0.1, -2.0, 3]) == [0.1, -2.0, 3.0]


def test_empty_text_rejected_without_touching_model() -> None:
    """Empty input raises before any model load is attempted."""
    provider = LocalEmbeddingProvider()
    with pytest.raises(SemanticMatcherError):
        provider.embed("   ")
    assert provider._model is None


def test_malformed_model_output_rejected() -> None:
    """Empty, non-numeric, or non-finite outputs raise (no silent fallback)."""
    with pytest.raises(SemanticMatcherError):
        validate_embedding([])
    with pytest.raises(SemanticMatcherError):
        validate_embedding([1.0, "x"])
    with pytest.raises(SemanticMatcherError):
        validate_embedding([1.0, float("nan")])
    with pytest.raises(SemanticMatcherError):
        validate_embedding("not-a-vector")


def test_inconsistent_dimensions_rejected() -> None:
    """A provider emitting changing dimensions fails clearly on second call."""
    provider = LocalEmbeddingProvider()
    provider._encode = lambda text: [0.1, 0.2]  # type: ignore[method-assign]
    assert provider.embed("hello") == [0.1, 0.2]
    provider._encode = lambda text: [0.1, 0.2, 0.3]  # type: ignore[method-assign]
    with pytest.raises(SemanticMatcherError):
        provider.embed("hello again")


def test_semantic_matcher_accepts_real_provider_shape() -> None:
    """SemanticMatcher works through the abstraction with a stubbed backend."""
    provider = LocalEmbeddingProvider()
    provider._encode = lambda text: [1.0, 0.0, 0.0]  # type: ignore[method-assign]
    matcher = SemanticMatcher(provider)
    from intelligence_engine.schemas import UserProfile

    from tests.fixtures import make_scheme

    result = matcher.score(UserProfile(purpose="food business"), make_scheme())
    assert result.score == pytest.approx(100.0 * max(0.0, result.cosine))


def test_importing_provider_loads_no_model() -> None:
    """Import stays lightweight: no sentence-transformers, no instance state."""
    assert "sentence_transformers" not in sys.modules
    provider = LocalEmbeddingProvider(model_name="unused-in-this-test")
    assert provider._model is None
    assert provider.model_name == "unused-in-this-test"


def test_default_model_and_env_override() -> None:
    """Default model documented; env var overrides without code changes."""
    assert LocalEmbeddingProvider().model_name == DEFAULT_MODEL_NAME
    import os

    os.environ["VICHAR_EMBEDDING_MODEL"] = "custom-model"
    try:
        assert LocalEmbeddingProvider().model_name == "custom-model"
    finally:
        del os.environ["VICHAR_EMBEDDING_MODEL"]
