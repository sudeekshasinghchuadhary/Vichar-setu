"""Optional manual integration test for real embeddings (NOT part of CI).

Runs only when RUN_EMBEDDING_INTEGRATION=1 AND sentence-transformers is
installed with the model available. Never runs by default, contains no
credentials, and never affects the offline unit suite.
"""

import os

import pytest

from intelligence_engine.schemas import UserProfile
from intelligence_engine.semantic_matcher import SemanticMatcher, cosine_similarity

RUN_INTEGRATION = os.environ.get("RUN_EMBEDDING_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION, reason="Set RUN_EMBEDDING_INTEGRATION=1 to run real-model checks."
)


def _provider() -> object:
    """Import lazily so collection never requires the optional dependency."""
    pytest.importorskip("sentence_transformers")
    from intelligence_engine.local_embedding_provider import LocalEmbeddingProvider

    return LocalEmbeddingProvider()


def test_real_model_related_outscores_unrelated() -> None:
    """Sanity check (manual only): related EN/Hindi need-text beats unrelated text."""
    from tests.fixtures import make_scheme

    matcher = SemanticMatcher(_provider())  # type: ignore[arg-type]
    scheme = make_scheme()
    english = matcher.score(
        UserProfile(purpose="I want financial support to start a small business"), scheme
    )
    hindi = matcher.score(
        UserProfile(purpose="मुझे छोटा व्यवसाय शुरू करने के लिए आर्थिक सहायता चाहिए।"), scheme
    )
    unrelated = matcher.score(UserProfile(purpose="stellar astronomy research"), scheme)
    assert english.score > unrelated.score
    assert hindi.score > unrelated.score


def test_real_model_output_is_stable_shape() -> None:
    """Two embeds share one dimension and self-similarity is ~1.0."""
    provider = _provider()
    first = provider.embed("small business support")  # type: ignore[attr-defined]
    second = provider.embed("small business support")  # type: ignore[attr-defined]
    assert len(first) == len(second) > 0
    assert cosine_similarity(first, second) == pytest.approx(1.0)
