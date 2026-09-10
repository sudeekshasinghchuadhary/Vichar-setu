"""Local multilingual embedding provider (real backend, optional dependency).

Model: paraphrase-multilingual-MiniLM-L12-v2 via sentence-transformers.
- 384-dimensional sentence embeddings, CPU feasible.
- Multilingual (50+ languages): documented English and Hindi
  (Devanagari) support. Hinglish (romanized Hindi/English code-mix)
  has NO officially guaranteed performance — it often works partially
  through shared English/code-switched tokens, but treat Hinglish
  similarity as best-effort and document that limitation to judges.
- Runs fully locally after a one-time model download: no API key,
  no per-request internet, no per-call cost.

Costs (approximate): sentence-transformers + CPU torch is several
hundred MB installed; the model download is roughly half a GB.
Only needed where real embeddings run — unit tests use the fake
provider and never touch this module's model.

Usage:
    provider = LocalEmbeddingProvider()  # lightweight, loads nothing
    vector = provider.embed("...")       # loads model once, then caches

Model name override (no secrets involved):
    VICHAR_EMBEDDING_MODEL=other-model-name

Embeddings improve semantic fit only. They do NOT determine
eligibility, replace rules, or prove qualification.
"""

import math
import os
from typing import Any

from intelligence_engine.semantic_matcher import EmbeddingProvider, SemanticMatcherError

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_ENV_VAR = "VICHAR_EMBEDDING_MODEL"


def validate_embedding(output: Any) -> list[float]:
    """Validate raw model output into a clean vector of finite floats.

    Raises:
        SemanticMatcherError: For empty, ragged, non-numeric, or
            non-finite outputs. Never silently coerces bad data.
    """
    if isinstance(output, list):
        values = output
    elif hasattr(output, "tolist"):
        values = output.tolist()
    else:
        raise SemanticMatcherError(
            f"Embedding output must be a list of floats, got {type(output).__name__}."
        )
    if len(values) == 0:
        raise SemanticMatcherError("Embedding output must be a non-empty vector.")
    cleaned: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise SemanticMatcherError("Embedding output must contain only numbers.")
        if not math.isfinite(entry):
            raise SemanticMatcherError("Embedding output must contain only finite numbers.")
        cleaned.append(float(entry))
    return cleaned


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers backend with lazy model loading.

    Importing this module never imports sentence-transformers and never
    downloads a model. The model loads once on the first embed() call.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Store config only; no model loading, no network, no key.

        Args:
            model_name: HuggingFace model id. Defaults to
                DEFAULT_MODEL_NAME or VICHAR_EMBEDDING_MODEL when set.
        """
        self.model_name = model_name or os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)
        self._model: Any = None
        self._dimension: int | None = None

    def embed(self, text: str) -> list[float]:
        """Embed non-empty text into a validated vector of finite floats.

        Raises:
            SemanticMatcherError: For empty text, missing/unloadable
                model, or malformed model output. No silent fallback.
        """
        if not isinstance(text, str) or not text.strip():
            raise SemanticMatcherError("embed() requires non-empty text.")
        vector = validate_embedding(self._encode(text.strip()))
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise SemanticMatcherError(
                f"Unexpected embedding dimension {len(vector)} (expected {self._dimension})."
            )
        return vector

    def _encode(self, text: str) -> Any:
        """Load the model lazily on first use, then encode one text."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SemanticMatcherError(
                    "sentence-transformers is not installed. Install the optional "
                    "embedding dependency to use LocalEmbeddingProvider; unit tests "
                    "use the fake provider and do not need it."
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise SemanticMatcherError(
                    f"Could not load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model.encode(text, normalize_embeddings=False)
