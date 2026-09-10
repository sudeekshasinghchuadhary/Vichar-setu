"""Semantic matching layer: embeddings + cosine similarity (fit only).

Sits BESIDE the deterministic MatchingEngine, never above eligibility:

    EligibilityEngine  ->  EligibilityResult (mandatory gates, unchanged)
    MatchingEngine     ->  deterministic 0-100 fit (exact wording, unchanged)
    SemanticMatcher    ->  semantic 0-100 fit (similar meanings, this module)
    combine_scores()   ->  hybrid 0-100 fit (weighted mix, ranking only)

Text embedded (need language only, never hard gates):
- user text: purpose + project_type + occupation.
- scheme text: supported_purposes + supported_project_types + description.
Age, income, and other numeric criteria are deliberately excluded.

The semantic score measures closeness between the user's need and
scheme information. It is NOT eligibility, approval/benefit
probability, or an official government score.

Stdlib only (math). No vendor SDK, network, database, or FastAPI.
Real embedding providers implement EmbeddingProvider; tests use a
deterministic fake defined in the test module.
"""

import math
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from intelligence_engine.schemas import Scheme, UserProfile


class SemanticMatcherError(ValueError):
    """Raised for invalid vectors, texts, or scores (never silent)."""


class EmbeddingProvider(ABC):
    """Abstract embedding backend. Real vendors implement this interface."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for non-empty text.

        Args:
            text: Non-empty input text.

        Returns:
            Non-empty list of finite floats.
        """
        raise NotImplementedError


class SemanticScore(BaseModel):
    """Structured semantic result (no natural-language claims).

    Pydantic model so scores crossing the future Intelligence API
    boundary serialize cleanly via model_dump()/model_dump_json().
    """

    score: float = Field()
    cosine: float = Field()
    user_text: str = Field()
    scheme_text: str = Field()


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity in [-1, 1] for two validated vectors.

    Raises:
        SemanticMatcherError: For zero vectors, dimension mismatch,
            empty inputs, or non-finite/non-numeric entries.
    """
    _validate_vector(first, "first")
    _validate_vector(second, "second")
    if len(first) != len(second):
        raise SemanticMatcherError(
            f"Vector dimension mismatch: {len(first)} vs {len(second)}."
        )
    dot = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(a * a for a in first))
    second_norm = math.sqrt(sum(b * b for b in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise SemanticMatcherError("Zero vectors have no defined cosine similarity.")
    return dot / (first_norm * second_norm)


def _validate_vector(vector: object, label: str) -> None:
    """Reject empty, non-list, or non-finite/non-numeric vectors."""
    if not isinstance(vector, list) or len(vector) == 0:
        raise SemanticMatcherError(f"{label} vector must be a non-empty list of floats.")
    for entry in vector:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise SemanticMatcherError(f"{label} vector must contain only numbers.")
        if not math.isfinite(entry):
            raise SemanticMatcherError(f"{label} vector must contain only finite numbers.")


def build_user_text(profile: UserProfile) -> str:
    """Join the user's need-oriented fields (purpose, project type, occupation)."""
    parts = [profile.purpose, profile.project_type, profile.occupation]
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_scheme_text(scheme: Scheme) -> str:
    """Join the scheme's need-oriented fields (purposes, project types, description)."""
    parts = [*scheme.supported_purposes, *scheme.supported_project_types, scheme.description]
    return " ".join(part.strip() for part in parts if part and part.strip())


def combine_scores(deterministic_score: float, semantic_score: float, semantic_weight: float = 0.3) -> float:
    """Mix deterministic and semantic 0-100 scores into a hybrid 0-100 score.

    hybrid = (1 - semantic_weight) * deterministic + semantic_weight * semantic.
    The default 0.3 keeps the explainable deterministic signal dominant;
    it is a prototype default, easy to change, not a tuned constant.
    """
    for label, value in (("deterministic_score", deterministic_score), ("semantic_score", semantic_score)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SemanticMatcherError(f"{label} must be a finite number.")
        if not 0 <= value <= 100:
            raise SemanticMatcherError(f"{label} must be within 0-100.")
    if (
        isinstance(semantic_weight, bool)
        or not isinstance(semantic_weight, (int, float))
        or not math.isfinite(semantic_weight)
        or not 0 <= semantic_weight <= 1
    ):
        raise SemanticMatcherError("semantic_weight must be a finite number within 0-1.")
    return round((1 - semantic_weight) * deterministic_score + semantic_weight * semantic_score, 2)


class SemanticMatcher:
    """Score need-text similarity between a profile and a scheme."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Store the injected embedding backend (no vendor coupling)."""
        self.provider = provider

    def score(self, profile: UserProfile, scheme: Scheme) -> SemanticScore:
        """Embed both texts and return the 0-100 semantic fit.

        score = round(max(0, cosine) * 100, 2): identical direction -> 100,
        orthogonal/unrelated -> 0, opposites clamp to 0 (fit has no
        negative direction). Empty user/scheme text -> 0.0 without
        calling the provider (no signal to compare).
        """
        user_text = build_user_text(profile)
        scheme_text = build_scheme_text(scheme)
        if not user_text or not scheme_text:
            return SemanticScore(score=0.0, cosine=0.0, user_text=user_text, scheme_text=scheme_text)
        user_vector = self.provider.embed(user_text)
        scheme_vector = self.provider.embed(scheme_text)
        if not any(value != 0 for value in user_vector) or not any(
            value != 0 for value in scheme_vector
        ):
            # No shared signal with the provider's vocabulary: zero overlap, not an error.
            return SemanticScore(score=0.0, cosine=0.0, user_text=user_text, scheme_text=scheme_text)
        cosine = cosine_similarity(user_vector, scheme_vector)
        return SemanticScore(
            score=round(max(0.0, cosine) * 100, 2),
            cosine=cosine,
            user_text=user_text,
            scheme_text=scheme_text,
        )
