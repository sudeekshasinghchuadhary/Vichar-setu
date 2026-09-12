"""Tests for Near-Miss semantic relevance enrichment (offline, fakes only).

Measurement only: relevance never influences eligibility, Near-Miss
qualification, ranking, or surfacing. No thresholds, no floors.
"""

from typing import Any

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.near_miss_engine import NearMissEngine
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.schemas import (
    RepresentationQuality,
    Scheme,
    SupportNeed,
    UserProfile,
)
from intelligence_engine.semantic_matcher import (
    EmbeddingProvider,
    SemanticMatcher,
    assess_representation,
)


class KeywordEmbeddings(EmbeddingProvider):
    """Deterministic keyword-count vectors (test-only)."""

    VOCABULARY = ("food", "business", "micro", "enterprise", "farm", "loan")

    def __init__(self) -> None:
        """Track provider invocations per text."""
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        """Count vocabulary hits, recording the call."""
        self.calls.append(text)
        tokens = text.lower().split()
        return [float(tokens.count(word)) for word in self.VOCABULARY]


def _confirmed(profile: UserProfile, rules: dict[str, Any], supported: set[str]):
    """Analyze with test-only injected closeness support (always restored)."""
    import intelligence_engine.near_miss_engine as engine_module

    original = engine_module._VALIDATED_CLOSENESS_CRITERIA
    engine_module._VALIDATED_CLOSENESS_CRITERIA = frozenset(supported)
    try:
        scheme = Scheme(id="s", name="S", eligibility_rules=rules)
        engine = EligibilityEngine()
        return NearMissEngine().analyze(profile, scheme, engine.evaluate(profile, scheme))
    finally:
        engine_module._VALIDATED_CLOSENESS_CRITERIA = original


def _profile() -> UserProfile:
    """Profile with need-oriented wording."""
    return UserProfile(
        purpose="farm loan",
        project_type="micro enterprise",
        occupation="tailor",
    )


def _needs() -> list[SupportNeed]:
    """Single need with rich description."""
    return [SupportNeed(need_type="machinery", context="buy farm equipment on loan")]


def _relevant_scheme() -> Scheme:
    """NOT_ELIGIBLE scheme (single occupation failure) with rich description."""
    return Scheme(
        id="rel",
        name="Relevant",
        description="farm loans for micro enterprise purchase",
        supported_purposes=["farm loan"],
        supported_project_types=["micro enterprise"],
        eligibility_rules={"occupations": ["clerk"]},
    )


def test_confirmed_near_miss_gets_relevance() -> None:
    """A: confirmed candidate is enriched; evidence untouched."""
    provider = KeywordEmbeddings()
    result = _confirmed(
        UserProfile(occupation="tailor"),
        {"occupations": ["clerk"]},
        {"occupation"},
    )
    assert result.is_near_miss is True
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(provider),
    )
    enriched = orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs())
    assert enriched.relevance is not None
    assert 0.0 <= enriched.relevance.score <= 100.0
    assert enriched.is_near_miss is True
    assert enriched.failed_criteria == result.failed_criteria
    assert enriched.reasons == result.reasons


def test_non_near_miss_not_enriched() -> None:
    """B: failing-but-unconfirmed schemes get no enrichment and no provider call."""
    provider = KeywordEmbeddings()
    result = _confirmed(
        UserProfile(occupation="tailor", annual_family_income=900000),
        {"occupations": ["clerk"], "max_annual_income": 100000},
        set(),
    )
    assert result.is_near_miss is False
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(provider),
    )
    enriched = orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs())
    assert enriched.relevance is None
    assert provider.calls == []


def test_multiple_confirmed_candidates_independent() -> None:
    """C: each confirmed candidate enriched separately; no leakage."""
    provider = KeywordEmbeddings()
    first = _confirmed(UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"})
    second = _confirmed(UserProfile(occupation="baker"), {"occupations": ["clerk"]}, {"occupation"})
    assert first.is_near_miss and second.is_near_miss
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(provider),
    )
    scheme_b = _relevant_scheme().model_copy(update={"id": "rel-b", "description": "food processing"})
    enriched_first = orch._enrich_near_miss(first, _profile(), _relevant_scheme(), _needs())
    enriched_second = orch._enrich_near_miss(second, _profile(), scheme_b, _needs())
    assert enriched_first.relevance is not None
    assert enriched_second.relevance is not None
    assert enriched_first.relevance.scheme_text != enriched_second.relevance.scheme_text
    assert enriched_first.relevance.score != enriched_second.relevance.score


def test_enrichment_requires_finalized_state() -> None:
    """D: structural gate — enrichment without is_near_miss=True is a no-op."""
    provider = KeywordEmbeddings()
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(provider),
    )
    result = _confirmed(
        UserProfile(occupation="tailor", annual_family_income=900000),
        {"occupations": ["clerk"], "max_annual_income": 100000},
        set(),
    )
    assert result.is_near_miss is False
    assert orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs()).relevance is None
    assert provider.calls == []


def test_enrichment_preserves_eligibility_evidence() -> None:
    """E: eligibility verdicts, traces, and failure evidence are identical."""
    result = _confirmed(
        UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"}
    )
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(KeywordEmbeddings()),
    )
    enriched = orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs())
    assert enriched.is_near_miss == result.is_near_miss
    assert enriched.failed_criteria == result.failed_criteria
    assert enriched.satisfied_criteria == result.satisfied_criteria
    assert enriched.reasons == result.reasons


def test_enrichment_preserves_qualification() -> None:
    """F: qualification fields survive enrichment byte-identically."""
    result = _confirmed(
        UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"}
    )
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(KeywordEmbeddings()),
    )
    enriched = orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs())
    assert enriched.scheme_id == result.scheme_id
    assert enriched.total_criteria == result.total_criteria
    assert enriched.is_near_miss is True


def test_representation_quality_states() -> None:
    """G: EMPTY/SPARSE/RICH distinguishable per side from structure alone."""
    assert assess_representation(UserProfile(), [], _relevant_scheme()).user == "EMPTY"
    assert assess_representation(
        UserProfile(), [SupportNeed(need_type="machinery")], _relevant_scheme()
    ).user == "SPARSE"
    assert assess_representation(_profile(), _needs(), _relevant_scheme()).user == "RICH"
    assert assess_representation(
        _profile(),
        _needs(),
        Scheme(id="e", name="E", description="", supported_purposes=[], supported_project_types=[]),
    ).scheme == "EMPTY"
    assert assess_representation(
        _profile(),
        _needs(),
        Scheme(id="s2", name="S", supported_purposes=["farm loan"], supported_project_types=[]),
    ).scheme == "RICH"
    assert assess_representation(_profile(), _needs(), _relevant_scheme()).scheme == "RICH"
    quality = assess_representation(_profile(), _needs(), _relevant_scheme())
    assert isinstance(quality, RepresentationQuality)
    assert (quality.user, quality.scheme) == ("RICH", "RICH")


def test_canonical_need_type_only_is_sparse() -> None:
    """Canonical need_type only (no context, no profile text) -> SPARSE."""
    quality = assess_representation(
        UserProfile(), [SupportNeed(need_type="machinery")], _relevant_scheme()
    )
    assert quality.user == "SPARSE"
    assert quality.scheme == "RICH"


def test_unrecognized_need_type_only_is_rich() -> None:
    """Unrecognized/free-text need_type only (no context, no profile text) -> RICH."""
    quality = assess_representation(
        UserProfile(), [SupportNeed(need_type="buy sewing machines for tailoring")], _relevant_scheme()
    )
    assert quality.user == "RICH"
    assert quality.scheme == "RICH"


def test_free_text_scheme_purpose_without_description_is_rich() -> None:
    """Non-empty supported_purposes (free text) without description -> RICH."""
    scheme = Scheme(
        id="s3",
        name="S",
        description="",
        supported_purposes=["financial assistance for acquisition of modern production equipment"],
        supported_project_types=[],
    )
    quality = assess_representation(_profile(), _needs(), scheme)
    assert quality.scheme == "RICH"
    assert quality.user == "RICH"


def test_canonical_need_type_only_sparse_regression() -> None:
    """Regression: canonical need_type only remains SPARSE (existing behavior)."""
    quality = assess_representation(
        UserProfile(), [SupportNeed(need_type="working_capital")], _relevant_scheme()
    )
    assert quality.user == "SPARSE"
    quality2 = assess_representation(
        UserProfile(), [SupportNeed(need_type="training")], _relevant_scheme()
    )
    assert quality2.user == "SPARSE"
    quality3 = assess_representation(
        UserProfile(), [SupportNeed(need_type="marketing")], _relevant_scheme()
    )
    assert quality3.user == "SPARSE"
    quality4 = assess_representation(
        UserProfile(), [SupportNeed(need_type="infrastructure")], _relevant_scheme()
    )
    assert quality4.user == "SPARSE"


def test_zero_score_is_not_unavailable() -> None:
    """H: measured 0.0, unavailable None, and EMPTY stay three states."""
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(KeywordEmbeddings()),
    )
    unrelated = _relevant_scheme().model_copy(
        update={
            "id": "u",
            "description": "qqq zzz",
            "supported_purposes": ["yyy"],
            "supported_project_types": ["xxx"],
        }
    )
    unrelated_result = _confirmed(UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"})
    enriched = orch._enrich_near_miss(unrelated_result, _profile(), unrelated, _needs())
    assert enriched.relevance is not None
    assert enriched.relevance.score == 0.0
    assert enriched.representation_quality is not None
    empty_scheme = Scheme(id="e2", name="E")
    empty_result = _confirmed(UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"})
    empty_enriched = orch._enrich_near_miss(empty_result, UserProfile(), empty_scheme, [])
    assert empty_enriched.relevance is not None
    assert empty_enriched.relevance.score == 0.0
    assert empty_enriched.representation_quality.user == "EMPTY"


def test_provider_failure_keeps_run_intact() -> None:
    """I: failing provider yields relevance None; near-miss result intact."""

    class ExplodingProvider(EmbeddingProvider):
        """Always-raising provider."""

        def embed(self, text: str) -> list[float]:
            """Raise instead of embedding."""
            raise RuntimeError("boom")

    result = _confirmed(
        UserProfile(occupation="tailor"), {"occupations": ["clerk"]}, {"occupation"}
    )
    orch = IntelligenceOrchestrator(
        profile_processor=None,
        semantic_matcher=SemanticMatcher(ExplodingProvider()),
    )
    enriched = orch._enrich_near_miss(result, _profile(), _relevant_scheme(), _needs())
    assert enriched.relevance is None
    assert enriched.is_near_miss is True
    assert enriched.failed_criteria == result.failed_criteria
    assert enriched.reasons == result.reasons


def test_conservative_off_produces_no_enrichment_targets() -> None:
    """J: empty validation set means no confirmed candidates exist to enrich."""
    profile = UserProfile(occupation="tailor")
    scheme = Scheme(id="s", name="S", eligibility_rules={"occupations": ["clerk"]})
    engine = EligibilityEngine()
    result = NearMissEngine().analyze(profile, scheme, engine.evaluate(profile, scheme))
    assert result.is_near_miss is False
