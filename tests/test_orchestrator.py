"""Integration tests for the IntelligenceOrchestrator (fakes only, offline)."""

from typing import Any

import pytest

from intelligence_engine.eligibility_engine import EligibilityEngineError
from intelligence_engine.clarification_generator import ClarificationGenerator
from intelligence_engine.explanation_generator import ExplanationGenerator
from intelligence_engine.llm_client import LLMClient, LLMExplanationProvider, NeedExtractor
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.schemas import (
    ApplicationStep,
    ChannelInfo,
    IntelligenceResult,
    Scheme,
    SchemeSupport,
    UserProfile,
)
from intelligence_engine.semantic_matcher import EmbeddingProvider, SemanticMatcher


class FakeLLM(LLMClient):
    """Canned profile extraction."""

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return a fixed eligible profile payload."""
        return {
            "age": 25,
            "annual_family_income": 180000.0,
            "occupation": "Farmer",
            "education_level": "Graduate",
            "social_category": "OBC",
            "state": "Uttar Pradesh",
            "purpose": "business expansion",
            "project_type": "micro-enterprise",
        }


class FakeNeeds(NeedExtractor):
    """Canned need extraction."""

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return a fixed machinery need."""
        return {
            "business_goal": "Expand tailoring business",
            "needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}],
        }


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic keyword-count vectors."""

    VOCABULARY = ("food", "business", "micro", "enterprise", "farm", "loan")

    def embed(self, text: str) -> list[float]:
        """Count vocabulary hits."""
        tokens = text.lower().split()
        return [float(tokens.count(word)) for word in self.VOCABULARY]


class EchoExplanations(LLMExplanationProvider):
    """Faithful echo of trace facts."""

    def generate_explanation(self, prompt: str) -> str:
        """Render facts without instructions."""
        return prompt.split("\n\nInstructions:")[0] + "\nStated with unknown missing."


class BadExplanations(LLMExplanationProvider):
    """Guarantee-laden draft forcing fallback."""

    def generate_explanation(self, prompt: str) -> str:
        """Return an ungrounded draft."""
        return "Guaranteed approval for everyone."


def _support(**overrides) -> SchemeSupport:
    """One-time machinery loan support."""
    data = {
        "support_type": "loan",
        "covers_need_types": ["machinery"],
        "max_amount": 500000,
        "max_amount_period": "one_time",
    }
    data.update(overrides)
    return SchemeSupport(**data)


def _open_scheme() -> Scheme:
    """Eligible scheme with support, docs, steps, link, and channel."""
    return Scheme(
        id="scheme-open",
        name="Open support",
        description="Financial assistance for micro-enterprises expanding business.",
        supported_purposes=["business expansion"],
        supported_project_types=["micro-enterprise"],
        eligibility_rules={
            "min_age": 18,
            "max_annual_income": 300000,
            "occupations": ["Farmer"],
            "min_education": "12th",
            "categories": ["OBC"],
            "states": ["Uttar Pradesh"],
        },
        support_options=[_support()],
        documents_required=["Aadhaar", "Income certificate"],
        application_link="https://example.com/apply",
        application_steps=[ApplicationStep(step_type="submit_application", title="Submit online")],
        application_channels=[ChannelInfo(name="District office", channel_type="offline")],
    )


def _strict_scheme() -> Scheme:
    """Scheme failing only on income (320000 fake? no: 180000 income vs 100000 cap)."""
    return Scheme(
        id="scheme-strict",
        name="Strict support",
        eligibility_rules={"max_annual_income": 100000},
    )


def _orchestrator(**overrides) -> IntelligenceOrchestrator:
    """Orchestrator wired to fakes with overrideable parts."""
    parts: dict[str, Any] = {
        "profile_processor": ProfileProcessor(FakeLLM()),
        "need_analyzer": NeedAnalyzer(FakeNeeds()),
    }
    parts.update(overrides)
    return IntelligenceOrchestrator(**parts)


def test_profile_to_eligibility() -> None:
    """Fake profile flows into per-scheme eligibility verdicts."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.profile.age == 25
    assert {s.scheme_id: s.eligibility.status for s in result.schemes} == {
        "scheme-open": "eligible",
        "scheme-strict": "not_eligible",
    }


def test_profile_to_matching() -> None:
    """Only the eligible scheme enters confirmed ranking."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["scheme-open"]
    assert result.ranked_matches[0].rank == 1


def test_eligibility_to_near_miss() -> None:
    """The strict scheme's single income failure reads as a near miss."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-open"].near_miss is None
    assert by_id["scheme-strict"].near_miss.is_near_miss is True


def test_needs_to_support_planning() -> None:
    """Analyzed needs map to verified supports with best coverage."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.needs.total_requested == 500000
    need_plan = result.support_plan.need_plans[0]
    assert need_plan.best_known_coverage == 500000
    assert need_plan.fully_covered is True


def test_support_to_financial_analysis() -> None:
    """Coverage results reuse eligible supports with exact numbers."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert len(result.coverage_results) == 1
    assert result.coverage_results[0].covered == 500000
    assert result.coverage_results[0].coverage_known is True


def test_pathway_to_decision_trace() -> None:
    """Eligible scheme with pathway data gains pathway + traces."""
    result = _orchestrator().run(
        "tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True}
    )
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-open"].pathway.readiness == "needs_information"
    subjects = [t.subject for t in by_id["scheme-open"].traces]
    assert subjects == [
        "eligibility:scheme-open",
        "matching:scheme-open",
        "pathway:scheme-open",
    ]
    assert by_id["scheme-strict"].pathway is None


def test_trace_to_grounded_explanation() -> None:
    """Every trace gets grounded wording; per-scheme primary attached."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(EchoExplanations()))
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True})
    assert len(result.explanations) == len(
        [t for s in result.schemes for t in s.traces] + result.traces
    )
    assert all(e.grounded for e in result.explanations)
    assert result.schemes[0].explanation is not None


def test_semantic_hybrid_ranking() -> None:
    """Hybrid scores blend deterministic + semantic without new weights."""
    orch = _orchestrator(
        semantic_matcher=SemanticMatcher(FakeEmbeddings()), semantic_weight=0.3
    )
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [m.scheme_id for m in result.ranked_matches] == ["scheme-open"]
    assert any("Semantic fit" in reason for reason in result.ranked_matches[0].reasons)


def test_complete_result_and_stages() -> None:
    """Full run populates every section with accurate stage tracking."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(EchoExplanations()))
    result = orch.run("tailoring business", [_open_scheme(), _strict_scheme()], {"Aadhaar": True})
    assert isinstance(result, IntelligenceResult)
    assert result.stages_completed == [
        "profile", "needs", "eligibility", "matching", "near_miss",
        "support", "financial", "pathway", "traces", "explanation",
    ]
    assert result.profile is not None and result.support_plan is not None


def test_missing_llm_provider_keeps_deterministic_output() -> None:
    """No generator means traces stay, explanations stay empty."""
    result = _orchestrator().run("tailoring business", [_open_scheme()])
    assert result.schemes[0].traces != []
    assert result.explanations == []
    assert result.schemes[0].explanation is None
    assert "explanation" not in result.stages_completed
    assert "traces" in result.stages_completed


def test_llm_fallback_preserves_traces() -> None:
    """Ungrounded drafts fall back without losing deterministic traces."""
    orch = _orchestrator(explanation_generator=ExplanationGenerator(BadExplanations()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert all(e.fallback_used for e in result.explanations)
    assert result.schemes[0].traces != []


def test_malformed_scheme_raises_typed_error() -> None:
    """Bad rule formats surface as EligibilityEngineError, never invented."""
    bad = Scheme(id="bad", name="Bad", eligibility_rules={"min_age": "old"})
    with pytest.raises(EligibilityEngineError):
        _orchestrator().run("tailoring business", [bad])


def test_multiple_schemes_associated() -> None:
    """Each scheme keeps its own eligibility/near-miss/pathway slots."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert [s.scheme_id for s in result.schemes] == ["scheme-open", "scheme-strict"]
    assert result.schemes[1].eligibility.status == "not_eligible"


def test_no_eligible_schemes() -> None:
    """Nothing ranked, near-miss still analyzed, plan stays uncertain."""
    result = _orchestrator().run("tailoring business", [_strict_scheme()])
    assert result.ranked_matches == []
    assert result.schemes[0].near_miss.is_near_miss is True
    assert result.schemes[0].pathway is None
    assert result.support_plan.need_plans[0].best_known_coverage is None


def test_needs_information_schemes() -> None:
    """Unknown profile fields keep needs_information out of ranking."""
    from intelligence_engine.profile_processor import ProfileProcessor as PP

    class SparseLLM(FakeLLM):
        def extract_profile_data(self, user_text: str) -> dict[str, Any]:
            """Profile missing occupation."""
            data = super().extract_profile_data(user_text)
            data.pop("occupation")
            return data

    orch = _orchestrator(profile_processor=PP(SparseLLM()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert result.schemes[0].eligibility.status == "needs_information"
    assert result.ranked_matches == []
    near_miss = result.schemes[0].near_miss
    assert near_miss is not None
    assert near_miss.is_near_miss is False
    assert near_miss.failed_criteria == []
    assert "occupation" not in near_miss.satisfied_criteria
    assert near_miss.total_criteria == 6


def test_needs_information_breakdown_counts() -> None:
    """Unresolved schemes expose satisfied/missing structure without verdict change."""
    from intelligence_engine.profile_processor import ProfileProcessor as PP

    class SparseLLM(FakeLLM):
        def extract_profile_data(self, user_text: str) -> dict[str, Any]:
            """Profile missing occupation."""
            data = super().extract_profile_data(user_text)
            data.pop("occupation")
            return data

    orch = _orchestrator(profile_processor=PP(SparseLLM()))
    result = orch.run("tailoring business", [_open_scheme()])
    near_miss = result.schemes[0].near_miss
    assert len(near_miss.satisfied_criteria) == 5
    assert any("occupation" in reason.lower() for reason in near_miss.reasons)


def test_immutability() -> None:
    """Inputs are byte-identical after the run."""
    schemes = [_open_scheme(), _strict_scheme()]
    before = [s.model_dump() for s in schemes]
    _orchestrator().run("tailoring business", schemes)
    assert [s.model_dump() for s in schemes] == before


def test_deterministic_repeated_execution() -> None:
    """Same inputs always give identical IntelligenceResults."""
    orch = _orchestrator()
    args = ("tailoring business", [_open_scheme(), _strict_scheme()])
    assert orch.run(*args) == orch.run(*args)


def test_dependency_injection() -> None:
    """Injected processor output flows through untouched."""

    class FixedProcessor:
        def __init__(self) -> None:
            """Record calls."""
            self.calls = 0

        def process_profile(self, user_text: str) -> UserProfile:
            """Return a fixed profile."""
            self.calls += 1
            return UserProfile(age=40)

    processor = FixedProcessor()
    result = IntelligenceOrchestrator(profile_processor=processor).run("hi", [])
    assert processor.calls == 1
    assert result.profile.age == 40
    assert result.schemes == []


def test_what_if_remains_separate() -> None:
    """Orchestrator never runs or references What-If machinery."""
    orch = _orchestrator()
    assert not hasattr(orch, "what_if")
    import intelligence_engine.orchestrator as orchestrator_module

    assert "WhatIf" not in dir(orchestrator_module)


def test_no_database_or_api_access() -> None:
    """Orchestrator module has no database or API surface."""
    import intelligence_engine.orchestrator as orchestrator_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(orchestrator_module)


def _structured_profile() -> UserProfile:
    """Pre-validated profile matching FakeLLM output."""
    return UserProfile(
        age=25,
        annual_family_income=180000.0,
        occupation="Farmer",
        education_level="Graduate",
        social_category="OBC",
        state="Uttar Pradesh",
        purpose="business expansion",
        project_type="micro-enterprise",
    )


def _structured_needs() -> Any:
    """Pre-analyzed needs matching FakeNeeds output."""
    from intelligence_engine.schemas import NeedAnalysisResult, SupportNeed

    return NeedAnalysisResult(
        business_goal="Expand tailoring business",
        needs=[SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        total_requested=500000,
    )


def test_structured_profile_path_matches_text_path() -> None:
    """run_from_profile reuses the identical downstream flow."""
    orch = _orchestrator()
    schemes = [_open_scheme(), _strict_scheme()]
    from_text = orch.run("tailoring business", schemes)
    profile = _structured_profile()
    from_structured = orch.run_from_profile(profile, schemes, _structured_needs())
    assert from_structured.profile == profile
    text_dump = from_text.model_dump()
    text_dump.pop("stages_completed")
    struct_dump = from_structured.model_dump()
    struct_dump.pop("stages_completed")
    assert struct_dump == text_dump
    assert from_structured.stages_completed == [
        "profile", "eligibility", "matching", "near_miss",
        "support", "financial", "pathway", "traces",
    ]


def test_structured_profile_without_needs_skips_support() -> None:
    """Omitted needs leave support/financial stages unmarked."""
    result = _orchestrator().run_from_profile(_structured_profile(), [_open_scheme()])
    assert result.needs is None
    assert result.support_plan is None
    assert result.coverage_results == []
    assert "support" not in result.stages_completed
    assert "financial" not in result.stages_completed
    assert result.ranked_matches[0].scheme_id == "scheme-open"


def test_structured_profile_never_calls_llm() -> None:
    """The structured path works without any LLM-backed processor output."""

    class ExplodingProcessor:
        def process_profile(self, user_text: str) -> UserProfile:
            """Must never be called."""
            raise AssertionError("LLM path must not run")

    orch = IntelligenceOrchestrator(profile_processor=ExplodingProcessor())
    result = orch.run_from_profile(_structured_profile(), [_open_scheme()], _structured_needs())
    assert result.profile.occupation == "Farmer"
    assert result.schemes[0].eligibility.status == "eligible"


def test_structured_profile_immutability() -> None:
    """Supplied profile and needs objects are unchanged."""
    profile = _structured_profile()
    needs = _structured_needs()
    before = (profile.model_dump(), needs.model_dump())
    _orchestrator().run_from_profile(profile, [_open_scheme()], needs)
    assert profile.model_dump() == before[0]
    assert needs.model_dump() == before[1]


def _sparse_orchestrator(**overrides):
    """Orchestrator whose profile lacks occupation."""
    from intelligence_engine.profile_processor import ProfileProcessor as PP

    class SparseLLM(FakeLLM):
        def extract_profile_data(self, user_text):
            """Profile missing occupation."""
            data = super().extract_profile_data(user_text)
            data.pop("occupation")
            return data

    parts = {"profile_processor": PP(SparseLLM())}
    parts.update(overrides)
    return _orchestrator(**parts)


def test_clarification_populated_on_missing_info() -> None:
    """A: unresolved eligibility yields questions plus the stage marker."""
    result = _sparse_orchestrator().run("tailoring business", [_open_scheme()])
    assert result.clarification is not None
    assert result.clarification.missing_fields == ["occupation"]
    assert result.clarification.questions[0].field == "occupation"
    assert "clarification" in result.stages_completed


def test_clarification_absent_on_complete_run() -> None:
    """B: complete runs carry None and no stage marker."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.clarification is None
    assert "clarification" not in result.stages_completed
    assert result.ranked_matches != []


def test_clarification_deduplicated_across_schemes() -> None:
    """C: two schemes missing occupation ask exactly once."""
    twin = _open_scheme().model_copy(update={"id": "scheme-twin", "name": "Twin"})
    result = _sparse_orchestrator().run("tailoring business", [_open_scheme(), twin])
    assert result.clarification is not None
    assert result.clarification.missing_fields == ["occupation"]
    assert len(result.clarification.questions) == 1


def test_clarification_ignores_failed_schemes() -> None:
    """D: only actionable (needs_information) gaps are asked about."""
    result = _sparse_orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.clarification is not None
    assert result.clarification.missing_fields == ["occupation"]
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-strict"].eligibility.status == "not_eligible"


def test_clarification_partial_result_has_no_nesting() -> None:
    """E: current_partial_result.clarification is always None."""
    result = _sparse_orchestrator().run("tailoring business", [_open_scheme()])
    partial = result.clarification.current_partial_result
    assert partial.clarification is None
    assert "clarification" not in partial.stages_completed


class _EchoWording:
    """Fake wording provider prefixing every question."""

    def __init__(self):
        """Record prompts."""
        self.prompts = []

    def rewrite_questions(self, prompt):
        """Return Hindi-marked rewordings by parsing the prompt lines."""
        self.prompts.append(prompt)
        items = []
        for line in prompt.splitlines():
            if line.startswith("- ["):
                field = line[3:].split("]")[0]
                items.append({"field": field, "question": f"HI ({field})?"})
        return items


def test_clarification_wording_enabled() -> None:
    """F: wording changes text only; fields and order preserved."""
    orch = _sparse_orchestrator(clarification_generator=ClarificationGenerator(_EchoWording()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert result.clarification.questions[0].question == "HI (occupation)?"
    assert result.clarification.missing_fields == ["occupation"]


def test_clarification_wording_failure_falls_back() -> None:
    """G: provider errors keep deterministic questions."""
    orch = _sparse_orchestrator(clarification_generator=ClarificationGenerator(_ExplodingWording()))
    result = orch.run("tailoring business", [_open_scheme()])
    assert result.clarification.questions[0].question == "What is your occupation?"
    assert "clarification" in result.stages_completed


class _ExplodingWording:
    """Wording provider that always fails."""

    def rewrite_questions(self, prompt):
        """Raise instead of wording."""
        raise RuntimeError("boom")


def test_run_from_profile_user_text_only_words() -> None:
    """H: user_text reaches wording context; decisions identical either way."""
    orch = _sparse_orchestrator(clarification_generator=ClarificationGenerator(_EchoWording()))
    profile = _structured_profile().model_copy(update={"occupation": None})
    schemes = [_open_scheme()]
    plain = orch.run_from_profile(profile, schemes)
    voiced = orch.run_from_profile(profile, schemes, user_text="meri umar 25 hai")
    assert plain.schemes[0].eligibility == voiced.schemes[0].eligibility
    assert plain.ranked_matches == voiced.ranked_matches
    assert voiced.clarification.missing_fields == ["occupation"]


def test_no_clarification_preserves_legacy_shape() -> None:
    """I: clean runs look exactly as before (None + unmarked + ranked)."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    assert result.clarification is None
    assert "clarification" not in result.stages_completed
    assert [m.scheme_id for m in result.ranked_matches] == ["scheme-open"]
    assert result.support_plan is not None


def test_clarification_run_immutability() -> None:
    """J: profile, schemes, and needs unchanged by clarification runs."""
    profile = _structured_profile().model_copy(update={"occupation": None})
    schemes = [_open_scheme(), _strict_scheme()]
    needs = _structured_needs()
    before = (profile.model_dump(), [s.model_dump() for s in schemes], needs.model_dump())
    _orchestrator().run_from_profile(profile, schemes, needs)
    assert profile.model_dump() == before[0]
    assert [s.model_dump() for s in schemes] == before[1]
    assert needs.model_dump() == before[2]


class _FakeTranscriber:
    """Canned transcription backend recording its inputs."""

    def __init__(self, text="tailoring business", language=None, error=None):
        """Store canned transcript output."""
        self.text = text
        self.language = language
        self.error = error
        self.calls = []

    def transcribe(self, audio, language_hint=None):
        """Return the canned transcript or raise."""
        from intelligence_engine.transcription import TranscriptionError

        self.calls.append({"audio": audio, "language_hint": language_hint})
        if self.error is not None:
            raise self.error
        if not audio:
            raise TranscriptionError("Empty audio payload.")
        from intelligence_engine.schemas import TranscriptionResult

        return TranscriptionResult(text=self.text, language=self.language)


def _voice_orchestrator(**overrides):
    """Orchestrator with fake transcription wired in."""
    parts = {"transcription_provider": _FakeTranscriber()}
    parts.update(overrides)
    return _orchestrator(**parts)


def test_voice_audio_to_result() -> None:
    """A: audio transcribes and flows through the identical pipeline."""
    result = _voice_orchestrator().run_from_audio(b"fake-audio-bytes", [_open_scheme()])
    assert result.profile.purpose == "business expansion"
    assert result.schemes[0].eligibility.status == "eligible"
    assert result.ranked_matches[0].scheme_id == "scheme-open"


def test_voice_result_matches_text_result() -> None:
    """B: audio output equals the same transcript run as text."""
    orch = _voice_orchestrator()
    schemes = [_open_scheme(), _strict_scheme()]
    from_audio = orch.run_from_audio(b"bytes", schemes)
    from_text = orch.run("tailoring business", schemes)
    assert from_audio == from_text


def test_provider_receives_exact_audio() -> None:
    """C: byte-identical audio reaches the provider."""
    provider = _FakeTranscriber()
    _voice_orchestrator(transcription_provider=provider).run_from_audio(
        b"\x00\x01voice", [_open_scheme()]
    )
    assert provider.calls == [{"audio": b"\x00\x01voice", "language_hint": None}]


def test_language_hint_forwarded_exactly() -> None:
    """D: caller hint passes through untouched."""
    provider = _FakeTranscriber()
    _voice_orchestrator(transcription_provider=provider).run_from_audio(
        b"x", [_open_scheme()], language_hint="hi"
    )
    assert provider.calls[0]["language_hint"] == "hi"


def test_reported_language_reaches_wording() -> None:
    """E: provider-reported language wins for clarification wording."""
    from intelligence_engine.clarification_generator import ClarificationGenerator

    seen = {}

    class _RecordingWording:
        """Capture wording inputs, echo deterministically."""

        def rewrite_questions(self, prompt):
            """Record and echo a fixed rewording."""
            seen["prompt"] = prompt
            return [{"field": "occupation", "question": "HI wording?"}]

    orch = _sparse_orchestrator(
        transcription_provider=_FakeTranscriber(language="Hindi"),
        clarification_generator=ClarificationGenerator(_RecordingWording()),
    )
    profile_schemes = [_open_scheme()]
    result = orch.run_from_audio(b"x", profile_schemes)
    assert result.clarification.questions[0].question == "HI wording?"
    assert " in Hindi" in seen["prompt"]


def test_no_reported_language_keeps_context_behavior() -> None:
    """F: without provider language, transcript context still flows."""
    from intelligence_engine.clarification_generator import ClarificationGenerator

    seen = {}

    class _RecordingWording:
        """Capture wording inputs, echo deterministically."""

        def rewrite_questions(self, prompt):
            """Record and echo a fixed rewording."""
            seen["prompt"] = prompt
            return [{"field": "occupation", "question": "Echo wording?"}]

    orch = _sparse_orchestrator(
        transcription_provider=_FakeTranscriber(language=None),
        clarification_generator=ClarificationGenerator(_RecordingWording()),
    )
    result = orch.run_from_audio(b"x", [_open_scheme()])
    assert result.clarification is not None
    assert "tailoring business" in seen["prompt"]


def test_transcription_error_propagates_unchanged() -> None:
    """G: provider failures abort with the original error, no partial result."""
    from intelligence_engine.transcription import TranscriptionError

    boom = TranscriptionError("mic unavailable")
    orch = _voice_orchestrator(transcription_provider=_FakeTranscriber(error=boom))
    try:
        orch.run_from_audio(b"x", [_open_scheme()])
    except TranscriptionError as exc:
        assert exc is boom
    else:
        raise AssertionError("TranscriptionError must propagate")


def test_missing_provider_raises_value_error() -> None:
    """H: no injected provider is an explicit configuration error."""
    with pytest.raises(ValueError, match="transcription_provider"):
        _orchestrator().run_from_audio(b"x", [_open_scheme()])


def test_audio_never_stored() -> None:
    """I: audio bytes appear nowhere in the result."""
    result = _voice_orchestrator().run_from_audio(b"secret-bytes", [_open_scheme()])
    assert "secret-bytes" not in result.model_dump_json()


def test_voice_inputs_unmodified() -> None:
    """J: schemes unchanged; audio bytes object untouched."""
    schemes = [_open_scheme(), _strict_scheme()]
    audio = b"\x00voice"
    before = [s.model_dump() for s in schemes]
    _voice_orchestrator().run_from_audio(audio, schemes)
    assert [s.model_dump() for s in schemes] == before
    assert audio == b"\x00voice"


def test_voice_runs_deterministic() -> None:
    """K: identical audio yields identical results."""
    orch = _voice_orchestrator()
    schemes = [_open_scheme(), _strict_scheme()]
    assert orch.run_from_audio(b"x", schemes) == orch.run_from_audio(b"x", schemes)


def test_voice_incomplete_info_clarifies() -> None:
    """L: transcript gaps follow the normal clarification flow."""
    provider = _FakeTranscriber(text="I am 25")
    orch = _sparse_orchestrator(transcription_provider=provider)
    result = orch.run_from_audio(b"x", [_open_scheme()])
    assert result.clarification is not None
    assert "occupation" in result.clarification.missing_fields
    assert "clarification" in result.stages_completed


def test_text_answers_need_no_voice_state() -> None:
    """M: follow-up text answers reuse run_from_profile with no voice residue."""
    voice_result = _sparse_orchestrator(transcription_provider=_FakeTranscriber()).run_from_audio(
        b"x", [_open_scheme()]
    )
    assert voice_result.clarification is not None
    profile = _structured_profile()
    text_result = _orchestrator().run_from_profile(profile, [_open_scheme()], _structured_needs())
    assert text_result.clarification is None
    assert "clarification" not in text_result.stages_completed
