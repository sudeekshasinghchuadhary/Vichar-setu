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
    SupportNeed,
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
    """Strict scheme analyzed but off under conservative-off; open scheme untouched."""
    result = _orchestrator().run("tailoring business", [_open_scheme(), _strict_scheme()])
    by_id = {s.scheme_id: s for s in result.schemes}
    assert by_id["scheme-open"].near_miss is None
    assert by_id["scheme-strict"].near_miss.is_near_miss is False


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
    """Nothing ranked, near-miss analyzed but off, plan stays uncertain."""
    result = _orchestrator().run("tailoring business", [_strict_scheme()])
    assert result.ranked_matches == []
    assert result.schemes[0].near_miss.is_near_miss is False
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


def _hybrid_form(**overrides):
    """Sparse form profile with overrides."""
    data = {"state": "Uttar Pradesh"}
    data.update(overrides)
    return UserProfile(**data)


def test_hybrid_form_only_input() -> None:
    """Form alone flows through unchanged with needs skipped."""
    from intelligence_engine.input_merger import merge_inputs

    form = _hybrid_form(age=25)
    profile, needs = merge_inputs(form)
    assert profile.age == 25 and profile.state == "Uttar Pradesh"
    assert needs is None
    result = _orchestrator().run_hybrid(form, [_open_scheme()])
    assert result.profile.state == "Uttar Pradesh"
    assert result.support_plan is None


def test_hybrid_text_only_input() -> None:
    """Empty form plus text behaves like extraction alone."""
    result = _orchestrator().run_hybrid(UserProfile(), [_open_scheme()], user_text="tailoring business")
    assert result.profile.occupation == "Farmer"
    assert result.schemes[0].eligibility.status == "eligible"


def test_hybrid_complementary_fields() -> None:
    """Form district plus text-extracted fields combine without loss."""
    result = _orchestrator().run_hybrid(
        _hybrid_form(district="Lucknow"), [_open_scheme()], user_text="tailoring business"
    )
    assert result.clarification is None
    assert result.profile.district == "Lucknow"
    assert result.profile.occupation == "Farmer"
    assert result.profile.state == "Uttar Pradesh"
    assert result.schemes[0].eligibility.status == "eligible"


def test_merge_profiles_prefills_gaps() -> None:
    """Low-level merge keeps form values; the orchestrator checks conflicts first."""
    from intelligence_engine.input_merger import merge_profiles

    merged = merge_profiles(_hybrid_form(age=30), _structured_profile())
    assert merged.age == 30
    assert merged.occupation == "Farmer"


def test_hybrid_text_fills_missing_form_fields() -> None:
    """Absent form fields take extracted values verbatim."""
    from intelligence_engine.input_merger import merge_profiles

    merged = merge_profiles(UserProfile(), _structured_profile())
    assert merged == _structured_profile()


def test_hybrid_multiple_needs_both_sources() -> None:
    """Form training need plus extracted machinery need both survive."""
    from intelligence_engine.input_merger import merge_needs
    from intelligence_engine.schemas import SupportNeed

    merged = merge_needs(
        [SupportNeed(need_type="training")],
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
    )
    assert [n.need_type for n in merged] == ["training", "machinery"]


def test_hybrid_duplicate_needs_merged() -> None:
    """Same need twice collapses per the existing duplicate contract."""
    from intelligence_engine.input_merger import merge_needs
    from intelligence_engine.schemas import SupportNeed

    merged = merge_needs(
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
        [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")],
    )
    assert len(merged) == 1
    assert merged[0].amount == 1000000


def test_hybrid_missing_unknown_values() -> None:
    """Unknowns stay unknown; no values invented in merging."""
    from intelligence_engine.input_merger import merge_inputs

    profile, needs = merge_inputs(UserProfile(), None, None, None)
    assert profile == UserProfile()
    assert needs is None
    profile2, needs2 = merge_inputs(
        UserProfile(), None, [SupportNeed(need_type="training")], None
    )
    assert needs2.needs[0].amount is None
    assert needs2.total_requested is None


def test_hybrid_immutability() -> None:
    """Source profile, needs, and schemes unchanged by merging and runs."""
    from intelligence_engine.schemas import SupportNeed

    form = _hybrid_form(age=25)
    form_needs = [SupportNeed(need_type="training")]
    schemes = [_open_scheme()]
    before = (form.model_dump(), [n.model_dump() for n in form_needs], [s.model_dump() for s in schemes])
    _orchestrator().run_hybrid(form, schemes, user_text="tailoring business", form_needs=form_needs)
    assert form.model_dump() == before[0]
    assert [n.model_dump() for n in form_needs] == before[1]
    assert [s.model_dump() for s in schemes] == before[2]


def test_hybrid_matches_structured_path() -> None:
    """Hybrid with complete form equals run_from_profile on that profile."""
    form = _structured_profile()
    hybrid = _orchestrator().run_hybrid(form, [_open_scheme(), _strict_scheme()], form_needs=[])
    direct = _orchestrator().run_from_profile(form, [_open_scheme(), _strict_scheme()])
    assert hybrid.profile == direct.profile
    assert [s.eligibility.status for s in hybrid.schemes] == [s.eligibility.status for s in direct.schemes]
    assert [m.scheme_id for m in hybrid.ranked_matches] == [m.scheme_id for m in direct.ranked_matches]


def test_hybrid_rejects_audio_and_text_together() -> None:
    """Ambiguous dual input raises instead of guessing precedence."""
    with pytest.raises(ValueError, match="either audio or user_text"):
        _orchestrator().run_hybrid(_hybrid_form(), [_open_scheme()], user_text="hi", audio=b"x")


def test_hybrid_audio_uses_text_pipeline() -> None:
    """Voice plus form merges the transcript extraction with form data."""
    orch = _voice_orchestrator()
    result = orch.run_hybrid(_hybrid_form(district="Lucknow"), [_open_scheme()], audio=b"bytes")
    assert result.profile.district == "Lucknow"
    assert result.profile.occupation == "Farmer"
    assert result.schemes[0].eligibility.status == "eligible"


def test_legacy_flows_unchanged() -> None:
    """run/run_from_profile/run_from_audio behave exactly as before."""
    orch = _orchestrator()
    schemes = [_open_scheme(), _strict_scheme()]
    text_result = orch.run("tailoring business", schemes)
    assert text_result.stages_completed == [
        "profile", "needs", "eligibility", "matching", "near_miss",
        "support", "financial", "pathway", "traces",
    ]
    assert _orchestrator().run_from_profile(_structured_profile(), schemes).profile.age == 25
    assert _voice_orchestrator().run_from_audio(b"x", schemes).profile.age == 25


def _conflict_form(**overrides):
    """Form profile differing from FakeLLM extraction (age 25)."""
    data = {"age": 32}
    data.update(overrides)
    return UserProfile(**data)


def test_identical_values_no_conflict() -> None:
    """Same values in both sources merge normally with no clarification."""
    from intelligence_engine.input_merger import detect_conflicts

    profile = UserProfile(age=25, occupation="Farmer")
    assert detect_conflicts(profile, UserProfile(age=25, occupation="Farmer")) == []
    result = _orchestrator().run_hybrid(UserProfile(), [_open_scheme()], user_text="tailoring business")
    assert result.clarification is None
    assert "clarification" not in result.stages_completed


def test_complementary_values_normal_merge() -> None:
    """Form state plus extracted everything else merges with no dispute."""
    result = _orchestrator().run_hybrid(
        UserProfile(state="Uttar Pradesh"), [_open_scheme()], user_text="tailoring business"
    )
    assert result.clarification is None
    assert result.profile.state == "Uttar Pradesh"
    assert result.profile.occupation == "Farmer"
    assert result.schemes[0].eligibility.status == "eligible"


def test_conflicting_numeric_values_clarify() -> None:
    """Form 32/4L vs extracted 25/1.8L asks instead of deciding."""
    form = _conflict_form(annual_family_income=400000.0)
    result = _orchestrator().run_hybrid(form, [_open_scheme(), _strict_scheme()], user_text="tailoring business")
    assert result.clarification is not None
    assert [q.field for q in result.clarification.questions] == ["age", "annual_family_income"]
    assert "400000" in result.clarification.questions[1].question
    assert "180000" in result.clarification.questions[1].question
    assert result.ranked_matches == []
    assert result.schemes == []
    assert "clarification" in result.stages_completed
    assert "matching" not in result.stages_completed


def test_conflicting_categorical_values_clarify() -> None:
    """Occupation Tailor vs Farmer is disputed, not overwritten."""
    form = UserProfile(occupation="Tailor")
    result = _orchestrator().run_hybrid(form, [_open_scheme()], user_text="tailoring business")
    assert result.clarification is not None
    assert result.clarification.questions[0].field == "occupation"
    assert "Tailor" in result.clarification.questions[0].question
    assert "Farmer" in result.clarification.questions[0].question


def test_multiple_conflicts_all_surfaced_once() -> None:
    """Every disputed field appears exactly once in field order."""
    from intelligence_engine.input_merger import detect_conflicts

    conflicts = detect_conflicts(
        UserProfile(age=32, occupation="Tailor", state="Bihar"),
        UserProfile(age=35, occupation="Farmer", state="UP"),
    )
    assert [c.field for c in conflicts] == ["age", "state", "occupation"]
    assert conflicts[0].form_value == 32 and conflicts[0].extracted_value == 35


def test_conflict_preserves_both_sources() -> None:
    """Neither source is overwritten; disputed profile fields stay unknown."""
    form = _conflict_form(annual_family_income=400000.0)
    before = form.model_dump()
    result = _orchestrator().run_hybrid(form, [_open_scheme()], user_text="tailoring business")
    assert form.model_dump() == before
    assert result.profile.age is None
    assert result.profile.annual_family_income is None
    assert result.profile.state == "Uttar Pradesh"


def test_confirmation_produces_canonical_value() -> None:
    """User confirmation selects the canonical value for reruns."""
    from intelligence_engine.input_merger import resolve_conflicts

    form = _conflict_form()
    extracted = UserProfile(age=25)
    assert resolve_conflicts(form, extracted, {"age": "extracted"}).age == 25
    assert resolve_conflicts(form, extracted, {"age": "form"}).age == 32
    import pytest as _pytest

    with _pytest.raises(ValueError):
        resolve_conflicts(form, extracted, {"age": "maybe"})
    with _pytest.raises(ValueError):
        resolve_conflicts(form, extracted, {"business_age": "form"})
    with _pytest.raises(ValueError):
        resolve_conflicts(form, None, {"age": "extracted"})


def test_conflict_request_structure() -> None:
    """Conflict requests carry conflicts, empty missing, and clean partials."""
    form = _conflict_form()
    result = _orchestrator().run_hybrid(form, [_open_scheme()], user_text="tailoring business")
    request = result.clarification
    assert request.missing_fields == []
    assert len(request.conflicts) == 1
    assert request.conflicts[0].field == "age"
    assert request.current_partial_result.clarification is None
    assert "clarification" not in request.current_partial_result.stages_completed


def test_default_values_never_conflict() -> None:
    """Schema defaults (e.g. income_period annual) are not user assertions."""
    from intelligence_engine.input_merger import detect_conflicts

    assert detect_conflicts(UserProfile(), UserProfile(age=25)) == []
    assert detect_conflicts(
        UserProfile(), UserProfile(age=25, income_period="unknown")
    ) == []
    assert detect_conflicts(UserProfile(), None) == []


def test_malformed_choices_rejected() -> None:
    """Resolution validates fields and choice values strictly."""
    import pytest as _pytest

    from intelligence_engine.input_merger import resolve_conflicts

    with _pytest.raises(ValueError):
        resolve_conflicts(UserProfile(age=32), UserProfile(age=35), {"age": 42})
    resolved = resolve_conflicts(UserProfile(age=32), UserProfile(age=35), {})
    assert resolved.age == 32


def test_conflict_wording_uses_generator() -> None:
    """Wording layer may rephrase conflict questions without changing fields."""
    from intelligence_engine.clarification_generator import ClarificationGenerator

    class _Echo:
        """Echo fields with fixed rewording."""

        def rewrite_questions(self, prompt):
            """Return deterministic rewordings parsed from the prompt."""
            items = []
            for line in prompt.splitlines():
                if line.startswith("- ["):
                    field = line[3:].split("]")[0]
                    items.append({"field": field, "question": f"Confirm {field}?"})
            return items

    orch = _orchestrator(clarification_generator=ClarificationGenerator(_Echo()))
    result = orch.run_hybrid(_conflict_form(), [_open_scheme()], user_text="tailoring business")
    assert result.clarification.questions[0].question == "Confirm age?"


def test_missing_info_clarification_unaffected() -> None:
    """needs_information path still works with no conflicts involved."""
    result = _sparse_orchestrator().run("tailoring business", [_open_scheme()])
    assert result.clarification is not None
    assert result.clarification.conflicts == []
    assert result.clarification.missing_fields == ["occupation"]


class _PayloadLLM(LLMClient):
    """LLM fake returning a configured extraction payload."""

    def __init__(self, payload):
        """Store the payload."""
        self.payload = payload

    def extract_profile_data(self, user_text):
        """Return the configured payload."""
        return dict(self.payload)


class _PayloadNeeds(NeedExtractor):
    """Need fake returning a configured extraction payload."""

    def __init__(self, payload):
        """Store the payload."""
        self.payload = payload

    def extract_need_data(self, user_text):
        """Return the configured payload."""
        return dict(self.payload)


def _payload_orchestrator(profile_payload, needs_payload=None, **overrides):
    """Orchestrator with fully controlled extraction outputs."""
    parts = {
        "profile_processor": ProfileProcessor(_PayloadLLM(profile_payload)),
        "need_analyzer": NeedAnalyzer(
            _PayloadNeeds(needs_payload if needs_payload is not None else {"needs": []})
        ),
    }
    parts.update(overrides)
    return _orchestrator(**parts)


def test_equivalent_casing_no_conflict() -> None:
    """Lucknow/lucknow and Female/female are equivalent, not conflicts."""
    from intelligence_engine.input_merger import detect_conflicts

    assert detect_conflicts(UserProfile(state="Lucknow"), UserProfile(state="lucknow")) == []
    assert detect_conflicts(UserProfile(gender="Female"), UserProfile(gender="female")) == []
    assert detect_conflicts(
        UserProfile(occupation="  Tailor  "), UserProfile(occupation="tailor")
    ) == []


def test_empty_strings_are_not_assertions() -> None:
    """Blank strings behave like missing values in detection."""
    from intelligence_engine.input_merger import detect_conflicts

    assert detect_conflicts(UserProfile(occupation=""), UserProfile(occupation="Farmer")) == []
    assert detect_conflicts(UserProfile(occupation="   "), UserProfile(occupation="Farmer")) == []
    assert detect_conflicts(UserProfile(), UserProfile()) == []


def test_currency_strings_canonicalize() -> None:
    """Indian-format amounts parse to identical canonical numbers."""
    from intelligence_engine.profile_processor import normalize_profile_data

    assert normalize_profile_data({"annual_family_income": "₹4,00,000"}) == {
        "annual_family_income": 400000.0
    }
    assert normalize_profile_data({"annual_family_income": "4 lakh"}) == {
        "annual_family_income": 400000.0
    }
    assert normalize_profile_data({"annual_family_income": "1.5 crore"}) == {
        "annual_family_income": 15000000.0
    }
    assert normalize_profile_data({"annual_family_income": "Rs. 2,50,000.50"}) == {
        "annual_family_income": 250000.5
    }
    assert normalize_profile_data({"annual_family_income": "lots of money"}) == {
        "annual_family_income": "lots of money"
    }


def test_monthly_annual_equivalence_no_conflict() -> None:
    """25000/month vs 300000/year is equivalent when periods are known."""
    from intelligence_engine.input_merger import detect_conflicts

    form = UserProfile(annual_family_income=25000, income_period="monthly")
    extracted = UserProfile(annual_family_income=300000, income_period="annual")
    assert detect_conflicts(form, extracted) == []


def test_monthly_mismatch_stays_conflict() -> None:
    """25000/month vs 50000/month is a genuine conflict."""
    from intelligence_engine.input_merger import detect_conflicts

    conflicts = detect_conflicts(
        UserProfile(annual_family_income=25000, income_period="monthly"),
        UserProfile(annual_family_income=50000, income_period="monthly"),
    )
    assert [c.field for c in conflicts] == ["annual_family_income"]


def test_unknown_period_blocks_equivalence() -> None:
    """Known monthly vs unknown period cannot be proven equivalent."""
    from intelligence_engine.input_merger import detect_conflicts

    conflicts = detect_conflicts(
        UserProfile(annual_family_income=25000, income_period="monthly"),
        UserProfile(annual_family_income=300000, income_period="unknown"),
    )
    assert [c.field for c in conflicts] == ["annual_family_income", "income_period"]


def test_zero_negative_huge_decimal_amounts() -> None:
    """Zero/decimal/huge values compare exactly; negatives fail validation."""
    from intelligence_engine.input_merger import detect_conflicts
    from pydantic import ValidationError

    assert detect_conflicts(UserProfile(annual_family_income=0), UserProfile(annual_family_income=0.0)) == []
    assert detect_conflicts(
        UserProfile(annual_family_income=1000000000000), UserProfile(annual_family_income=1000000000000.0)
    ) == []
    assert detect_conflicts(
        UserProfile(annual_family_income=250000.5), UserProfile(annual_family_income=250000.6)
    )[0].field == "annual_family_income"
    with pytest.raises(ValidationError):
        UserProfile(annual_family_income=-5)


def test_four_simultaneous_conflicts() -> None:
    """All disputed fields surface once each in field order; engine idle."""
    orch = _payload_orchestrator(
        {
            "age": 40,
            "occupation": "Teacher",
            "state": "Bihar",
            "annual_family_income": 900000,
        }
    )
    form = UserProfile(age=32, occupation="Tailor", state="Uttar Pradesh", annual_family_income=400000)
    result = orch.run_hybrid(form, [_open_scheme(), _strict_scheme()], user_text="anything")
    assert [q.field for q in result.clarification.questions] == [
        "age",
        "state",
        "occupation",
        "annual_family_income",
    ]
    assert result.ranked_matches == []
    assert result.schemes == []
    assert "eligibility" not in result.stages_completed
    assert "matching" not in result.stages_completed


def test_conflict_plus_missing_needs() -> None:
    """Disputed profile blocks the engine; unknown needs ride along unsolved."""
    orch = _payload_orchestrator({"age": 35}, {"needs": [{"need_type": "training"}]})
    result = orch.run_hybrid(
        UserProfile(age=32), [_open_scheme()], user_text="hi", form_needs=[SupportNeed(need_type="training")]
    )
    assert [q.field for q in result.clarification.questions] == ["age"]
    assert result.clarification.missing_fields == []
    assert result.schemes == []
    assert result.support_plan is None


def test_need_amount_conflict_clarifies() -> None:
    """Form 5L vs text 8L for machinery asks instead of summing."""
    from intelligence_engine.input_merger import detect_need_conflicts

    form_needs = [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")]
    text_needs = [SupportNeed(need_type="machinery", amount=800000, amount_period="one_time")]
    conflicts = detect_need_conflicts(form_needs, text_needs)
    assert len(conflicts) == 1
    assert conflicts[0].field == "needs:machinery:one_time"
    assert conflicts[0].form_value == 500000
    assert conflicts[0].extracted_value == 800000

    orch = _payload_orchestrator(
        {"age": 25},
        {"needs": [{"need_type": "machinery", "amount": 800000, "amount_period": "one_time"}]},
    )
    result = orch.run_hybrid(
        UserProfile(age=25), [_open_scheme()], user_text="hi", form_needs=form_needs
    )
    assert [q.field for q in result.clarification.questions] == ["needs:machinery:one_time"]
    assert "500000" in result.clarification.questions[0].question
    assert "800000" in result.clarification.questions[0].question
    assert result.ranked_matches == []
    assert result.support_plan is None


def test_need_distinct_types_merge() -> None:
    """Machinery 5L plus working-capital 2L merge without dispute."""
    from intelligence_engine.input_merger import detect_need_conflicts, merge_needs

    form_needs = [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")]
    text_needs = [SupportNeed(need_type="working_capital", amount=200000, amount_period="one_time")]
    assert detect_need_conflicts(form_needs, text_needs) == []
    merged = merge_needs(form_needs, text_needs)
    assert [n.need_type for n in merged] == ["machinery", "working_capital"]


def test_need_resolution_selects_canonical_amount() -> None:
    """Confirmed need choices produce the canonical needs list."""
    from intelligence_engine.input_merger import resolve_need_conflicts

    form_needs = [SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")]
    text_needs = [SupportNeed(need_type="machinery", amount=800000, amount_period="one_time")]
    resolved = resolve_need_conflicts(
        form_needs, text_needs, {"needs:machinery:one_time": "extracted"}
    )
    assert len(resolved) == 1
    assert resolved[0].amount == 800000
    resolved_form = resolve_need_conflicts(
        form_needs, text_needs, {"needs:machinery:one_time": "form"}
    )
    assert resolved_form[0].amount == 500000
    with pytest.raises(ValueError):
        resolve_need_conflicts(form_needs, text_needs, {"needs:machinery:one_time": "maybe"})
    with pytest.raises(ValueError):
        resolve_need_conflicts(form_needs, text_needs, {"needs:training:one_time": "form"})


def test_voice_agreeing_values_run_normally() -> None:
    """Voice transcript matching the form executes the full engine."""
    orch = _voice_orchestrator()
    form = UserProfile(district="Lucknow")
    result = orch.run_hybrid(form, [_open_scheme()], audio=b"bytes")
    assert result.clarification is None
    assert result.profile.district == "Lucknow"
    assert result.schemes[0].eligibility.status == "eligible"


def test_voice_conflicting_values_clarify() -> None:
    """Voice contradicting the form asks instead of running engines."""
    orch = _voice_orchestrator()
    result = orch.run_hybrid(UserProfile(age=32), [_open_scheme()], audio=b"bytes")
    assert [q.field for q in result.clarification.questions] == ["age"]
    assert result.ranked_matches == []


def test_voice_transcription_failure_propagates() -> None:
    """Transcription errors abort hybrid runs with the original error."""
    from intelligence_engine.transcription import TranscriptionError

    boom = TranscriptionError("mic unavailable")
    orch = _voice_orchestrator(transcription_provider=_FakeTranscriber(error=boom))
    with pytest.raises(TranscriptionError) as exc_info:
        orch.run_hybrid(UserProfile(), [_open_scheme()], audio=b"x")
    assert exc_info.value is boom


def test_invalid_extracted_profile_rejected() -> None:
    """Wrong-typed extraction fails validation before any engine runs."""
    from intelligence_engine.profile_processor import ProfileProcessingError

    orch = _payload_orchestrator({"age": "old"})
    with pytest.raises(ProfileProcessingError):
        orch.run_hybrid(UserProfile(), [_open_scheme()], user_text="hi")


def test_invalid_extracted_values_rejected() -> None:
    """Negative ages, non-finite amounts, and malformed needs all fail."""
    from intelligence_engine.need_analyzer import NeedAnalysisError
    from intelligence_engine.profile_processor import ProfileProcessingError

    with pytest.raises(ProfileProcessingError):
        _payload_orchestrator({"age": -5}).run_hybrid(UserProfile(), [_open_scheme()], user_text="hi")
    with pytest.raises(ProfileProcessingError):
        _payload_orchestrator({"annual_family_income": float("inf")}).run_hybrid(
            UserProfile(), [_open_scheme()], user_text="hi"
        )
    with pytest.raises(NeedAnalysisError):
        _payload_orchestrator({"age": 25}, {"needs": "machinery"}).run_hybrid(
            UserProfile(), [_open_scheme()], user_text="hi"
        )
    with pytest.raises(NeedAnalysisError):
        _payload_orchestrator({"age": 25}, {"needs": [{"amount": 5}]}).run_hybrid(
            UserProfile(), [_open_scheme()], user_text="hi"
        )


def test_conflict_integrity_end_to_end() -> None:
    """One entry/question per conflict; values exact; nothing unrelated."""
    orch = _payload_orchestrator({"age": 35, "occupation": "Teacher"})
    result = orch.run_hybrid(
        UserProfile(age=32, occupation="Tailor"), [_open_scheme()], user_text="hi"
    )
    assert [q.field for q in result.clarification.questions] == ["age", "occupation"]
    assert len(result.clarification.conflicts) == 2
    by_field = {c.field: c for c in result.clarification.conflicts}
    assert (by_field["age"].form_value, by_field["age"].extracted_value) == (32, 35)
    assert "Which one is correct?" in result.clarification.questions[0].question


def test_conflict_wording_fallback_intact() -> None:
    """Bad rewrites keep deterministic conflict questions."""
    from intelligence_engine.clarification_generator import ClarificationGenerator

    class _Bad:
        """Always-invalid rewording."""

        def rewrite_questions(self, prompt):
            """Return mismatched fields."""
            return [{"field": "nope", "question": "What?"}]

    orch = _payload_orchestrator({"age": 35})
    orch.clarification_generator = ClarificationGenerator(_Bad())
    result = orch.run_hybrid(UserProfile(age=32), [_open_scheme()], user_text="hi")
    assert "32" in result.clarification.questions[0].question
    assert "35" in result.clarification.questions[0].question


def _isolation_profile() -> UserProfile:
    """Profile engineered for exact deterministic/semantic score control."""
    return UserProfile(
        age=25,
        purpose=None,
        project_type=None,
        occupation="farm loan officer",
        state="S",
        social_category="C",
    )


def _isolation_schemes() -> list[Scheme]:
    """Two eligible schemes with inverted det/sem profiles, plus excluded ones."""
    return [
        Scheme(
            id="scheme-det",
            name="Deterministic fit",
            description="qqq",
            supported_purposes=["zzz"],
            supported_project_types=["zzz"],
            eligibility_rules={
                "min_age": 18,
                "occupations": ["farm loan officer"],
                "states": ["S"],
                "categories": ["C"],
            },
        ),
        Scheme(
            id="scheme-sem",
            name="Semantic fit",
            description="farm loan micro enterprise",
            supported_purposes=["zzz"],
            supported_project_types=["zzz"],
            eligibility_rules={"min_age": 18},
        ),
        Scheme(
            id="scheme-out",
            name="Ineligible but relevant",
            description="farm loan micro enterprise",
            supported_purposes=["zzz"],
            eligibility_rules={"occupations": ["zzz"]},
        ),
        Scheme(
            id="scheme-unk",
            name="Unknown income",
            description="qqq",
            supported_purposes=["zzz"],
            eligibility_rules={"max_annual_income": 300000},
        ),
    ]


def _isolation_results() -> dict[float, Any]:
    """Run the same fixture at three materially different weights."""
    profile = _isolation_profile()
    schemes = _isolation_schemes()
    results = {}
    for weight in (0.0, 0.3, 0.9):
        orch = _orchestrator(
            semantic_matcher=SemanticMatcher(FakeEmbeddings()),
            semantic_weight=weight,
        )
        results[weight] = orch.run_from_profile(profile, schemes)
    return results


def test_semantic_weight_isolation_regression() -> None:
    """Weight may reorder eligible candidates; it must not move anything else.

    EXPECTED: ranking order may change across weights.
    FORBIDDEN: any change to eligibility states, traces, eligible
    admission, or Near-Miss membership.
    """
    results = _isolation_results()
    baseline = results[0.0]

    for weight in (0.3, 0.9):
        result = results[weight]
        assert [
            (insight.scheme_id, insight.eligibility.status) for insight in result.schemes
        ] == [
            (insight.scheme_id, insight.eligibility.status) for insight in baseline.schemes
        ]
        assert [
            insight.eligibility.reasons for insight in result.schemes
        ] == [
            insight.eligibility.reasons for insight in baseline.schemes
        ]
        assert [
            insight.scheme_id for insight in result.schemes
            if insight.eligibility.status == "eligible"
        ] == [
            insight.scheme_id for insight in baseline.schemes
            if insight.eligibility.status == "eligible"
        ]
        assert {
            insight.scheme_id: (insight.near_miss.is_near_miss if insight.near_miss else None)
            for insight in result.schemes
        } == {
            insight.scheme_id: (insight.near_miss.is_near_miss if insight.near_miss else None)
            for insight in baseline.schemes
        }
        assert all(
            insight.scheme_id not in {match.scheme_id for match in result.ranked_matches}
            for insight in result.schemes
            if insight.eligibility.status != "eligible"
        )

    assert [match.scheme_id for match in results[0.0].ranked_matches] == ["scheme-det", "scheme-sem"]
    assert [match.scheme_id for match in results[0.3].ranked_matches] == ["scheme-det", "scheme-sem"]
    assert [match.scheme_id for match in results[0.9].ranked_matches] == ["scheme-sem", "scheme-det"]
    assert results[0.9].ranked_matches[0].score == 63.64
    assert results[0.9].ranked_matches[1].score == 10.0
    assert results[0.0].ranked_matches[0].score == 100.0
    assert results[0.0].ranked_matches[1].score == 0.0


def test_eligible_zero_semantic_still_ranked_CURRENT_BEHAVIOR_PENDING_POLICY() -> None:
    """CURRENT BEHAVIOR (pending product policy, not final specification).

    Today an eligible scheme with zero semantic relevance is still
    ranked: there is no relevance floor. Whether eligible-but-irrelevant
    schemes should be surfaced, flagged, or suppressed is an open
    product-policy question; update this test when it is decided.
    """
    results = _isolation_results()
    det_match = next(
        match for match in results[0.9].ranked_matches if match.scheme_id == "scheme-det"
    )
    assert det_match.semantic_score == 0.0
    assert det_match.rank == 2
