"""Offline tests for BHASHINI NMT language preprocessing (mocked HTTP).

Scope — what these tests PROVE (adapter/plumbing correctness):
- request/response envelope shape, error mapping, original-text
  preservation, opt-in routing, and fail-closed behavior.

What they DO NOT prove (stated explicitly, never claimed):
- real BHASHINI NMT semantic quality, multilingual accuracy, or
  preservation of any financial nuance by the live service. The
  "uncertainty / numeric / negation" cases below use CONTROLLED
  canned translations to verify the plumbing carries the distinction
  through — they say nothing about what live BHASHINI returns.
No live round-trip has been performed; see bhashini_provider.py.
"""

from typing import Any

import pytest

from intelligence_engine.bhashini_provider import (
    BhashiniError,
    BhashiniNmtConfig,
    BhashiniTranslationProvider,
    bhashini_nmt_enabled,
    select_translation_provider,
)
from intelligence_engine.llm_client import LLMClient, NeedExtractor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.schemas import TranslationResult
from intelligence_engine.transcription import TranscriptionProvider
from intelligence_engine.translation import (
    TranslationError,
    TranslationProvider,
    validate_translation,
)
from intelligence_engine.schemas import TranscriptionResult


def _config(**overrides: Any) -> BhashiniNmtConfig:
    """Test NMT config without touching the environment."""
    params: dict[str, Any] = {"user_id": "test-user", "api_key": "test-key"}
    params.update(overrides)
    return BhashiniNmtConfig(**params)


def _pipeline(service_id: str = "nmt-hi-en-1") -> dict[str, Any]:
    """Canned ULCA pipeline-discovery response for translation."""
    return {
        "pipelineResponseConfig": [
            {
                "taskType": "translation",
                "config": [
                    {
                        "serviceId": service_id,
                        "language": {"sourceLanguage": "hi", "targetLanguage": "en"},
                    }
                ],
            }
        ],
        "pipelineInferenceAPIEndPoint": {
            "callbackUrl": "https://infer.example/pipeline",
            "inferenceApiKey": {"name": "Authorization", "value": "tok"},
        },
    }


def _infer(target: Any) -> dict[str, Any]:
    """Canned ULCA inference response carrying the translation."""
    return {"pipelineResponse": [{"taskType": "translation", "output": [{"target": target}]}]}


class FakeHttp:
    """Canned HTTP backend recording every call."""

    def __init__(self, pipeline: Any = None, inference: Any = None, error: Any = None) -> None:
        """Store canned responses or an error to raise."""
        self.pipeline = pipeline if pipeline is not None else _pipeline()
        self.inference = inference if inference is not None else _infer("hello world")
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, payload: dict[str, Any], headers: dict[str, str]):
        """Record the call, then answer from the canned fixtures."""
        self.calls.append({"url": url, "payload": payload, "headers": headers})
        if self.error is not None:
            raise self.error
        if "getModelsPipeline" in url:
            return self.pipeline
        return self.inference


def _provider(http: FakeHttp, **overrides: Any) -> BhashiniTranslationProvider:
    """NMT provider wired to the fake HTTP backend."""
    return BhashiniTranslationProvider(config=_config(**overrides), http_post=http)


# --- Provider contract ---

def test_implements_translation_abstraction() -> None:
    """BHASHINI NMT provider satisfies the TranslationProvider contract."""
    assert issubclass(BhashiniTranslationProvider, TranslationProvider)
    result = _provider(FakeHttp()).translate("नमस्ते", "hi")
    assert isinstance(result, TranslationResult)
    assert result.original_text == "नमस्ते"
    assert result.translated_text == "hello world"
    assert result.source_language == "hi"
    assert result.target_language == "en"


def test_successful_translation_envelope() -> None:
    """Discovery asks for translation hi->en; inference sends text + serviceId."""
    http = FakeHttp()
    _provider(http).translate("मुझे मदद चाहिए", "hi")
    discovery = http.calls[0]["payload"]
    task = discovery["pipelineTasks"][0]
    assert task["taskType"] == "translation"
    assert task["config"]["language"] == {"sourceLanguage": "hi", "targetLanguage": "en"}
    inference = http.calls[1]["payload"]
    itask = inference["pipelineTasks"][0]
    assert itask["taskType"] == "translation"
    assert itask["config"]["serviceId"] == "nmt-hi-en-1"
    assert itask["config"]["language"] == {"sourceLanguage": "hi", "targetLanguage": "en"}
    assert inference["inputData"] == {"input": [{"source": "मुझे मदद चाहिए"}]}


def test_malformed_discovery_rejected() -> None:
    """Bad pipeline shapes raise TranslationError, never fabricated English."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(pipeline={})).translate("x", "hi")
    with pytest.raises(TranslationError):
        _provider(FakeHttp(pipeline={"pipelineResponseConfig": []})).translate("x", "hi")


def test_malformed_inference_rejected() -> None:
    """Responses without a target string raise instead of guessing."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(inference={})).translate("x", "hi")
    with pytest.raises(TranslationError):
        _provider(FakeHttp(inference=_infer(None))).translate("x", "hi")
    with pytest.raises(TranslationError):
        _provider(FakeHttp(inference=_infer(42))).translate("x", "hi")


def test_empty_translation_rejected() -> None:
    """Blank translations fail closed — never a silent fallback to input."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(inference=_infer("   "))).translate("मदद", "hi")


def test_auth_failure_converted() -> None:
    """Credential failures become TranslationError, never raw tracebacks."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(error=RuntimeError("401 Unauthorized"))).translate("x", "hi")


def test_network_failure_converted() -> None:
    """Socket-level failures become TranslationError."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(error=ConnectionError("boom"))).translate("x", "hi")


def test_timeout_converted() -> None:
    """Timeouts surface as TranslationError through the provider contract."""
    with pytest.raises(TranslationError):
        _provider(FakeHttp(error=TimeoutError("timed out"))).translate("x", "hi")


def test_source_language_configuration() -> None:
    """Explicit per-call source wins; config default applies otherwise."""
    http = FakeHttp()
    _provider(http).translate("x", "pa")
    assert http.calls[0]["payload"]["pipelineTasks"][0]["config"]["language"]["sourceLanguage"] == "pa"
    http2 = FakeHttp()
    _provider(http2, source_language="ta").translate("x", "")
    assert http2.calls[0]["payload"]["pipelineTasks"][0]["config"]["language"]["sourceLanguage"] == "ta"


def test_target_language_defaults_to_english() -> None:
    """Default target is English; an explicit target is honored."""
    assert _provider(FakeHttp()).translate("x", "hi").target_language == "en"
    http = FakeHttp()
    result = _provider(http).translate("x", "hi", target_language="mr")
    assert result.target_language == "mr"
    assert http.calls[0]["payload"]["pipelineTasks"][0]["config"]["language"]["targetLanguage"] == "mr"


def test_whitespace_handling() -> None:
    """Both sides collapse whitespace losslessly like all providers."""
    result = _provider(FakeHttp(inference=_infer("  hello   world  "))).translate("  मदद   चाहिए  ", "hi")
    assert result.original_text == "मदद चाहिए"
    assert result.translated_text == "hello world"


def test_original_text_preserved() -> None:
    """The source wording is retained verbatim (modulo whitespace)."""
    result = _provider(FakeHttp(inference=_infer("I need help"))).translate("मुझे मदद चाहिए", "hi")
    assert result.original_text == "मुझे मदद चाहिए"
    assert result.translated_text != result.original_text


def test_empty_source_text_rejected() -> None:
    """Empty/blank/non-string inputs fail before any HTTP call."""
    http = FakeHttp()
    with pytest.raises(TranslationError):
        _provider(http).translate("   ", "hi")
    with pytest.raises(TranslationError):
        _provider(http).translate(42, "hi")  # type: ignore[arg-type]
    assert http.calls == []


def test_uncertainty_numeric_negation_plumbing() -> None:
    """CANNED translations preserve hedging/numbers/negation through the adapter.

    These prove the adapter carries the distinction end to end; they do
    NOT prove live BHASHINI preserves these meanings.
    """
    cases = [
        ("लगभग 5 लाख", "approximately 500000"),
        ("मुझे लोन नहीं चाहिए", "I do not want a loan"),
        ("शायद मुझे 5 लाख की जरूरत होगी", "maybe I will need 500000"),
        ("मुझे 5 लाख से ज्यादा नहीं चाहिए", "I do not want more than 500000"),
    ]
    for source, canned in cases:
        result = _provider(FakeHttp(inference=_infer(canned))).translate(source, "hi")
        assert result.original_text == source  # original untouched
        assert result.translated_text == canned  # controlled distinction carried


def test_validate_translation_helper() -> None:
    """Caller-held pairs validate without network; bad shapes raise typed errors."""
    assert validate_translation("a", "b", "hi").translated_text == "b"
    with pytest.raises(TranslationError):
        validate_translation("   ", "b", "hi")
    with pytest.raises(TranslationError):
        validate_translation("a", "   ", "hi")
    with pytest.raises(TranslationError):
        validate_translation(42, "b", "hi")


def test_disabled_keeps_existing_provider() -> None:
    """Without the flag, selection returns the current provider untouched."""
    current = object()
    assert select_translation_provider(current, environ={}) is current
    assert select_translation_provider(None, environ={"BHASHINI_NMT_ENABLED": "false"}) is None
    assert bhashini_nmt_enabled({}) is False


def test_enabled_selects_nmt_provider() -> None:
    """With the flag plus shared credentials, selection builds the NMT provider."""
    provider = select_translation_provider(
        None,
        environ={"BHASHINI_NMT_ENABLED": "true", "BHASHINI_USER_ID": "u", "BHASHINI_API_KEY": "k"},
    )
    assert isinstance(provider, BhashiniTranslationProvider)


def test_missing_credentials_rejected_at_construction() -> None:
    """Enabled selection without credentials fails closed with a typed error."""
    with pytest.raises(BhashiniError):
        select_translation_provider(
            None, environ={"BHASHINI_NMT_ENABLED": "1", "BHASHINI_USER_ID": "u"}
        )


# --- Orchestrator routing ---

class FakeTranslator(TranslationProvider):
    """Canned NMT backend: mapping or error, recording every call."""

    def __init__(self, mapping: dict[str, str] | None = None, error: Any = None) -> None:
        """Store canned translations."""
        self.mapping = mapping or {}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def translate(self, text: str, source_language: str, target_language: str = "en"):
        """Return the canned translation or raise the canned error."""
        self.calls.append(
            {"text": text, "source_language": source_language, "target_language": target_language}
        )
        if self.error is not None:
            raise self.error
        return validate_translation(text, self.mapping.get(text, "translated " + text), source_language)


class _FakeLLM(LLMClient):
    """Canned profile extraction recording the text it received."""

    def __init__(self) -> None:
        """Track received inputs."""
        self.seen: list[str] = []

    def extract_profile_data(self, user_text: str):
        """Return a fixed profile."""
        self.seen.append(user_text)
        return {"age": 30, "occupation": "tailor"}


class _FakeNeeds(NeedExtractor):
    """Canned need extraction."""

    def extract_need_data(self, user_text: str):
        """Return an empty need set."""
        return {"needs": []}


def _orchestrator(llm: _FakeLLM, **kwargs: Any) -> IntelligenceOrchestrator:
    """Orchestrator wired to recording fakes."""
    return IntelligenceOrchestrator(
        profile_processor=ProfileProcessor(llm),
        need_analyzer=NeedAnalyzer(_FakeNeeds()),
        **kwargs,  # type: ignore[arg-type]
    )


def test_nmt_disabled_hindi_text_unchanged() -> None:
    """Default behavior: Hindi text reaches extraction untouched, no slots set."""
    llm = _FakeLLM()
    orch = _orchestrator(llm)
    result = orch.run("मुझे मदद चाहिए", [], source_language="hi")
    assert llm.seen == ["मुझे मदद चाहिए"]
    assert result.source_text is None
    assert result.translated_text is None
    assert "translation" not in result.stages_completed


def test_nmt_enabled_but_no_provider_unchanged() -> None:
    """Flag without a provider is a safe no-op (existing behavior preserved)."""
    llm = _FakeLLM()
    orch = _orchestrator(llm, translation_enabled=True)
    result = orch.run("मुझे मदद चाहिए", [], source_language="hi")
    assert llm.seen == ["मुझे मदद चाहिए"]
    assert result.source_text is None


def test_nmt_enabled_success_routes_translated_retains_source() -> None:
    """Extraction consumes English; original retained in audit slots + stage."""
    llm = _FakeLLM()
    translator = FakeTranslator({"मुझे मदद चाहिए": "I need help"})
    orch = _orchestrator(llm, translation_provider=translator, translation_enabled=True)
    result = orch.run("मुझे मदद चाहिए", [], source_language="hi")
    assert llm.seen == ["I need help"]
    assert result.source_text == "मुझे मदद चाहिए"
    assert result.translated_text == "I need help"
    assert result.stages_completed[0] == "translation"
    assert translator.calls[0]["source_language"] == "hi"
    assert translator.calls[0]["target_language"] == "en"


def test_nmt_enabled_failure_fails_closed() -> None:
    """Translation failure raises typed error with zero fabricated extraction."""
    llm = _FakeLLM()
    translator = FakeTranslator(error=TranslationError("BHASHINI down"))
    orch = _orchestrator(llm, translation_provider=translator, translation_enabled=True)
    with pytest.raises(TranslationError):
        orch.run("मुझे मदद चाहिए", [], source_language="hi")
    assert llm.seen == []  # nothing reached extraction; nothing fabricated


def test_english_source_never_translated() -> None:
    """Explicit English skips NMT even when enabled with a provider."""
    llm = _FakeLLM()
    translator = FakeTranslator()
    orch = _orchestrator(llm, translation_provider=translator, translation_enabled=True)
    result = orch.run("I need help", [], source_language="en")
    assert translator.calls == []
    assert llm.seen == ["I need help"]
    assert result.source_text is None


def test_translated_uncertainty_reaches_extraction_verbatim() -> None:
    """The CONTROLLED translated hedging string (not the original) feeds extraction."""
    llm = _FakeLLM()
    translator = FakeTranslator({"शायद मुझे 5 लाख की जरूरत होगी": "maybe I will need 500000"})
    orch = _orchestrator(llm, translation_provider=translator, translation_enabled=True)
    orch.run("शायद मुझे 5 लाख की जरूरत होगी", [], source_language="hi")
    assert llm.seen == ["maybe I will need 500000"]


class _FakeTranscriber(TranscriptionProvider):
    """Canned transcription backend for voice-flow tests."""

    def __init__(self, text: str = "मुझे मदद चाहिए", language: Any = "hi") -> None:
        """Store canned output."""
        self.text = text
        self.language = language

    def transcribe(self, audio: bytes, language_hint: Any = None):
        """Return the canned transcript."""
        return TranscriptionResult(text=self.text, language=self.language)


def test_voice_flow_translation_converges() -> None:
    """audio -> transcript -> NMT -> existing _run_text; original retained."""
    llm = _FakeLLM()
    translator = FakeTranslator({"मुझे मदद चाहिए": "I need help"})
    orch = _orchestrator(
        llm,
        transcription_provider=_FakeTranscriber(),
        translation_provider=translator,
        translation_enabled=True,
    )
    result = orch.run_from_audio(b"fake-audio", [])
    assert llm.seen == ["I need help"]
    assert result.source_text == "मुझे मदद चाहिए"
    assert result.translated_text == "I need help"


def test_voice_flow_translation_failure_fails_closed() -> None:
    """Voice + NMT failure raises typed error with no extraction."""
    llm = _FakeLLM()
    translator = FakeTranslator(error=TranslationError("down"))
    orch = _orchestrator(
        llm,
        transcription_provider=_FakeTranscriber(),
        translation_provider=translator,
        translation_enabled=True,
    )
    with pytest.raises(TranslationError):
        orch.run_from_audio(b"fake-audio", [])
    assert llm.seen == []


def test_voice_flow_disabled_unchanged() -> None:
    """Voice without NMT keeps the existing transcript-to-extraction path."""
    llm = _FakeLLM()
    orch = _orchestrator(llm, transcription_provider=_FakeTranscriber())
    result = orch.run_from_audio(b"fake-audio", [])
    assert llm.seen == ["मुझे मदद चाहिए"]
    assert result.source_text is None


def test_hybrid_text_path_translation() -> None:
    """run_hybrid text input translates before extraction; original retained."""
    from intelligence_engine.schemas import UserProfile

    llm = _FakeLLM()
    translator = FakeTranslator({"मुझे मदद चाहिए": "I need help"})
    orch = _orchestrator(llm, translation_provider=translator, translation_enabled=True)
    result = orch.run_hybrid(
        UserProfile(), [], user_text="मुझे मदद चाहिए", source_language="hi"
    )
    assert llm.seen == ["I need help"]
    assert result.source_text == "मुझे मदद चाहिए"
    assert result.translated_text == "I need help"
