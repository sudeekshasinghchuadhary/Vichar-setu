"""Offline tests for the BHASHINI ASR provider (mocked HTTP, no network/keys).

Covers: BHASHINI API response -> BhashiniTranscriptionProvider ->
TranscriptionResult -> existing voice pipeline. No ASR accuracy is
claimed from these mocked tests.
"""

from typing import Any

import pytest

from intelligence_engine.bhashini_provider import (
    BhashiniConfig,
    BhashiniError,
    BhashiniTranscriptionProvider,
    bhashini_enabled,
    select_transcription_provider,
)
from intelligence_engine.llm_client import LLMClient, NeedExtractor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.orchestrator import IntelligenceOrchestrator
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.transcription import TranscriptionError, TranscriptionProvider


def _config(**overrides: Any) -> BhashiniConfig:
    """Test config without touching the environment."""
    params: dict[str, Any] = {"user_id": "test-user", "api_key": "test-key"}
    params.update(overrides)
    return BhashiniConfig(**params)


def _pipeline(service_id: str = "asr-hi-1") -> dict[str, Any]:
    """Canned ULCA pipeline-discovery response."""
    return {
        "pipelineResponseConfig": [
            {"taskType": "asr", "config": [{"serviceId": service_id}]}
        ],
        "pipelineInferenceAPIEndPoint": {
            "callbackUrl": "https://infer.example/pipeline",
            "inferenceApiKey": {"name": "Authorization", "value": "tok"},
        },
    }


def _infer(text: Any) -> dict[str, Any]:
    """Canned ULCA inference response carrying the transcript."""
    return {"pipelineResponse": [{"taskType": "asr", "output": [{"source": text}]}]}


class FakeHttp:
    """Canned HTTP backend recording every call."""

    def __init__(
        self,
        pipeline: Any = None,
        inference: Any = None,
        error: Any = None,
    ) -> None:
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


def _provider(http: FakeHttp, **overrides: Any) -> BhashiniTranscriptionProvider:
    """Provider wired to the fake HTTP backend."""
    return BhashiniTranscriptionProvider(config=_config(**overrides), http_post=http)


def test_implements_transcription_abstraction() -> None:
    """BHASHINI provider satisfies the existing TranscriptionProvider contract."""
    assert issubclass(BhashiniTranscriptionProvider, TranscriptionProvider)
    result = _provider(FakeHttp()).transcribe(b"fake-audio-bytes")
    assert result.text == "hello world"
    assert result.language is None


def test_language_hint_reaches_asr_request() -> None:
    """Caller hint selects the ULCA source language without changing validation."""
    http = FakeHttp()
    result = _provider(http).transcribe(b"x", language_hint="hi")
    assert result.text == "hello world"
    discovery = http.calls[0]["payload"]
    assert discovery["pipelineTasks"][0]["config"]["language"] == {"sourceLanguage": "hi"}
    inference = http.calls[1]["payload"]
    assert inference["pipelineTasks"][0]["config"]["language"] == {"sourceLanguage": "hi"}
    assert inference["inputData"]["audio"][0]["audioFormat"] == "wav"


def test_empty_transcript_rejected() -> None:
    """Blank BHASHINI output fails through existing validation, never silent."""
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(inference=_infer("   "))).transcribe(b"x")


def test_malformed_responses_rejected() -> None:
    """Bad pipeline/inference shapes raise TranscriptionError, never fabricated text."""
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(pipeline={})).transcribe(b"x")
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(pipeline={"pipelineResponseConfig": []})).transcribe(b"x")
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(inference={})).transcribe(b"x")
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(inference=_infer(None))).transcribe(b"x")
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp()).transcribe(b"")


def test_auth_failure_converted() -> None:
    """Credential/transport failures become TranscriptionError, never raw tracebacks."""
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(error=RuntimeError("401 Unauthorized"))).transcribe(b"x")


def test_network_failure_converted() -> None:
    """Socket-level failures become TranscriptionError."""
    with pytest.raises(TranscriptionError):
        _provider(FakeHttp(error=ConnectionError("boom"))).transcribe(b"x")


def test_whitespace_normalized() -> None:
    """Transcript wording collapses whitespace losslessly like all providers."""
    result = _provider(FakeHttp(inference=_infer("  hello   world  "))).transcribe(b"x")
    assert result.text == "hello world"


def test_language_never_guessed() -> None:
    """ULCA reports no detected language, so none is invented."""
    http = FakeHttp()
    result = _provider(http).transcribe(b"x", language_hint="hi")
    assert result.language is None


class _FakeLLM(LLMClient):
    """Canned profile extraction recording the text it received."""

    def __init__(self) -> None:
        """Track received inputs."""
        self.seen: list[str] = []

    def extract_profile_data(self, user_text: str):
        """Return a fixed profile for the BHASHINI transcript."""
        self.seen.append(user_text)
        return {"age": 28, "occupation": "tailor", "state": "Uttar Pradesh"}


class _FakeNeeds(NeedExtractor):
    """Canned need extraction."""

    def extract_need_data(self, user_text: str):
        """Return an empty need set."""
        return {"needs": []}


def test_orchestrator_voice_flow_with_bhashini() -> None:
    """run_from_audio works unchanged with the injected BHASHINI provider."""
    llm = _FakeLLM()
    orch = IntelligenceOrchestrator(
        profile_processor=ProfileProcessor(llm),
        need_analyzer=NeedAnalyzer(_FakeNeeds()),
        transcription_provider=_provider(FakeHttp()),
    )
    result = orch.run_from_audio(b"fake-audio-bytes", [])
    assert llm.seen == ["hello world"]
    assert result.profile.age == 28
    assert result.profile.occupation == "tailor"


def test_disabled_keeps_existing_provider() -> None:
    """Without the flag, selection returns the current provider untouched."""
    gemini_like = object()
    assert select_transcription_provider(gemini_like, environ={}) is gemini_like
    assert select_transcription_provider(None, environ={"BHASHINI_ENABLED": "false"}) is None
    assert bhashini_enabled({}) is False


def test_enabled_selects_bhashini_provider() -> None:
    """With the flag plus credentials, selection builds the BHASHINI provider."""
    provider = select_transcription_provider(
        None,
        environ={
            "BHASHINI_ENABLED": "true",
            "BHASHINI_USER_ID": "u",
            "BHASHINI_API_KEY": "k",
        },
    )
    assert isinstance(provider, BhashiniTranscriptionProvider)


def test_missing_credentials_rejected_at_construction() -> None:
    """Enabled selection without credentials fails closed with a typed error."""
    with pytest.raises(BhashiniError):
        select_transcription_provider(
            None, environ={"BHASHINI_ENABLED": "1", "BHASHINI_USER_ID": "u"}
        )
