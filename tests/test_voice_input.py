"""Tests for voice input (fake transcription provider, offline)."""

import pytest

from intelligence_engine.llm_client import LLMClient, NeedExtractor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.schemas import TranscriptionResult
from intelligence_engine.transcription import (
    TranscriptionError,
    TranscriptionProvider,
    validate_transcript,
)


class FakeTranscriber(TranscriptionProvider):
    """Canned transcription backend for offline tests."""

    def __init__(self, text="I am 28, a tailor from Uttar Pradesh.", language="en", error=None):
        """Store canned output."""
        self.text = text
        self.language = language
        self.error = error
        self.calls = []

    def transcribe(self, audio, language_hint=None):
        """Return the canned transcript or raise the canned error."""
        self.calls.append({"audio": audio, "language_hint": language_hint})
        if self.error is not None:
            raise self.error
        if not audio:
            raise TranscriptionError("Empty audio payload.")
        return TranscriptionResult(text=self.text, language=self.language)


class FakeLLM(LLMClient):
    """Canned profile extraction."""

    def extract_profile_data(self, user_text):
        """Return a fixed profile dict."""
        return {"age": 28, "occupation": "tailor", "state": "Uttar Pradesh"}


class FakeNeeds(NeedExtractor):
    """Canned need extraction."""

    def extract_need_data(self, user_text):
        """Return a fixed machinery need."""
        return {"needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}]}


def test_provider_abstraction() -> None:
    """Concrete providers implement the transcription contract."""
    assert issubclass(FakeTranscriber, TranscriptionProvider)
    result = FakeTranscriber().transcribe(b"fake-bytes", language_hint="hi")
    assert result.text.startswith("I am 28")
    assert result.language == "en"
    assert FakeTranscriber().transcribe(b"x").language is not None


def test_empty_audio_rejected() -> None:
    """Silence/empty payloads raise instead of producing empty profiles."""
    with pytest.raises(TranscriptionError):
        FakeTranscriber().transcribe(b"")


def test_provider_errors_surface_typed() -> None:
    """Transport failures propagate as TranscriptionError."""
    with pytest.raises(TranscriptionError):
        FakeTranscriber(error=TranscriptionError("mic unavailable")).transcribe(b"x")


def test_schema_rejects_empty_text() -> None:
    """Blank transcripts fail validation; whitespace collapses losslessly."""
    with pytest.raises(Exception):
        TranscriptionResult(text="   ")
    assert TranscriptionResult(text="  hello   world  ").text == "hello world"
    assert TranscriptionResult(text="hi").language is None


def test_validate_transcript_helper() -> None:
    """Caller-held text validates without audio I/O; non-strings rejected."""
    assert validate_transcript(" hello ").text == "hello"
    with pytest.raises(TranscriptionError):
        validate_transcript(42)
    with pytest.raises(TranscriptionError):
        validate_transcript("   ")


def test_transcript_feeds_profile_processor() -> None:
    """Transcript text flows into the existing profile pipeline unchanged."""
    transcript = FakeTranscriber().transcribe(b"audio-bytes")
    profile = ProfileProcessor(FakeLLM()).process_profile(transcript.text)
    assert profile.age == 28
    assert profile.occupation == "tailor"
    assert profile.state == "Uttar Pradesh"


def test_transcript_feeds_need_analyzer() -> None:
    """Transcript text flows into the existing need pipeline unchanged."""
    transcript = FakeTranscriber().transcribe(b"audio-bytes")
    result = NeedAnalyzer(FakeNeeds()).analyze(transcript.text)
    assert result.needs[0].need_type == "machinery"
    assert result.total_requested == 500000


def test_meaning_preserved_verbatim() -> None:
    """Transcription never adds profile/need information."""
    spoken = "Mujhe tailoring ke liye madad chahiye"
    result = FakeTranscriber(text=spoken, language="hi").transcribe(b"x")
    assert result.text == spoken


def test_deterministic_repeated_calls() -> None:
    """Same audio always yields the same transcript."""
    provider = FakeTranscriber()
    assert provider.transcribe(b"x") == provider.transcribe(b"x")


def test_no_llm_database_or_api_surface() -> None:
    """Transcription module holds only the abstraction and validation."""
    import intelligence_engine.transcription as transcription_module

    for forbidden in (
        "LLMClient", "llm_client", "EligibilityEngine", "MatchingEngine",
        "sqlite", "postgres", "requests", "fastapi", "whisper",
    ):
        assert forbidden not in dir(transcription_module)


class _FakeSpeechModels:
    """Stub for SDK generate_content."""

    def __init__(self, text=None, error=None):
        """Store response text or error."""
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, model=None, contents=None, config=None):
        """Record the call and return stubbed text."""
        self.calls.append({"model": model, "contents": contents})
        if self.error is not None:
            raise self.error
        from types import SimpleNamespace

        return SimpleNamespace(text=self.text)


class _FakeSpeechClient:
    """Stub SDK client exposing .models."""

    def __init__(self, text=None, error=None):
        """Build the stub backend."""
        self.models = _FakeSpeechModels(text=text, error=error)


def _gemini_provider(**overrides):
    """GeminiTranscriptionProvider with injected fake client."""
    from intelligence_engine.gemini_provider import GeminiConfig, GeminiTranscriptionProvider

    params = {"config": GeminiConfig(api_key="test-key"), "client": _FakeSpeechClient("hello world")}
    params.update(overrides)
    return GeminiTranscriptionProvider(**params)


def test_gemini_provider_implements_abstraction() -> None:
    """Concrete Gemini provider satisfies TranscriptionProvider."""
    from intelligence_engine.gemini_provider import GeminiTranscriptionProvider

    assert issubclass(GeminiTranscriptionProvider, TranscriptionProvider)
    result = _gemini_provider().transcribe(b"fake-audio")
    assert result.text == "hello world"


def test_language_hint_propagated() -> None:
    """Hint reaches the model call without changing validation."""
    from intelligence_engine.gemini_provider import GeminiConfig, GeminiTranscriptionProvider

    client = _FakeSpeechClient("namaste")
    provider = GeminiTranscriptionProvider(config=GeminiConfig(api_key="k"), client=client)
    assert provider.transcribe(b"x", language_hint="hi").text == "namaste"
    assert any("hi" in str(part) for part in client.models.calls[0]["contents"])


def test_empty_provider_response_rejected() -> None:
    """Blank transcripts fail through existing validation, never silent."""
    from intelligence_engine.transcription import TranscriptionError as TE

    import pytest as _pytest

    with _pytest.raises(TE):
        _gemini_provider(client=_FakeSpeechClient("   ")).transcribe(b"x")


def test_provider_error_converted() -> None:
    """SDK failures become TranscriptionError, never vendor tracebacks."""
    import pytest as _pytest

    from intelligence_engine.transcription import TranscriptionError as TE

    with _pytest.raises(TE):
        _gemini_provider(client=_FakeSpeechClient(error=RuntimeError("boom"))).transcribe(b"x")


def test_empty_audio_rejected_before_call() -> None:
    """Empty payloads raise without touching the SDK."""
    import pytest as _pytest

    from intelligence_engine.transcription import TranscriptionError as TE

    client = _FakeSpeechClient("hi")
    with _pytest.raises(TE):
        _gemini_provider(client=client).transcribe(b"")
    assert client.models.calls == []
