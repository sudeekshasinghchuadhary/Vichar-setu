"""Speech-to-text provider abstraction (vendor-neutral voice input).

Flow: audio bytes -> TranscriptionProvider.transcribe() ->
validated TranscriptionResult -> existing ProfileProcessor /
NeedAnalyzer (callers pass result.text onward as ordinary text).

The provider transcribes wording only: no profiling, no need typing,
no invented content. Empty/silent audio and transport failures raise
TranscriptionError explicitly. Stdlib-only at import; real vendors
load lazily in their own subclasses. No audio is stored anywhere.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from intelligence_engine.schemas import TranscriptionResult


class TranscriptionError(Exception):
    """Empty/invalid transcription or provider failure. Never silent."""


class TranscriptionProvider(ABC):
    """Abstract speech-to-text backend. Concrete vendors implement this."""

    @abstractmethod
    def transcribe(
        self, audio: bytes, language_hint: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes into validated text.

        Args:
            audio: Raw audio payload (e.g. WAV/OGG bytes) as received
                from the caller. Never persisted by the engine.
            language_hint: Optional caller hint (e.g. "hi", "en").
                A hint is a hint: the provider reports what it hears.

        Returns:
            Validated TranscriptionResult with verbatim wording.

        Raises:
            TranscriptionError: For empty audio, empty transcripts, or
                transport/authentication failures.
        """
        raise NotImplementedError


def validate_transcript(text: Any) -> TranscriptionResult:
    """Validate caller-held transcript text without any audio I/O."""
    if not isinstance(text, str):
        raise TranscriptionError("Transcript must be a string.")
    try:
        return TranscriptionResult(text=text)
    except Exception as exc:
        raise TranscriptionError(f"Invalid transcript: {exc}") from exc
