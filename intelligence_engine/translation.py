"""Text-translation provider abstraction (vendor-neutral language preprocessing).

Flow: source-language text -> TranslationProvider.translate() ->
validated TranslationResult (original + translated) -> existing
ProfileProcessor / NeedAnalyzer consume the translated text ONLY when
the caller explicitly enabled preprocessing and translation succeeded.

The provider translates wording only: no profiling, no need typing,
no invented content. The original text is always preserved inside the
result for auditing, debugging, and evaluation. Failures raise
TranslationError explicitly (fail-closed): callers must never
fabricate English, silently fall back, or silently modify user text.
No language detection lives here; callers pass an explicit source
language. Stdlib-only at import; real vendors load lazily in their
own subclasses.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from intelligence_engine.schemas import TranslationResult


class TranslationError(Exception):
    """Invalid input or provider failure. Never silent, never fabricated."""


class TranslationProvider(ABC):
    """Abstract text-translation backend. Concrete vendors implement this."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str = "en",
    ) -> TranslationResult:
        """Translate text into the target language, preserving the original.

        Args:
            text: Source-language text (non-empty; never modified in place).
            source_language: Explicit source language code (e.g. "hi").
                No detection is performed; empty values are rejected.
            target_language: Target language code (default "en").

        Returns:
            Validated TranslationResult carrying both the original and
            the translated text.

        Raises:
            TranslationError: For empty input, empty/unreported
                translations, or transport/authentication failures.
        """
        raise NotImplementedError


def validate_translation(
    original_text: Any,
    translated_text: Any,
    source_language: Optional[str] = None,
    target_language: str = "en",
) -> TranslationResult:
    """Validate caller-held translation output without any network I/O."""
    if not isinstance(original_text, str):
        raise TranslationError("Original text must be a string.")
    if not isinstance(translated_text, str):
        raise TranslationError("Translated text must be a string.")
    try:
        return TranslationResult(
            original_text=original_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
        )
    except Exception as exc:
        raise TranslationError(f"Invalid translation: {exc}") from exc
