"""BHASHINI ASR transcription provider (opt-in, ASR only).

Implements the existing TranscriptionProvider abstraction using the
Government of India ULCA/Dhruva inference flow:

1. Discover an ASR service: POST {pipeline_url} (ULCA getModelsPipeline)
   with the caller's user ID + API key, asking for an ``asr`` pipeline
   task in the requested source language.
2. Transcribe: POST the returned inference endpoint with the audio
   (base64) and the discovered service ID, then read the transcript
   from the first ASR output.

Scope: speech-to-text only. No translation, transliteration, TTS,
language detection, or output rendering lives here. Transcript text
passes through verbatim into the existing _run_text() pipeline; Gemini
remains responsible for profile/need extraction and normalization.

Configuration (environment, never hardcoded):
    BHASHINI_ENABLED=false      Master switch; anything else keeps the
                                current Gemini transcription path.
    BHASHINI_USER_ID            ULCA user ID (required when enabled).
    BHASHINI_API_KEY            ULCA API key (required when enabled).
    BHASHINI_ASR_SOURCE_LANGUAGE  Default ASR source language when the
                                caller passes no language_hint
                                (default "hi" for the Hindi-first
                                prototype; a per-call hint always wins).
    BHASHINI_ASR_AUDIO_FORMAT   Audio container label sent with the
                                request (default "wav", matching the
                                engine's WAV/OGG-bytes representation).

Default behavior is unchanged: BHASHINI is used only when explicitly
enabled and only through select_transcription_provider(); the
Intelligence Engine never knows which provider produced a transcript.

No language-coverage claim is made here: whether a source language has
an ASR model is decided by ULCA at discovery time, and discovery
failure surfaces as TranscriptionError (fail-closed).

Stdlib-only HTTP (urllib); no new dependencies.
"""

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from intelligence_engine.schemas import TranscriptionResult
from intelligence_engine.transcription import (
    TranscriptionError,
    TranscriptionProvider,
    validate_transcript,
)

BHASHINI_ENABLED_ENV_VAR = "BHASHINI_ENABLED"
DEFAULT_PIPELINE_URL = (
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
)
DEFAULT_SOURCE_LANGUAGE = "hi"
DEFAULT_AUDIO_FORMAT = "wav"
_REQUEST_TIMEOUT_SECONDS = 30


class BhashiniError(Exception):
    """BHASHINI configuration, transport, or malformed-response failure.

    Never carries fabricated data; transcribe() converts this into the
    existing TranscriptionError contract.
    """


@dataclass(frozen=True)
class BhashiniConfig:
    """BHASHINI ASR configuration (credentials from environment, never committed)."""

    user_id: str
    api_key: str
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    audio_format: str = DEFAULT_AUDIO_FORMAT
    pipeline_url: str = DEFAULT_PIPELINE_URL

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "BhashiniConfig":
        """Read BHASHINI_USER_ID (required) and BHASHINI_API_KEY (required).

        Precedence: explicit environ mapping > process environment.
        Optional overrides: BHASHINI_ASR_SOURCE_LANGUAGE,
        BHASHINI_ASR_AUDIO_FORMAT. No .env file is read here.
        """
        base: dict[str, str] = dict(os.environ)
        if environ is not None:
            base.update(environ)
        user_id = (base.get("BHASHINI_USER_ID") or "").strip()
        api_key = (base.get("BHASHINI_API_KEY") or "").strip()
        if not user_id or not api_key:
            raise BhashiniError(
                "BHASHINI_USER_ID and BHASHINI_API_KEY must both be set "
                "before constructing a BHASHINI provider."
            )
        source_language = (
            base.get("BHASHINI_ASR_SOURCE_LANGUAGE") or ""
        ).strip() or DEFAULT_SOURCE_LANGUAGE
        audio_format = (
            base.get("BHASHINI_ASR_AUDIO_FORMAT") or ""
        ).strip() or DEFAULT_AUDIO_FORMAT
        pipeline_url = (base.get("BHASHINI_PIPELINE_URL") or "").strip() or DEFAULT_PIPELINE_URL
        return cls(
            user_id=user_id,
            api_key=api_key,
            source_language=source_language,
            audio_format=audio_format,
            pipeline_url=pipeline_url,
        )


def bhashini_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True only when BHASHINI_ENABLED is explicitly truthy (default off)."""
    base: dict[str, str] = dict(os.environ)
    if environ is not None:
        base.update(environ)
    return (base.get(BHASHINI_ENABLED_ENV_VAR) or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def select_transcription_provider(
    current: Optional[TranscriptionProvider] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[TranscriptionProvider]:
    """Return the BHASHINI provider when enabled, else `current` unchanged.

    Disabled (default) preserves the existing Gemini transcription path
    exactly: whatever provider was already wired keeps working.
    """
    if not bhashini_enabled(environ):
        return current
    return BhashiniTranscriptionProvider(config=BhashiniConfig.from_env(environ))


HttpPost = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]
"""POST JSON and return the parsed object (injectable for offline tests)."""


def _urllib_post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    """POST a JSON body with the stdlib and return the parsed object."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise BhashiniError(
            f"BHASHINI HTTP call failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BhashiniError(f"BHASHINI returned malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BhashiniError("BHASHINI response must be a JSON object.")
    return parsed


def _source_language(language_hint: Optional[str], default: str) -> str:
    """Map the caller hint to a ULCA source-language code.

    The TranscriptionProvider contract passes BCP-47-ish hints ("hi",
    "en"); ULCA expects ISO-639-1 codes, so only the primary subtag is
    sent ("hi-IN" -> "hi"). A missing hint falls back to the configured
    default. Mapping stays inside the provider; the public contract is
    untouched.
    """
    hint = (language_hint or "").strip().lower()
    if hint:
        return hint.replace("_", "-").split("-")[0]
    return default


class BhashiniTranscriptionProvider(TranscriptionProvider):
    """TranscriptionProvider backed by BHASHINI ULCA/Dhruva ASR.

    Stateless per call: discovers an ASR service for the requested
    source language, then runs inference on the audio bytes. Anything
    unexpected (transport, auth, pipeline shape, empty transcript)
    becomes TranscriptionError; nothing is fabricated, translated, or
    silently retried through another service.
    """

    def __init__(
        self,
        config: Optional[BhashiniConfig] = None,
        http_post: Optional[HttpPost] = None,
    ) -> None:
        """Store config; HTTP stays injectable so unit tests use fakes."""
        self.config = config or BhashiniConfig.from_env()
        self._http_post = http_post or _urllib_post_json

    def _discover_service(self, source_language: str) -> tuple[str, str, str, str]:
        """Return (service_id, endpoint, auth_name, auth_value) for ASR."""
        try:
            response = self._http_post(
                self.config.pipeline_url,
                {
                    "pipelineTasks": [
                        {
                            "taskType": "asr",
                            "config": {"language": {"sourceLanguage": source_language}},
                        }
                    ]
                },
                {"userID": self.config.user_id, "ulcaApiKey": self.config.api_key},
            )
        except BhashiniError:
            raise
        except Exception as exc:
            raise BhashiniError(
                f"BHASHINI pipeline discovery failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            task = response["pipelineResponseConfig"][0]
            service_id = task["config"][0]["serviceId"]
            endpoint = response["pipelineInferenceAPIEndPoint"]["callbackUrl"]
            auth = response["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]
            auth_name, auth_value = auth["name"], auth["value"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BhashiniError(
                f"BHASHINI pipeline response has no usable ASR service: {exc}"
            ) from exc
        if not all(isinstance(v, str) and v for v in (service_id, endpoint, auth_name, auth_value)):
            raise BhashiniError("BHASHINI pipeline response has no usable ASR service.")
        return service_id, endpoint, auth_name, auth_value

    def _run_inference(
        self, audio: bytes, source_language: str, service_id: str, endpoint: str, auth: dict[str, str]
    ) -> str:
        """Run ASR inference and return the raw transcript string."""
        encoded = base64.b64encode(bytes(audio)).decode("ascii")
        try:
            response = self._http_post(
                endpoint,
                {
                    "pipelineTasks": [
                        {
                            "taskType": "asr",
                            "config": {
                                "language": {"sourceLanguage": source_language},
                                "serviceId": service_id,
                            },
                        }
                    ],
                    "inputData": {
                        "audio": [
                            {
                                "audioContent": encoded,
                                "audioFormat": self.config.audio_format,
                            }
                        ]
                    },
                },
                auth,
            )
        except BhashiniError:
            raise
        except Exception as exc:
            raise BhashiniError(
                f"BHASHINI inference call failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            text = response["pipelineResponse"][0]["output"][0]["source"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BhashiniError(
                f"BHASHINI inference response carries no transcript: {exc}"
            ) from exc
        if not isinstance(text, str):
            raise BhashiniError("BHASHINI transcript must be a string.")
        return text

    def transcribe(
        self, audio: bytes, language_hint: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio bytes via BHASHINI; failures become TranscriptionError."""
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            raise TranscriptionError("Empty audio payload.")
        source_language = _source_language(language_hint, self.config.source_language)
        try:
            service_id, endpoint, auth_name, auth_value = self._discover_service(
                source_language
            )
            text = self._run_inference(
                audio, source_language, service_id, endpoint, {auth_name: auth_value}
            )
        except TranscriptionError:
            raise
        except BhashiniError as exc:
            raise TranscriptionError(str(exc)) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Bhashini transcription call failed: {type(exc).__name__}: {exc}"
            ) from exc
        return validate_transcript(text)
