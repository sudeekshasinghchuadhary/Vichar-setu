"""Google Gemini provider behind the intelligence abstractions.

Implements LLMClient (profile extraction) and NeedExtractor (need
extraction) as TWO separate Gemini API calls sharing only transport
code. Domain validation stays in ProfileProcessor/NeedAnalyzer; this
module only extracts, parses, and reports transport problems.

Configuration follows the backend .env convention (same variable
names, no secrets in code):
    GEMINI_API_KEY  (required)
    GEMINI_MODEL    (default: gemini-3.6-flash)

Stdlib-only at import time: google-genai loads lazily on first real
call, so unit tests and offline environments never need the SDK.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from intelligence_engine.llm_client import LLMClient, NeedExtractor
from intelligence_engine.need_vocabulary import CANONICAL_NEED_TYPES
from intelligence_engine.transcription import (
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
    validate_transcript,
)

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiError(Exception):
    """Transport, authentication, configuration, or malformed-response failure.

    Never carries fabricated data; callers surface this instead of guessing.
    """


@dataclass(frozen=True)
class GeminiConfig:
    """Provider configuration (key from environment, never committed)."""

    api_key: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        dotenv_path: Optional[str] = None,
    ) -> "GeminiConfig":
        """Read GEMINI_API_KEY (required) and GEMINI_MODEL (optional).

        Precedence: explicit environ mapping > process environment >
        .env file (same KEY=VALUE format as the backend convention).
        The file is read only when no explicit mapping is given.
        """
        file_values = _read_dotenv(dotenv_path) if environ is None else {}
        base = dict(file_values)
        base.update(os.environ)
        if environ is not None:
            base.update(environ)
        key = (base.get("GEMINI_API_KEY") or "").strip()
        if not key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. Export it (e.g. via the backend .env) "
                "before constructing a real Gemini provider."
            )
        model = (base.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL
        return cls(api_key=key, model=model)


def _read_dotenv(path: Optional[str] = None) -> dict[str, str]:
    """Parse a KEY=VALUE .env file; missing/unreadable files yield {}.

    Minimal reader for the backend .env convention (comments, blanks,
    and single/double quotes handled). Never raises for file problems.
    """
    candidate = path or os.path.join(os.getcwd(), ".env")
    values: dict[str, str] = {}
    try:
        with open(candidate, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        name = name.strip()
        val = raw.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if name:
            values[name] = val
    return values


_PROFILE_FIELDS = (
    "age", "gender", "social_category", "state", "district", "occupation",
    "annual_family_income", "income_period", "purpose", "project_type",
    "estimated_project_cost", "education_level",
)

_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "age": {"type": "integer"},
        "gender": {"type": "string"},
        "social_category": {"type": "string"},
        "state": {"type": "string"},
        "district": {"type": "string"},
        "occupation": {"type": "string"},
        "annual_family_income": {"type": "number"},
        "income_period": {"type": "string", "enum": ["monthly", "annual", "unknown"]},
        "purpose": {"type": "string"},
        "project_type": {"type": "string"},
        "estimated_project_cost": {"type": "number"},
        "education_level": {"type": "string"},
    },
}

_NEED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "business_goal": {"type": "string"},
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "need_type": {"type": "string"},
                    "amount": {"type": "number"},
                    "amount_period": {
                        "type": "string",
                        "enum": ["one_time", "monthly", "annual", "unknown"],
                    },
                    "context": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["need_type"],
            },
        },
    },
}

_PROFILE_PROMPT = """Extract only information explicitly stated by the user as JSON
with any of these fields: {fields}.
Understand English, Hindi (Devanagari), Hinglish, and transliterated Hindi.
Rules: Omit any field that is not explicitly stated; never infer occupation,
education_level, gender, social_category, state, district, purpose, or
project_type from context, name, or other fields; write income_period
as "monthly" when the wording is monthly (e.g. per month, monthly, mahine),
"annual" when yearly, and "unknown" when ambiguous — never invent annual;
amounts as plain numbers (lakh = 100000, crore = 10000000); do not decide
eligibility, ranking, schemes, scores, benefits, or calculations.

----- USER INPUT START -----
{text}
----- USER INPUT END -----"""

_NEED_PROMPT = """Extract the user's business goal and support needs as JSON with
"business_goal" (optional string) and "needs" (array of need_type, amount,
amount_period, context, priority). Canonical need types: {types}.
Understand English, Hindi (Devanagari), Hinglish, and transliterated Hindi.
Rules: extract only stated information; if a genuine support need is
stated but its kind is unclear, use need_type "other" rather than guessing
the nearest canonical type; amounts as plain numbers;
amount_period "monthly"/"annual" only when explicit else omit it; priority
only when explicitly stated; never invent needs, schemes, eligibility,
benefits, or calculations.

----- USER INPUT START -----
{text}
----- USER INPUT END -----"""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences around model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _call_json(
    client: Any, model: str, prompt: str, schema: dict[str, Any]
) -> dict[str, Any]:
    """One Gemini JSON call; returns a plain dict or raises GeminiError."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": schema},
        )
        text = response.text
    except GeminiError:
        raise
    except Exception as exc:
        raise GeminiError(f"Gemini API call failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise GeminiError("Gemini returned an empty response.")
    try:
        parsed = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise GeminiError(f"Gemini returned malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GeminiError("Gemini response must be a JSON object.")
    return parsed


def _real_client(api_key: str) -> Any:
    """Lazily construct the SDK client (import deferred to first use)."""
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiError(
            "google-genai is not installed. Install the optional provider "
            "dependency to use Gemini; unit tests use fakes."
        ) from exc
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise GeminiError(f"Could not initialize Gemini client: {exc}") from exc


class GeminiProfileClient(LLMClient):
    """LLMClient backed by one Gemini extraction call per input."""

    def __init__(self, config: Optional[GeminiConfig] = None, client: Any = None) -> None:
        """Store config; the SDK client builds lazily unless injected (tests)."""
        self.config = config or GeminiConfig.from_env()
        self._client = client

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Extract profile fields; monthly/unknown periods preserved as stated."""
        if self._client is None:
            self._client = _real_client(self.config.api_key)
        prompt = _PROFILE_PROMPT.format(fields=", ".join(_PROFILE_FIELDS), text=user_text)
        return _call_json(self._client, self.config.model, prompt, _PROFILE_SCHEMA)


class GeminiNeedExtractor(NeedExtractor):
    """NeedExtractor backed by one Gemini extraction call per input."""

    def __init__(self, config: Optional[GeminiConfig] = None, client: Any = None) -> None:
        """Store config; the SDK client builds lazily unless injected (tests)."""
        self.config = config or GeminiConfig.from_env()
        self._client = client

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Extract goal/needs; unknown information stays absent."""
        if self._client is None:
            self._client = _real_client(self.config.api_key)
        prompt = _NEED_PROMPT.format(types=", ".join(CANONICAL_NEED_TYPES), text=user_text)
        return _call_json(self._client, self.config.model, prompt, _NEED_SCHEMA)


TRANSCRIPTION_MODEL_ENV_VAR = "GEMINI_TRANSCRIPTION_MODEL"

_TRANSCRIBE_INSTRUCTION = (
    "Transcribe the speech in the attached audio verbatim as plain text. "
    "Do not translate, summarize, profile, or add information."
)


class GeminiTranscriptionProvider(TranscriptionProvider):
    """TranscriptionProvider using Gemini audio input (generate_content).

    No dedicated transcription endpoint exists in google-genai 1.x, so
    audio travels as an inline part to a regular model call. The model
    defaults to the extraction model and is configurable per call site:
    explicit constructor arg > GEMINI_TRANSCRIPTION_MODEL > config.model.
    Transcription only: no profiling, typing, or reasoning here.
    """

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        client: Any = None,
        model: Optional[str] = None,
        audio_mime_type: str = "audio/wav",
    ) -> None:
        """Store config; SDK client and model resolve lazily/at call time."""
        self.config = config or GeminiConfig.from_env()
        self._client = client
        self._model = model
        self.audio_mime_type = audio_mime_type

    @property
    def model(self) -> str:
        """Effective transcription model name."""
        explicit = (self._model or "").strip()
        if explicit:
            return explicit
        env_model = os.environ.get(TRANSCRIPTION_MODEL_ENV_VAR, "").strip()
        return env_model or self.config.model

    def transcribe(
        self, audio: bytes, language_hint: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio bytes; failures become TranscriptionError."""
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            raise TranscriptionError("Empty audio payload.")
        instruction = _TRANSCRIBE_INSTRUCTION
        if language_hint and language_hint.strip():
            instruction += f" The speaker may be using: {language_hint.strip()}."
        if self._client is None:
            try:
                self._client = _real_client(self.config.api_key)
            except GeminiError as exc:
                raise TranscriptionError(str(exc)) from exc
            try:
                from google.genai import types
            except ImportError as exc:
                raise TranscriptionError(
                    "google-genai is not installed. Install the optional provider "
                    "dependency to use Gemini; unit tests use fakes."
                ) from exc
            contents: list[Any] = [
                types.Part.from_bytes(data=bytes(audio), mime_type=self.audio_mime_type),
                instruction,
            ]
        else:
            contents = [bytes(audio), instruction]
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            text = response.text
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Gemini transcription call failed: {type(exc).__name__}: {exc}"
            ) from exc
        return validate_transcript(text)
