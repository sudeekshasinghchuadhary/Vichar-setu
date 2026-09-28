"""Offline tests for the Gemini provider (fake SDK client, no network/keys)."""

from types import SimpleNamespace

import pytest

from intelligence_engine.gemini_provider import (
    DEFAULT_MODEL,
    GeminiConfig,
    GeminiError,
    GeminiNeedExtractor,
    GeminiProfileClient,
)
from intelligence_engine.llm_client import LLMClient, NeedExtractor


class _FakeModels:
    """Stub for client.models.generate_content."""

    def __init__(self, text=None, error=None):
        """Store the response text or error to raise."""
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, model=None, contents=None, config=None):
        """Record the call and return the stubbed response."""
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


class _FakeClient:
    """Stub SDK client exposing .models."""

    def __init__(self, text=None, error=None):
        """Build the stub models backend."""
        self.models = _FakeModels(text=text, error=error)


def _config() -> GeminiConfig:
    """Test config without touching the environment."""
    return GeminiConfig(api_key="test-key")


def test_implements_abstractions() -> None:
    """Provider classes satisfy the intelligence contracts."""
    assert issubclass(GeminiProfileClient, LLMClient)
    assert issubclass(GeminiNeedExtractor, NeedExtractor)


def test_profile_extraction_returns_plain_dict() -> None:
    """Profile call returns the parsed dict with one model call."""
    client = _FakeClient('{"age": 25, "income_period": "monthly", "annual_family_income": 20000}')
    result = GeminiProfileClient(_config(), client=client).extract_profile_data("hi")
    assert result == {"age": 25, "income_period": "monthly", "annual_family_income": 20000}
    assert client.models.calls[0]["model"] == DEFAULT_MODEL
    assert "income_period" in client.models.calls[0]["contents"]


def test_need_extraction_preserves_unknowns() -> None:
    """Need call keeps absent values absent; no invention."""
    client = _FakeClient('{"business_goal": "Expand", "needs": [{"need_type": "training"}]}')
    result = GeminiNeedExtractor(_config(), client=client).extract_need_data("hi")
    assert result["needs"] == [{"need_type": "training"}]
    assert "amount" not in result["needs"][0]


def test_separate_calls_for_profile_and_needs() -> None:
    """Profile and need extraction issue independent API calls."""
    profile_client = _FakeClient('{"age": 30}')
    need_client = _FakeClient('{"needs": []}')
    GeminiProfileClient(_config(), client=profile_client).extract_profile_data("hi")
    GeminiNeedExtractor(_config(), client=need_client).extract_need_data("hi")
    assert len(profile_client.models.calls) == 1
    assert len(need_client.models.calls) == 1
    assert profile_client.models.calls[0]["contents"] != need_client.models.calls[0]["contents"]


def test_transport_error_raises_gemini_error() -> None:
    """SDK failures surface as GeminiError, never fabricated data."""
    client = _FakeClient(error=RuntimeError("boom"))
    with pytest.raises(GeminiError):
        GeminiProfileClient(_config(), client=client).extract_profile_data("hi")


def test_malformed_and_non_object_responses_rejected() -> None:
    """Bad JSON, empty text, and arrays all raise GeminiError."""
    with pytest.raises(GeminiError):
        GeminiProfileClient(_config(), client=_FakeClient("not json")).extract_profile_data("hi")
    with pytest.raises(GeminiError):
        GeminiProfileClient(_config(), client=_FakeClient("   ")).extract_profile_data("hi")
    with pytest.raises(GeminiError):
        GeminiProfileClient(_config(), client=_FakeClient("[1, 2]")).extract_profile_data("hi")
    with pytest.raises(GeminiError):
        GeminiNeedExtractor(_config(), client=_FakeClient(None)).extract_need_data("hi")


def test_fenced_json_accepted() -> None:
    """Markdown-fenced model output still parses."""
    result = GeminiProfileClient(
        _config(), client=_FakeClient('```json\n{"age": 40}\n```')
    ).extract_profile_data("hi")
    assert result == {"age": 40}


def test_config_from_env_and_missing_key() -> None:
    """Env config reads key/model; absent key fails clearly."""
    config = GeminiConfig.from_env({"GEMINI_API_KEY": "k", "GEMINI_MODEL": "custom"})
    assert config == GeminiConfig(api_key="k", model="custom")
    assert GeminiConfig.from_env({"GEMINI_API_KEY": "k"}).model == DEFAULT_MODEL
    with pytest.raises(GeminiError):
        GeminiConfig.from_env({})
    with pytest.raises(GeminiError):
        GeminiConfig.from_env({"GEMINI_API_KEY": "  "})


def test_construction_without_key_fails_fast() -> None:
    """Real construction without configuration raises, never calls network."""
    with pytest.raises(GeminiError):
        GeminiProfileClient(GeminiConfig.from_env({}))


def test_deterministic_repeated_calls() -> None:
    """Same stub yields identical results."""
    client = _FakeClient('{"age": 25}')
    provider = GeminiProfileClient(_config(), client=client)
    assert provider.extract_profile_data("hi") == provider.extract_profile_data("hi")


def test_dotenv_file_supplies_key_outside_backend(monkeypatch) -> None:
    """A cwd .env file works when the process environment lacks the key."""
    import os

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as handle:
            handle.write('# comment line\n\nGEMINI_API_KEY="file-key"\nGEMINI_MODEL=file-model\n')
        config = GeminiConfig.from_env(dotenv_path=os.path.join(tmp, ".env"))
        assert config == GeminiConfig(api_key="file-key", model="file-model")


def test_process_env_beats_dotenv_file() -> None:
    """Explicit process values win over file values; explicit mapping wins all."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, ".env")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("GEMINI_API_KEY=file-key\nGEMINI_MODEL=file-model\n")
        old_key, old_model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL")
        os.environ["GEMINI_API_KEY"] = "env-key"
        try:
            assert GeminiConfig.from_env(dotenv_path=path).api_key == "env-key"
            assert GeminiConfig.from_env({"GEMINI_API_KEY": "map-key"}).api_key == "map-key"
        finally:
            if old_key is None:
                del os.environ["GEMINI_API_KEY"]
            else:
                os.environ["GEMINI_API_KEY"] = old_key
            if old_model is not None:
                os.environ["GEMINI_MODEL"] = old_model
        assert GeminiConfig.from_env({"GEMINI_API_KEY": "k"}, dotenv_path=path).model == DEFAULT_MODEL


def test_missing_dotenv_file_is_not_an_error() -> None:
    """Absent .env falls back to environment behavior unchanged."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(GeminiError):
            GeminiConfig.from_env(dotenv_path=os.path.join(tmp, ".env"))


def test_prompts_delimit_user_input() -> None:
    """Both prompts wrap user text in explicit delimiters."""
    profile_client = _FakeClient('{"age": 25}')
    GeminiProfileClient(_config(), client=profile_client).extract_profile_data("hello user")
    prompt = profile_client.models.calls[0]["contents"]
    assert "----- USER INPUT START -----\nhello user\n----- USER INPUT END -----" in prompt
    need_client = _FakeClient('{"needs": []}')
    GeminiNeedExtractor(_config(), client=need_client).extract_need_data("hello user")
    need_prompt = need_client.models.calls[0]["contents"]
    assert "----- USER INPUT START -----\nhello user\n----- USER INPUT END -----" in need_prompt
    assert need_prompt != prompt


def test_default_model_is_current() -> None:
    """Default model tracks the supported generation."""
    assert DEFAULT_MODEL == "gemini-3.6-flash"
    assert GeminiConfig(api_key="k").model == "gemini-3.6-flash"


def test_profile_prompt_forbids_inference() -> None:
    """Profile prompt requires explicit-only extraction per field."""
    from intelligence_engine.gemini_provider import _PROFILE_PROMPT

    prompt = _PROFILE_PROMPT.format(fields="age", text="hi")
    assert "Omit any field that is not explicitly stated" in prompt
    for field in (
        "occupation", "education_level", "gender", "social_category",
        "state", "district", "purpose", "project_type",
    ):
        assert field in prompt
    assert "never infer" in prompt.lower()
    assert '"monthly"' in prompt and '"unknown"' in prompt


def test_need_prompt_prefers_other_over_guessing() -> None:
    """Need prompt directs unclear kinds to other, not nearest canonical."""
    from intelligence_engine.gemini_provider import _NEED_PROMPT

    prompt = _NEED_PROMPT.format(types="machinery", text="hi")
    assert 'need_type "other"' in prompt
    assert "rather than guessing" in prompt
