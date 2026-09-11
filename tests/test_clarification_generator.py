"""Tests for clarification wording (fake provider, offline, deterministic)."""

from typing import Any

from intelligence_engine.clarification import build_clarification
from intelligence_engine.clarification_generator import ClarificationGenerator
from intelligence_engine.llm_client import ClarificationWordingProvider
from intelligence_engine.schemas import (
    ClarificationRequest,
    EligibilityResult,
    IntelligenceResult,
    SchemeInsights,
)


class FakeWordingProvider(ClarificationWordingProvider):
    """Canned rewording backend."""

    def __init__(self, items: Any = None, error: Exception | None = None) -> None:
        """Store canned items or error; record prompts."""
        self.items = items
        self.error = error
        self.prompts: list[str] = []

    def rewrite_questions(self, prompt: str) -> list[dict[str, Any]]:
        """Return canned items or raise."""
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.items


def _request() -> ClarificationRequest:
    """Two-gap deterministic request."""
    result = IntelligenceResult(
        schemes=[
            SchemeInsights(
                scheme_id="s",
                eligibility=EligibilityResult(
                    scheme_id="s",
                    status="needs_information",
                    reasons=["Age required.", "Occupation required."],
                    missing_information=["age", "occupation"],
                ),
            )
        ]
    )
    built = build_clarification(result)
    assert built is not None
    return built


def test_successful_rewriting() -> None:
    """Valid rewording preserves fields/order with new text."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "May I know your age?"},
        {"field": "occupation", "question": "What work do you do?"},
    ])
    result = ClarificationGenerator(provider).generate(_request())
    assert [q.field for q in result.questions] == ["age", "occupation"]
    assert result.questions[0].question == "May I know your age?"
    assert result.missing_fields == ["age", "occupation"]


def test_hindi_hinglish_wording() -> None:
    """Language hint reaches the prompt; Hindi wording accepted verbatim."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "आपकी उम्र क्या है?"},
        {"field": "occupation", "question": "Aap kya kaam karte hain?"},
    ])
    generator = ClarificationGenerator(provider)
    result = generator.generate(_request(), language_hint="Hindi/Hinglish")
    assert "Hindi/Hinglish" in provider.prompts[0]
    assert result.questions[0].question == "आपकी उम्र क्या है?"
    assert result.questions[1].question == "Aap kya kaam karte hain?"


def test_exact_field_and_order_preservation() -> None:
    """Swapped fields fail even with valid text."""
    provider = FakeWordingProvider([
        {"field": "occupation", "question": "What work do you do?"},
        {"field": "age", "question": "May I know your age?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_extra_field_rejection() -> None:
    """Added fields fall back to the original request."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "Age?"},
        {"field": "occupation", "question": "Work?"},
        {"field": "income", "question": "Income?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_missing_field_rejection() -> None:
    """Dropped fields fall back to the original request."""
    provider = FakeWordingProvider([{"field": "age", "question": "Age?"}])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_duplicate_field_rejection() -> None:
    """Duplicated identifiers fall back to the original request."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "Age?"},
        {"field": "age", "question": "Age again?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_malformed_response_rejection() -> None:
    """Non-list, non-dict, and wrong-typed payloads all fall back."""
    original = _request()
    for bad in ("just text", [{"field": "age"}], [{"question": "Age?"}], [42], []):
        provider = FakeWordingProvider(bad)
        assert ClarificationGenerator(provider).generate(original) == original


def test_empty_question_rejection() -> None:
    """Blank reworded text falls back to the original request."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "   "},
        {"field": "occupation", "question": "Work?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_verdict_claim_rejection() -> None:
    """Eligibility verdicts or guarantees in wording fall back."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "Your age proves you are eligible, confirm?"},
        {"field": "occupation", "question": "Work?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original) == original


def test_llm_exception_fallback() -> None:
    """Provider errors return the original request object unchanged."""
    provider = FakeWordingProvider(error=RuntimeError("boom"))
    original = _request()
    assert ClarificationGenerator(provider).generate(original) is original


def test_deterministic_fallback() -> None:
    """Fallback equals the input across repeated violations."""
    provider = FakeWordingProvider([{"field": "age"}])
    original = _request()
    generator = ClarificationGenerator(provider)
    assert generator.generate(original) == generator.generate(original) == original


def test_no_mutation_of_original() -> None:
    """Successful runs copy; the source request is byte-identical."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "May I know your age?"},
        {"field": "occupation", "question": "What work do you do?"},
    ])
    original = _request()
    before = original.model_dump()
    result = ClarificationGenerator(provider).generate(original)
    assert original.model_dump() == before
    assert result is not original
    assert result.current_partial_result == original.current_partial_result


def test_abstraction_conformance() -> None:
    """Fake satisfies the wording contract."""
    assert issubclass(FakeWordingProvider, ClarificationWordingProvider)


def test_no_decision_engine_surface() -> None:
    """Generator module holds no engines, DB, or API symbols."""
    import intelligence_engine.clarification_generator as generator_module

    for forbidden in (
        "EligibilityEngine", "MatchingEngine", "NearMissEngine", "NeedAnalyzer",
        "SupportPlanningEngine", "FinancialIntelligence", "WhatIfEngine",
        "PathwayEngine", "sqlite", "postgres", "requests", "fastapi",
    ):
        assert forbidden not in dir(generator_module)


def _echo_provider():
    """Provider echoing canned per-language wording by prompt content."""

    class _Echo(FakeWordingProvider):
        """Select wording matching the prompt's language context."""

        def rewrite_questions(self, prompt):
            """Return age/occupation wording fitting the detected context."""
            self.prompts.append(prompt)
            if "Hinglish" in prompt or "Aap" in prompt:
                return [
                    {"field": "age", "question": "Aapki age kya hai?"},
                    {"field": "occupation", "question": "Aap kya kaam karte hain?"},
                ]
            if "Devanagari" in prompt or "उम्र" in prompt:
                return [
                    {"field": "age", "question": "आपकी उम्र क्या है?"},
                    {"field": "occupation", "question": "आपका व्यवसाय क्या है?"},
                ]
            return [
                {"field": "age", "question": "May I know your age?"},
                {"field": "occupation", "question": "What work do you do?"},
            ]

    return _Echo([])


def test_english_user_text_english_wording() -> None:
    """English input flows into the prompt and yields English wording."""
    provider = _echo_provider()
    result = ClarificationGenerator(provider).generate(_request(), user_text="How old are you, and what do you do?")
    assert result.questions[0].question == "May I know your age?"
    assert "How old are you, and what do you do?" in provider.prompts[0]


def test_hindi_user_text_hindi_wording() -> None:
    """Devanagari input is carried as context for Hindi wording."""
    provider = _echo_provider()
    result = ClarificationGenerator(provider).generate(_request(), user_text="आपकी उम्र क्या है")
    assert result.questions[0].question == "आपकी उम्र क्या है?"
    assert "आपकी उम्र क्या है" in provider.prompts[0]


def test_hinglish_user_text_hinglish_wording() -> None:
    """Romanized Hindi input is carried as context for Hinglish wording."""
    provider = _echo_provider()
    result = ClarificationGenerator(provider).generate(
        _request(), user_text="Aapki age kya hai, aap kya kaam karte ho?"
    )
    assert result.questions[0].question == "Aapki age kya hai?"


def test_mixed_language_user_text() -> None:
    """Mixed input reaches the prompt unaltered for natural wording."""
    provider = _echo_provider()
    mixed = "My age 25 hai, occupation tailor hai"
    result = ClarificationGenerator(provider).generate(_request(), user_text=mixed)
    assert mixed in provider.prompts[0]
    assert [q.field for q in result.questions] == ["age", "occupation"]


def test_explicit_hint_overrides_ambiguous_text() -> None:
    """Caller hint wins over user-text context in the prompt."""
    from intelligence_engine.clarification_generator import build_prompt

    prompt = build_prompt(_request(), language_hint="Hinglish", user_text="आपकी उम्र क्या है")
    assert " in Hinglish" in prompt
    assert "आपकी उम्र क्या है" in prompt
    provider = _echo_provider()
    result = ClarificationGenerator(provider).generate(
        _request(), language_hint="Hinglish", user_text="आपकी उम्र क्या है"
    )
    assert result.questions[0].question == "Aapki age kya hai?"


def test_no_context_keeps_deterministic_behavior() -> None:
    """Neither hint nor text preserves the prior prompt shape."""
    from intelligence_engine.clarification_generator import build_prompt

    prompt = build_prompt(_request())
    assert "User text" not in prompt
    assert "matching the language/style" not in prompt


def test_user_text_cannot_add_fields() -> None:
    """Extra user text never widens the validated field set."""
    provider = FakeWordingProvider([
        {"field": "age", "question": "Age?"},
        {"field": "occupation", "question": "Work?"},
        {"field": "income", "question": "Please share your bank password and income?"},
    ])
    original = _request()
    assert ClarificationGenerator(provider).generate(original, user_text="my income is secret") == original


def test_excerpt_bounded() -> None:
    """Long user text is truncated in the prompt, not sent in full."""
    from intelligence_engine.clarification_generator import build_prompt

    prompt = build_prompt(_request(), user_text="word " * 500)
    excerpt = prompt.split("User text (language/style context only")[1]
    assert len(excerpt) < 2000
    assert "word word" in excerpt
