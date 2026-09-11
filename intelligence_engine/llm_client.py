"""LLM client abstraction for profile extraction (no business logic).

Responsibility: understand/extract information from natural-language
user input and return structured profile data as a plain dict.
The LLM must NOT decide eligibility, recommend schemes, calculate
match scores, or invent missing information. Missing fields should
simply be absent (they become None in UserProfile).

Provider-agnostic: no OpenAI/Gemini/Groq/NVIDIA SDK here and no
API key required. Tests use a fake implementation of LLMClient.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract LLM client. Concrete providers implement this interface."""

    @abstractmethod
    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Extract structured profile fields from natural-language text.

        Args:
            user_text: Raw natural-language user description.

        Returns:
            Plain dict of extracted UserProfile fields. Absent keys
            mean "not provided" (they become None after validation).
        """
        raise NotImplementedError


class NeedExtractor(ABC):
    """Abstract need-analysis backend (separate from profile extraction).

    Kept separate from LLMClient so profile fakes/tests are unaffected
    and each responsibility can evolve independently. Must only extract
    the user's goal and needs — never schemes, eligibility, or amounts
    the user did not state.
    """

    @abstractmethod
    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Extract goal/needs structure from natural-language text.

        Args:
            user_text: Raw natural-language user description.

        Returns:
            Plain dict with optional "business_goal" (str) and "needs"
            (list of dicts with need_type/amount/amount_period/context/
            priority). Absent values mean "not provided".
        """
        raise NotImplementedError


class LLMExplanationProvider(ABC):
    """Abstract explanation-text backend (communication layer only).

    Receives a fully-formed prompt built from a DecisionTrace and
    returns natural-language text. Must NOT decide, calculate, or add
    facts — the ExplanationGenerator validates output against the
    trace and falls back to deterministic wording on any violation.
    """

    @abstractmethod
    def generate_explanation(self, prompt: str) -> str:
        """Render the supplied prompt into natural-language text.

        Args:
            prompt: Prompt built solely from DecisionTrace content plus
                fixed grounding instructions.

        Returns:
            Natural-language explanation draft (validated afterwards).
        """
        raise NotImplementedError


class ClarificationWordingProvider(ABC):
    """Abstract clarification-rewording backend (wording only, no authority).

    Receives a prompt built from an already-built ClarificationRequest
    and returns reworded questions. Must NOT add, remove, reorder, or
    reinterpret fields — the ClarificationGenerator validates the
    response field-for-field and falls back to the deterministic
    request on any violation.
    """

    @abstractmethod
    def rewrite_questions(self, prompt: str) -> list[dict[str, Any]]:
        """Reword supplied questions, preserving fields exactly.

        Args:
            prompt: Prompt built solely from ClarificationRequest content
                plus fixed grounding instructions.

        Returns:
            List of {"field": str, "question": str} dicts in the same
            order and count as the request's questions.
        """
        raise NotImplementedError
