"""SAVED MODEL OUTPUT EVALUATION (offline — NOT live Gemini tests).

Each case pairs a raw user input with a SAVED Gemini-style raw response
dict. The saved dict is fed through the REAL production path
(ProfileProcessor / NeedAnalyzer: validation + normalization + merging)
via a stub extractor, and the final structured result is asserted.

What this establishes (B: normalization correctness):
- saved response -> production pipeline -> expected structured data.

What this does NOT establish (A: extraction correctness):
- whether real Gemini would actually emit the saved response for the
  raw input. That requires the opt-in live suite
  (tests/integration/test_real_gemini_extraction.py, RUN_GEMINI_INTEGRATION=1).

Provider/API behavior (C) remains covered by tests/test_gemini_provider.py.

No test here imports gemini_provider, reads GEMINI_API_KEY, or touches
the network. These run in the normal suite with plain pytest.

Future refresh workflow (no automation implemented): run a handful of
live opt-in extractions, paste the VERIFIED raw response dicts into the
corresponding case below, and re-run this module. Never paste
unreviewed model output as expected truth.
"""

from typing import Any

from intelligence_engine.llm_client import LLMClient, NeedExtractor
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.profile_processor import ProfileProcessor


class SavedProfileExtractor(LLMClient):
    """Stub returning one saved profile response regardless of input."""

    def __init__(self, saved: dict[str, Any]) -> None:
        """Store the saved raw response dict."""
        self.saved = saved

    def extract_profile_data(self, user_text: str) -> dict[str, Any]:
        """Return the saved response (input kept for documentation only)."""
        return self.saved


class SavedNeedExtractor(NeedExtractor):
    """Stub returning one saved need response regardless of input."""

    def __init__(self, saved: dict[str, Any]) -> None:
        """Store the saved raw response dict."""
        self.saved = saved

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return the saved response (input kept for documentation only)."""
        return self.saved


def _process(profile_saved: dict[str, Any], text: str = "saved input"):
    """Run a saved profile response through the production pipeline."""
    return ProfileProcessor(SavedProfileExtractor(profile_saved)).process_profile(text)


def _analyze(needs_saved: dict[str, Any], text: str = "saved input"):
    """Run a saved need response through the production pipeline."""
    return NeedAnalyzer(SavedNeedExtractor(needs_saved)).analyze(text)


def test_saved_english_profile_and_need() -> None:
    """Saved EN response normalizes to the expected profile + need."""
    raw = "I am 28, a tailor from Uttar Pradesh. Annual income Rs 2.8 lakh. Need Rs 5 lakh for machinery."
    profile = _process(
        {
            "age": 28,
            "occupation": "tailor",
            "state": "Uttar Pradesh",
            "annual_family_income": 280000.0,
            "income_period": "annual",
            "purpose": "expand tailoring business",
            "project_type": "micro-enterprise",
        },
        raw,
    )
    assert profile.age == 28
    assert profile.state == "Uttar Pradesh"
    assert profile.annual_family_income == 280000.0
    assert profile.occupation == "tailor"
    result = _analyze(
        {
            "business_goal": "Expand tailoring business",
            "needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}],
        },
        raw,
    )
    assert result.business_goal == "Expand tailoring business"
    assert len(result.needs) == 1
    assert result.needs[0].need_type == "machinery"
    assert result.needs[0].amount == 500000
    assert result.needs[0].amount_period == "one_time"
    assert result.total_requested == 500000


def test_saved_hindi_profile_and_need() -> None:
    """Saved Devanagari-alias response normalizes to canonical values."""
    raw = "मुझे सिलाई व्यवसाय के लिए मशीनों हेतु 2 लाख चाहिए। मैं उत्तर प्रदेश में रहता हूँ।"
    profile = _process(
        {"age": 30, "state": "उत्तर प्रदेश", "social_category": "अनुसूचित जाति"},
        raw,
    )
    assert profile.age == 30
    assert profile.state == "Uttar Pradesh"
    assert profile.social_category == "SC"
    assert profile.annual_family_income is None
    result = _analyze(
        {"needs": [{"need_type": "machinery", "amount": 200000, "amount_period": "one_time"}]},
        raw,
    )
    assert result.needs[0].need_type == "machinery"
    assert result.needs[0].amount == 200000
    assert result.total_requested == 200000


def test_saved_hinglish_profile_and_need() -> None:
    """Saved Latin-script alias response normalizes to canonical values."""
    raw = "mujhe tailoring ke liye 5 lakh chahiye, main UP se hoon, meri age 30 hai"
    profile = _process({"age": 30, "state": "UP", "occupation": "tailor"}, raw)
    assert profile.age == 30
    assert profile.state == "Uttar Pradesh"
    result = _analyze(
        {
            "business_goal": "tailoring business expand karna",
            "needs": [{"need_type": "working_capital", "amount": 15000, "amount_period": "monthly"}],
        },
        raw,
    )
    assert result.needs[0].need_type == "working_capital"
    assert result.needs[0].amount == 15000
    assert result.needs[0].amount_period == "monthly"
    assert result.total_requested is None


def test_saved_lakh_crore_amount_parsing() -> None:
    """Saved currency-string amounts parse to exact numeric values."""
    raw = "Annual income Rs 2.8 lakh; need Rs 1.5 crore for infrastructure."
    profile = _process({"annual_family_income": "Rs 2.8 lakh"}, raw)
    assert profile.annual_family_income == 280000.0
    result = _analyze(
        {"needs": [{"need_type": "infrastructure", "amount": 15000000, "amount_period": "one_time"}]},
        raw,
    )
    assert result.needs[0].amount == 15000000
    assert result.total_requested == 15000000


def test_saved_monthly_income_annualized() -> None:
    """Saved monthly income follows the production annualization contract."""
    raw = "My monthly income is Rs 20000."
    profile = _process({"annual_family_income": 20000, "income_period": "monthly"}, raw)
    assert profile.annual_family_income == 240000.0
    assert profile.income_period == "annual"


def test_saved_incomplete_input_stays_unknown() -> None:
    """Saved sparse response leaves everything unstated as None/unknown."""
    raw = "I need training for stitching work."
    profile = _process({}, raw)
    assert profile.age is None
    assert profile.state is None
    assert profile.occupation is None
    assert profile.annual_family_income is None
    result = _analyze(
        {"needs": [{"need_type": "training", "context": "stitching course"}]}, raw
    )
    assert result.needs[0].amount is None
    assert result.needs[0].amount_period == "unknown"
    assert result.total_requested is None
    assert result.business_goal is None


def test_saved_multi_need_stays_separate() -> None:
    """Saved two-need response keeps distinct types, amounts, and total."""
    raw = "Need 5 lakh for machinery and 3 lakh as working capital."
    result = _analyze(
        {
            "needs": [
                {"need_type": "machines", "amount": 500000, "amount_period": "one_time"},
                {"need_type": "working capital", "amount": 300000, "amount_period": "one_time"},
            ]
        },
        raw,
    )
    assert [n.need_type for n in result.needs] == ["machinery", "working_capital"]
    assert [n.amount for n in result.needs] == [500000, 300000]
    assert result.total_requested == 800000


def test_saved_inference_trap_and_unrecognized_need() -> None:
    """Saved correct output invents no occupation; unknown need kept verbatim."""
    raw = "My brother is a farmer. I need sewing machines for tailoring."
    profile = _process({}, raw)
    assert profile.occupation is None
    result = _analyze(
        {
            "needs": [
                {
                    "need_type": "sewing machine",
                    "amount": 500000,
                    "amount_period": "one_time",
                    "context": "buy sewing machines for tailoring",
                }
            ]
        },
        raw,
    )
    need = result.needs[0]
    assert need.need_type == "sewing machine"
    assert need.amount == 500000
    assert need.context == "buy sewing machines for tailoring"
    assert any("kept as stated" in note for note in result.notes)
