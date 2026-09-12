"""Opt-in real-Gemini extraction spot-check (NOT part of CI).

Runs only when RUN_GEMINI_INTEGRATION=1 with GEMINI_API_KEY set and the
google-genai SDK installed. Calls the REAL Gemini extraction provider
(no canned outputs) through the production ProfileProcessor/NeedAnalyzer
path, asserting only stable semantic/structural facts (numeric amounts,
periods, canonical vocabulary, explicit absence) — never exact free-form
wording.

METHODOLOGICAL BOUNDARY: this is an initial real-model spot-check of
about ten representative inputs. Passing these cases does NOT establish
production accuracy, multilingual accuracy percentages, statistical
reliability, benchmark performance, or formal Hindi/Hinglish validation.
A defensible reliability claim requires the human-judged evaluation set
described in the Profile & Need audit, not this module.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_INTEGRATION") != "1",
    reason="Set RUN_GEMINI_INTEGRATION=1 with GEMINI_API_KEY to run real-model spot-checks.",
)

from intelligence_engine.gemini_provider import (
    GeminiError,
    GeminiNeedExtractor,
    GeminiProfileClient,
)
from intelligence_engine.need_analyzer import NeedAnalyzer
from intelligence_engine.profile_processor import ProfileProcessor


@pytest.fixture(scope="module")
def profile_processor() -> ProfileProcessor:
    """Real Gemini-backed profile processor (skips without SDK/key)."""
    pytest.importorskip("google.genai")
    try:
        return ProfileProcessor(GeminiProfileClient())
    except GeminiError as exc:
        pytest.skip(f"No Gemini credentials: {exc}")


@pytest.fixture(scope="module")
def need_analyzer() -> NeedAnalyzer:
    """Real Gemini-backed need analyzer (skips without SDK/key)."""
    pytest.importorskip("google.genai")
    try:
        return NeedAnalyzer(GeminiNeedExtractor())
    except GeminiError as exc:
        pytest.skip(f"No Gemini credentials: {exc}")


def _profile(text: str, processor: ProfileProcessor):
    """Extract a profile, showing input and output on failure."""
    profile = processor.process_profile(text)
    assert profile is not None, f"input={text!r}"
    return profile


def _needs(text: str, analyzer: NeedAnalyzer):
    """Analyze needs, showing input and output on failure."""
    result = analyzer.analyze(text)
    assert result is not None, f"input={text!r}"
    return result


def test_real_english_profile_and_need(profile_processor, need_analyzer) -> None:
    """English age/state/income/need extract to stable structured facts."""
    text = (
        "I am 28 years old, a tailor from Uttar Pradesh. "
        "My annual family income is Rs 2.8 lakh. I need Rs 5 lakh for machinery."
    )
    profile = _profile(text, profile_processor)
    assert profile.age == 28, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state == "Uttar Pradesh", f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.annual_family_income == 280000.0, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.occupation is not None and "tailor" in profile.occupation.lower(), (
        f"input={text!r} actual={profile.model_dump()!r}"
    )
    needs = _needs(text, need_analyzer)
    match = [n for n in needs.needs if n.need_type == "machinery"]
    assert match, f"input={text!r} actual={needs.model_dump()!r}"
    assert match[0].amount == 500000.0, f"input={text!r} actual={needs.model_dump()!r}"


def test_real_english_missing_fields_stay_absent(profile_processor, need_analyzer) -> None:
    """Omitted profile fields stay None; amount-less need stays unknown."""
    text = "I need training for stitching work."
    profile = _profile(text, profile_processor)
    assert profile.age is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.annual_family_income is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.occupation is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.purpose is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.project_type is None, f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    assert needs.business_goal is None, f"input={text!r} actual={needs.model_dump()!r}"
    training = [n for n in needs.needs if n.need_type == "training"]
    assert training, f"input={text!r} actual={needs.model_dump()!r}"
    assert training[0].amount is None, f"input={text!r} actual={needs.model_dump()!r}"
    assert training[0].amount_period == "unknown", f"input={text!r} actual={needs.model_dump()!r}"


def test_real_hindi_devanagari_profile_and_need(profile_processor, need_analyzer) -> None:
    """Devanagari age/state/amount extract to stable structured facts."""
    text = (
        "मुझे सिलाई व्यवसाय के लिए मशीनों हेतु 2 लाख चाहिए। "
        "मैं उत्तर प्रदेश में रहता हूँ। मेरी उम्र 30 साल है।"
    )
    profile = _profile(text, profile_processor)
    assert profile.age == 30, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state == "Uttar Pradesh", f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    assert needs.needs, f"input={text!r} actual={needs.model_dump()!r}"
    assert any(n.amount == 200000.0 for n in needs.needs), (
        f"input={text!r} actual={needs.model_dump()!r}"
    )


def test_real_hindi_financial_expression(profile_processor) -> None:
    """Devanagari annual income wording becomes the correct numeric amount."""
    text = "मेरी वार्षिक पारिवारिक आय 3 लाख रुपये है।"
    profile = _profile(text, profile_processor)
    assert profile.annual_family_income == 300000.0, (
        f"input={text!r} actual={profile.model_dump()!r}"
    )
    assert profile.income_period in ("annual", "unknown"), (
        f"input={text!r} actual={profile.model_dump()!r}"
    )


def test_real_hinglish_profile_and_need(profile_processor, need_analyzer) -> None:
    """Latin-script Hindi age/state/amount extract to stable facts."""
    text = "mujhe tailoring ke liye 5 lakh chahiye, main UP se hoon, meri age 30 hai"
    profile = _profile(text, profile_processor)
    assert profile.age == 30, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state == "Uttar Pradesh", f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    assert needs.needs, f"input={text!r} actual={needs.model_dump()!r}"
    assert any(n.amount == 500000.0 for n in needs.needs), (
        f"input={text!r} actual={needs.model_dump()!r}"
    )


def test_real_hinglish_colloquial_business_no_invention(profile_processor, need_analyzer) -> None:
    """Colloquial need kept without inventing amounts or profile facts."""
    text = "tailoring ka kaam expand karna hai, sewing machine leni hai"
    profile = _profile(text, profile_processor)
    assert profile.age is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state is None, f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    assert needs.needs, f"input={text!r} actual={needs.model_dump()!r}"
    for need in needs.needs:
        assert need.need_type, f"input={text!r} actual={needs.model_dump()!r}"
        assert need.amount is None, (
            f"input={text!r} actual={needs.model_dump()!r}"
        )


def test_real_lakh_crore_amounts(profile_processor, need_analyzer) -> None:
    """Lakh and crore expressions become exact numeric amounts."""
    text = (
        "My annual family income is Rs 2.8 lakh. "
        "I need Rs 1.5 crore for infrastructure development."
    )
    profile = _profile(text, profile_processor)
    assert profile.annual_family_income == 280000.0, (
        f"input={text!r} actual={profile.model_dump()!r}"
    )
    needs = _needs(text, need_analyzer)
    match = [n for n in needs.needs if n.amount == 15000000.0]
    assert match, f"input={text!r} actual={needs.model_dump()!r}"
    assert match[0].need_type, f"input={text!r} actual={needs.model_dump()!r}"


def test_real_monthly_income_annualized(profile_processor) -> None:
    """Monthly income follows the production annualization contract."""
    text = "My monthly income is Rs 20000."
    profile = _profile(text, profile_processor)
    assert profile.annual_family_income == 240000.0, (
        f"input={text!r} actual={profile.model_dump()!r}"
    )
    assert profile.income_period == "annual", f"input={text!r} actual={profile.model_dump()!r}"


def test_real_uncertainty_not_fabricated(profile_processor, need_analyzer) -> None:
    """Hedged wording never becomes unrelated certainty or profile facts."""
    text = "maybe I need around 5 lakh for machines, not sure about the exact amount"
    profile = _profile(text, profile_processor)
    assert profile.age is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.annual_family_income is None, f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    assert needs.needs, f"input={text!r} actual={needs.model_dump()!r}"
    for need in needs.needs:
        assert need.amount is None or need.amount == 500000.0, (
            f"input={text!r} actual={needs.model_dump()!r}"
        )


def test_real_multi_need(profile_processor, need_analyzer) -> None:
    """Two explicit needs survive with their own types and amounts."""
    text = "I need Rs 5 lakh for machinery and Rs 3 lakh as working capital."
    needs = _needs(text, need_analyzer)
    by_type = {n.need_type: n.amount for n in needs.needs}
    assert by_type.get("machinery") == 500000.0, f"input={text!r} actual={needs.model_dump()!r}"
    assert by_type.get("working_capital") == 300000.0, (
        f"input={text!r} actual={needs.model_dump()!r}"
    )


def test_real_inference_trap_no_occupation_invention(profile_processor, need_analyzer) -> None:
    """Another person's occupation is not copied onto the user."""
    text = "My brother is a farmer. I need training for stitching work."
    profile = _profile(text, profile_processor)
    assert profile.occupation is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.age is None, f"input={text!r} actual={profile.model_dump()!r}"
    assert profile.state is None, f"input={text!r} actual={profile.model_dump()!r}"
    needs = _needs(text, need_analyzer)
    training = [n for n in needs.needs if n.need_type == "training"]
    assert training, f"input={text!r} actual={needs.model_dump()!r}"
