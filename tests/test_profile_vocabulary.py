"""Focused tests for shared state/category normalization (offline)."""

from intelligence_engine.profile_processor import ProfileProcessor, normalize_profile_data
from intelligence_engine.profile_vocabulary import (
    normalize_social_category,
    normalize_state,
)
from intelligence_engine.llm_client import LLMClient
from intelligence_engine.schemas import UserProfile


class _FakeLLM(LLMClient):
    """Canned extraction payload."""

    def __init__(self, payload):
        """Store the payload."""
        self.payload = payload

    def extract_profile_data(self, user_text):
        """Return the canned payload."""
        return self.payload


def test_existing_english_aliases() -> None:
    """English/UP/category aliases keep their canonical outputs."""
    assert normalize_state("UP") == "Uttar Pradesh"
    assert normalize_state("u.p.") == "Uttar Pradesh"
    assert normalize_state("uttar pradesh") == "Uttar Pradesh"
    assert normalize_social_category("Scheduled Caste") == "SC"
    assert normalize_social_category("st") == "ST"
    assert normalize_social_category("OBC") == "OBC"


def test_devanagari_aliases() -> None:
    """Unambiguous Devanagari forms map to canonical values."""
    assert normalize_state("उत्तर प्रदेश") == "Uttar Pradesh"
    assert normalize_social_category("अनुसूचित जाति") == "SC"
    assert normalize_social_category("अनुसूचित जनजाति") == "ST"
    assert normalize_social_category("अन्य पिछड़ा वर्ग") == "OBC"
    assert normalize_social_category("सामान्य") == "General"


def test_justified_hinglish_aliases() -> None:
    """Common Latin-script variants of the same referents map correctly."""
    assert normalize_state("yupi") == "Uttar Pradesh"
    assert normalize_social_category("schedule caste") == "SC"
    assert normalize_social_category("schedule tribe") == "ST"


def test_direct_construction_normalizes() -> None:
    """Backend-style direct construction applies the shared vocabulary."""
    profile = UserProfile(state="UP", social_category="अनुसूचित जाति")
    assert profile.state == "Uttar Pradesh"
    assert profile.social_category == "SC"
    assert UserProfile().state is None
    assert UserProfile().social_category is None


def test_processor_construction_matches_direct() -> None:
    """Both paths produce identical normalized profiles."""
    payload = {"state": "yupi", "social_category": "schedule caste"}
    via_processor = ProfileProcessor(_FakeLLM(payload)).process_profile("hi")
    via_direct = UserProfile(**payload)
    assert via_processor == via_direct
    assert via_direct.state == "Uttar Pradesh"
    assert via_direct.social_category == "SC"


def test_normalization_idempotent() -> None:
    """Re-normalizing canonical outputs changes nothing."""
    for value in ["Uttar Pradesh", "SC", "ST", "OBC", "General", "EWS", "उत्तर प्रदेश"]:
        first = normalize_state(value)
        assert normalize_state(first) == first
        first_cat = normalize_social_category(value)
        assert normalize_social_category(first_cat) == first_cat
    assert normalize_profile_data(normalize_profile_data({"state": "UP"})) == {"state": "Uttar Pradesh"}
