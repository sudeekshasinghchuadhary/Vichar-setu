"""Tests for need analysis (fake extractor, offline, deterministic)."""

from typing import Any

import pytest

from intelligence_engine.llm_client import NeedExtractor
from intelligence_engine.need_analyzer import (
    NeedAnalyzer,
    NeedAnalysisError,
    normalize_need_type,
)
from intelligence_engine.schemas import NeedAnalysisResult


class FakeNeedExtractor(NeedExtractor):
    """Fake backend returning canned goal/needs structures."""

    def __init__(self, payload: Any) -> None:
        """Store the canned payload."""
        self.payload = payload

    def extract_need_data(self, user_text: str) -> dict[str, Any]:
        """Return the canned payload regardless of input."""
        return self.payload


def test_one_need_with_amount() -> None:
    """A single stated need becomes one validated need with a total."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "business_goal": "Expand tailoring business",
                "needs": [{"need_type": "machinery", "amount": 500000, "amount_period": "one_time"}],
            }
        )
    )
    result = analyzer.analyze("expand tailoring, need 5 lakh for machines")
    assert isinstance(result, NeedAnalysisResult)
    assert result.business_goal == "Expand tailoring business"
    assert len(result.needs) == 1
    assert result.needs[0].amount == 500000
    assert result.total_requested == 500000


def test_multiple_needs_and_total() -> None:
    """Machinery 500000 + working capital 300000 totals 800000 deterministically."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "business_goal": "Expand tailoring business",
                "needs": [
                    {"need_type": "machines", "amount": 500000, "amount_period": "one_time"},
                    {"need_type": "working capital", "amount": 300000, "amount_period": "one_time"},
                ],
            }
        )
    )
    result = analyzer.analyze("need 8 lakh for machines and working capital")
    assert [need.need_type for need in result.needs] == ["machinery", "working_capital"]
    assert result.total_requested == 800000


def test_missing_amount_stays_unknown() -> None:
    """A need without an amount keeps amount None and no total."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor({"needs": [{"need_type": "training", "context": "stitching course"}]})
    )
    result = analyzer.analyze("need training")
    assert result.needs[0].amount is None
    assert result.needs[0].amount_period == "unknown"
    assert result.total_requested is None


def test_ambiguous_period_never_guessed() -> None:
    """Unstated periods stay unknown; monthly and annual never mix into the total."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "needs": [
                    {"need_type": "working_capital", "amount": 20000},
                    {"need_type": "working_capital", "amount": 20000, "amount_period": "monthly"},
                    {"need_type": "machinery", "amount": 100000, "amount_period": "annual"},
                ]
            }
        )
    )
    result = analyzer.analyze("need money monthly and yearly")
    periods = {(need.need_type, need.amount_period) for need in result.needs}
    assert ("working_capital", "unknown") in periods
    assert ("working_capital", "monthly") in periods
    assert result.total_requested is None


def test_duplicate_needs_merged() -> None:
    """Same type+period merges amounts and contexts deterministically."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "needs": [
                    {"need_type": "machines", "amount": 200000, "amount_period": "one_time", "context": "stitching"},
                    {"need_type": "machinery", "amount": 300000, "amount_period": "one_time", "context": "cutting"},
                ]
            }
        )
    )
    result = analyzer.analyze("machines twice")
    assert len(result.needs) == 1
    assert result.needs[0].amount == 500000
    assert result.needs[0].context == "stitching; cutting"


def test_hindi_example_structured() -> None:
    """Hindi input structures through the same contract (fake backend)."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "business_goal": "सिलाई व्यवसाय बढ़ाना",
                "needs": [{"need_type": "machinery", "amount": 200000, "amount_period": "one_time"}],
            }
        )
    )
    result = analyzer.analyze("मुझे सिलाई व्यवसाय के लिए मशीनों हेतु 2 लाख चाहिए")
    assert result.business_goal == "सिलाई व्यवसाय बढ़ाना"
    assert result.total_requested == 200000


def test_hinglish_example_structured() -> None:
    """Hinglish input structures through the same contract (fake backend)."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "business_goal": "tailoring business expand karna",
                "needs": [{"need_type": "working_capital", "amount_period": "monthly", "amount": 15000}],
            }
        )
    )
    result = analyzer.analyze("mujhe tailoring ke liye monthly 15000 working capital chahiye")
    assert result.needs[0].need_type == "working_capital"
    assert result.needs[0].amount_period == "monthly"


def test_malformed_output_rejected() -> None:
    """Non-dict output, non-list needs, and non-dict entries raise cleanly."""
    with pytest.raises(NeedAnalysisError):
        NeedAnalyzer(FakeNeedExtractor("nope")).analyze("hi")  # type: ignore[arg-type]
    with pytest.raises(NeedAnalysisError):
        NeedAnalyzer(FakeNeedExtractor({"needs": "machinery"})).analyze("hi")
    with pytest.raises(NeedAnalysisError):
        NeedAnalyzer(FakeNeedExtractor({"needs": ["machinery"]})).analyze("hi")


def test_validation_failures_rejected() -> None:
    """Negative amounts and empty types fail validation without fabrication."""
    with pytest.raises(NeedAnalysisError):
        NeedAnalyzer(FakeNeedExtractor({"needs": [{"need_type": "machinery", "amount": -5}]})).analyze("hi")
    with pytest.raises(NeedAnalysisError):
        NeedAnalyzer(FakeNeedExtractor({"needs": [{"need_type": "   "}]})).analyze("hi")


def test_normalization_keeps_distinct_concepts_apart() -> None:
    """machines->machinery, but working capital/training/marketing stay distinct."""
    assert normalize_need_type("machines") == ("machinery", True)
    assert normalize_need_type("equipment") == ("machinery", True)
    assert normalize_need_type("working capital") == ("working_capital", True)
    assert normalize_need_type("training") == ("training", True)
    assert normalize_need_type("marketing") == ("marketing", True)
    analyzer = NeedAnalyzer(
        FakeNeedExtractor({"needs": [{"need_type": "raw material", "amount": 50000}]})
    )
    result = analyzer.analyze("need raw material")
    assert result.needs[0].need_type == "raw material"
    assert any("kept as stated" in note for note in result.notes)


def test_no_hallucinated_information() -> None:
    """Output contains only what the extractor provided (one need stays one)."""
    analyzer = NeedAnalyzer(FakeNeedExtractor({"needs": [{"need_type": "training"}]}))
    result = analyzer.analyze("need training")
    assert len(result.needs) == 1
    assert result.business_goal is None
    assert result.total_requested is None
    assert result.needs[0].priority is None


def test_no_eligibility_or_scheme_decisions() -> None:
    """Analyzer exposes no eligibility/scheme/match behavior."""
    analyzer = NeedAnalyzer(FakeNeedExtractor({"needs": []}))
    for attr in ("eligible", "eligibility", "scheme", "recommend", "match", "score"):
        assert not hasattr(analyzer, attr)
    import intelligence_engine.need_analyzer as need_module

    for forbidden in ("EligibilityEngine", "MatchingEngine", "Scheme", "LLMClient"):
        assert forbidden not in dir(need_module)


def test_no_database_or_api_calls() -> None:
    """Need module has no database or API surface."""
    import intelligence_engine.need_analyzer as need_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(need_module)


def test_repeated_execution_is_identical() -> None:
    """Same inputs always produce identical need results."""
    payload = {
        "business_goal": "Expand tailoring business",
        "needs": [{"need_type": "machines", "amount": 500000, "amount_period": "one_time"}],
    }
    analyzer = NeedAnalyzer(FakeNeedExtractor(payload))
    assert analyzer.analyze("hi").model_dump() == analyzer.analyze("hi").model_dump()


def test_rich_description_preserved_with_canonical_type() -> None:
    """Context keeps full meaning while need_type stays canonical; no purpose field."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "business_goal": "Expand tailoring business",
                "needs": [
                    {
                        "need_type": "machines",
                        "amount": 500000,
                        "amount_period": "one_time",
                        "context": "buy computerized sewing machines to expand my tailoring business",
                    }
                ],
            }
        )
    )
    result = analyzer.analyze("I need money to buy sewing machines for expanding my tailoring business.")
    need = result.needs[0]
    assert need.need_type == "machinery"
    assert need.amount == 500000
    assert need.context == "buy computerized sewing machines to expand my tailoring business"
    assert not hasattr(need, "purpose")
    assert "purpose" not in type(need).model_fields


def test_rich_description_without_canonical_category() -> None:
    """Unmappable kinds keep wording and context; nothing is invented."""
    analyzer = NeedAnalyzer(
        FakeNeedExtractor(
            {
                "needs": [
                    {
                        "need_type": "sewing machine",
                        "amount": 500000,
                        "amount_period": "one_time",
                        "context": "buy computerized sewing machines to expand my tailoring business",
                    }
                ],
            }
        )
    )
    result = analyzer.analyze("I need money to buy sewing machines for expanding my tailoring business.")
    need = result.needs[0]
    assert need.need_type == "sewing machine"
    assert need.amount == 500000
    assert need.context == "buy computerized sewing machines to expand my tailoring business"
