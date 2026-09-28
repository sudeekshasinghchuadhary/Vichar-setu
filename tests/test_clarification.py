"""Tests for stateless clarification (deterministic, offline)."""

from intelligence_engine.clarification import (
    build_clarification,
    classify_financial_unknown,
)
from intelligence_engine.schemas import (
    EligibilityResult,
    IntelligenceResult,
    NeedAnalysisResult,
    SchemeInsights,
    SchemeSupport,
    SupportNeed,
    UserProfile,
)


def _support(need_type="machinery", amount=300000, period="one_time") -> SchemeSupport:
    """Candidate scheme support for detector tests."""
    return SchemeSupport(
        support_type="loan",
        covers_need_types=[need_type],
        max_amount=amount,
        max_amount_period=period,
    )


def _result(schemes=None, needs=None) -> IntelligenceResult:
    """Minimal product result."""
    return IntelligenceResult(
        profile=UserProfile(age=25),
        needs=needs,
        schemes=schemes or [],
        stages_completed=["profile"],
    )


def _needs_info(scheme_id, *fields) -> SchemeInsights:
    """Scheme insight with a needs_information verdict."""
    return SchemeInsights(
        scheme_id=scheme_id,
        eligibility=EligibilityResult(
            scheme_id=scheme_id,
            status="needs_information",
            reasons=["Age information is required."],
            missing_information=list(fields),
        ),
    )


def test_one_missing_eligibility_field() -> None:
    """A single missing field yields one grounded question."""
    request = build_clarification(_result(schemes=[_needs_info("s", "age")]))
    assert request is not None
    assert request.missing_fields == ["age"]
    assert request.questions[0].field == "age"
    assert request.questions[0].question == "How old are you?"


def test_multiple_missing_fields_deduped() -> None:
    """Repeated fields across schemes ask once, in first-seen order."""
    request = build_clarification(
        _result(schemes=[_needs_info("a", "age", "occupation"), _needs_info("b", "occupation", "state")])
    )
    assert request is not None
    assert request.missing_fields == ["age", "occupation", "state"]
    assert len(request.questions) == 3


def test_unknown_need_amount() -> None:
    """Amount-less needs ask for funding without inventing numbers."""
    needs = NeedAnalysisResult(needs=[SupportNeed(need_type="machinery")])
    request = build_clarification(_result(needs=needs))
    assert request is not None
    assert request.missing_fields == ["needs[0].amount"]
    assert "machinery" in request.questions[0].question
    assert "500000" not in request.questions[0].question


def test_multiple_clarification_reasons() -> None:
    """Eligibility gaps and need gaps combine in one request."""
    needs = NeedAnalysisResult(
        needs=[SupportNeed(need_type="training", amount=50000, amount_period="unknown")]
    )
    request = build_clarification(_result(schemes=[_needs_info("s", "annual_family_income")], needs=needs))
    assert request is not None
    assert request.missing_fields == ["annual_family_income", "needs[0].amount_period"]
    assert len(request.questions) == 2


def test_no_missing_information() -> None:
    """Complete results produce no request (None, not an empty form)."""
    assert build_clarification(_result()) is None
    eligible = SchemeInsights(
        scheme_id="s",
        eligibility=EligibilityResult(scheme_id="s", status="eligible", reasons=["ok"]),
    )
    assert build_clarification(_result(schemes=[eligible])) is None


def test_deterministic_question_generation() -> None:
    """Same gaps always yield identical questions."""
    result = _result(schemes=[_needs_info("s", "state", "age")])
    assert build_clarification(result) == build_clarification(result)


def test_no_invented_questions() -> None:
    """Failed schemes and known data never generate questions."""
    failed = SchemeInsights(
        scheme_id="s",
        eligibility=EligibilityResult(
            scheme_id="s", status="not_eligible", reasons=["Too old."],
            missing_information=["occupation"],
        ),
    )
    needs = NeedAnalysisResult(
        needs=[SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")]
    )
    request = build_clarification(_result(schemes=[failed], needs=needs))
    assert request is None


def test_partial_result_preserved() -> None:
    """The request carries the unchanged input result for re-invocation."""
    result = _result(schemes=[_needs_info("s", "district")])
    request = build_clarification(result)
    assert request is not None
    assert request.current_partial_result == result
    assert request.questions[0].question == "Which district do you live in?"


def test_no_llm_database_or_api_surface() -> None:
    """Clarification module is pure data transformation."""
    import intelligence_engine.clarification as clarification_module

    for forbidden in (
        "LLMClient", "llm_client", "EligibilityEngine", "MatchingEngine",
        "sqlite", "postgres", "requests", "fastapi",
    ):
        assert forbidden not in dir(clarification_module)


def test_detector_need_amount_unknown() -> None:
    """Missing need amount is user-answerable and asks exactly needs[0].amount."""
    need = SupportNeed(need_type="machinery")
    assert classify_financial_unknown(need, [_support()]) == "need_amount_unknown"
    request = build_clarification(_result(needs=NeedAnalysisResult(needs=[need])))
    assert request is not None
    assert request.missing_fields == ["needs[0].amount"]
    assert len(request.questions) == 1
    assert request.questions[0].field == "needs[0].amount"


def test_detector_need_period_unknown() -> None:
    """Missing need period is user-answerable via needs[0].amount_period."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="unknown")
    assert classify_financial_unknown(need, [_support()]) == "need_period_unknown"
    request = build_clarification(_result(needs=NeedAnalysisResult(needs=[need])))
    assert request is not None
    assert request.missing_fields == ["needs[0].amount_period"]


def test_detector_support_amount_unknown_asks_nothing() -> None:
    """Unknown scheme caps are backend gaps: identified, never asked."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")
    unknown_cap = _support(amount=None)
    assert classify_financial_unknown(need, [unknown_cap]) == "support_amount_unknown"
    assert build_clarification(_result(needs=NeedAnalysisResult(needs=[need]))) is None


def test_detector_no_period_matched_support_asks_nothing() -> None:
    """No usable support (none, other type, or unknown periods) asks nothing."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")
    assert classify_financial_unknown(need, []) == "no_period_matched_support"
    assert (
        classify_financial_unknown(need, [_support(need_type="training")])
        == "no_period_matched_support"
    )
    assert (
        classify_financial_unknown(
            need, [_support(period="unknown", amount=None)]
        )
        == "no_period_matched_support"
    )
    assert build_clarification(_result(needs=NeedAnalysisResult(needs=[need]))) is None


def test_detector_period_mismatch_asks_nothing() -> None:
    """Known-but-different periods are backend mismatch: identified, never asked."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")
    monthly = _support(period="monthly", amount=20000)
    assert classify_financial_unknown(need, [monthly]) == "period_mismatch"
    assert build_clarification(_result(needs=NeedAnalysisResult(needs=[need]))) is None


def test_detector_computable_returns_none() -> None:
    """Matching known support means analysis can run: no cause, no question."""
    need = SupportNeed(need_type="machinery", amount=500000, amount_period="one_time")
    assert classify_financial_unknown(need, [_support()]) is None


def test_multiple_needs_deterministic_first_seen() -> None:
    """Missing amounts/periods ask in order; repeats across schemes asked once."""
    needs = NeedAnalysisResult(
        needs=[
            SupportNeed(need_type="machinery"),
            SupportNeed(need_type="training", amount=70000, amount_period="unknown"),
        ]
    )
    result = _result(
        schemes=[_needs_info("a", "age"), _needs_info("b", "age")], needs=needs
    )
    request = build_clarification(result)
    assert request is not None
    assert request.missing_fields == ["age", "needs[0].amount", "needs[1].amount_period"]
