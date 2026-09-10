"""Tests for application & pathway intelligence (deterministic, offline)."""

import pytest

from intelligence_engine.pathway_engine import ApplicationPathwayEngine, PathwayError
from intelligence_engine.schemas import ApplicationStep, ChannelInfo, Scheme


def _scheme(**overrides) -> Scheme:
    """Scheme with documents, steps, link, and channel by default."""
    data = {
        "id": "scheme-001",
        "name": "Scheme",
        "documents_required": ["Aadhaar", "Income certificate"],
        "application_link": "https://example.com/apply",
        "application_steps": [
            ApplicationStep(step_type="prepare_documents", title="Gather documents"),
            ApplicationStep(step_type="submit_application", title="Submit online"),
        ],
        "application_channels": [ChannelInfo(name="District office", channel_type="offline")],
    }
    data.update(overrides)
    return Scheme(**data)


def test_all_documents_available_ready() -> None:
    """Every required document available means ready (not approval)."""
    result = ApplicationPathwayEngine().plan(
        _scheme(), {"Aadhaar": True, "Income certificate": "available"}
    )
    assert result.readiness == "ready"
    assert all(check.status == "available" for check in result.documents)
    assert any("not that approval is guaranteed" in reason for reason in result.reasons)


def test_one_document_missing_not_ready() -> None:
    """A single explicitly missing document blocks readiness."""
    result = ApplicationPathwayEngine().plan(_scheme(), {"Aadhaar": True, "Income certificate": False})
    assert result.readiness == "not_ready"
    assert any("Obtain 'Income certificate'" in action for action in result.next_actions)


def test_unknown_document_status_needs_information() -> None:
    """Unmentioned documents stay unknown, never missing."""
    result = ApplicationPathwayEngine().plan(_scheme(), {"Aadhaar": True})
    assert result.readiness == "needs_information"
    assert result.documents[1].status == "unknown"
    assert any("Confirm whether you have" in action for action in result.next_actions)


def test_no_required_documents_ready() -> None:
    """No listed requirements means ready with an explicit reason."""
    result = ApplicationPathwayEngine().plan(_scheme(documents_required=[]), {})
    assert result.readiness == "ready"
    assert any("No documents are listed as required" in reason for reason in result.reasons)


def test_duplicate_documents_deduped() -> None:
    """Repeated required entries collapse case-insensitively."""
    result = ApplicationPathwayEngine().plan(
        _scheme(documents_required=["Aadhaar", "aadhaar ", "AADHAAR"]), {"aadhaar": True}
    )
    assert len(result.documents) == 1
    assert result.readiness == "ready"


def test_invalid_document_data_rejected() -> None:
    """Non-dict maps and unrecognized values raise cleanly."""
    with pytest.raises(PathwayError):
        ApplicationPathwayEngine().plan(_scheme(), ["Aadhaar"])  # type: ignore[arg-type]
    with pytest.raises(PathwayError):
        ApplicationPathwayEngine().plan(_scheme(), {"Aadhaar": 42})


def test_deterministic_readiness() -> None:
    """Same inputs always give identical readiness and checks."""
    engine = ApplicationPathwayEngine()
    status = {"Aadhaar": True, "Income certificate": False}
    assert engine.plan(_scheme(), status) == engine.plan(_scheme(), status)


def test_pathway_step_ordering_preserved() -> None:
    """Steps come out in the verified listed order."""
    result = ApplicationPathwayEngine().plan(_scheme(), {})
    assert [step.title for step in result.steps] == ["Gather documents", "Submit online"]


def test_missing_application_link_no_submit_action() -> None:
    """Without a link there is no submission action, only a stated gap."""
    result = ApplicationPathwayEngine().plan(_scheme(application_link=None), {})
    assert not any("Submit the application" in action for action in result.next_actions)
    assert any("No application link" in reason for reason in result.reasons)


def test_known_application_link_submit_action() -> None:
    """An explicit link produces a submission action with the URL."""
    result = ApplicationPathwayEngine().plan(_scheme(), {})
    assert "Submit the application at https://example.com/apply." in result.next_actions


def test_known_channel_grounded() -> None:
    """Structured channels pass through as info, never recommendations."""
    result = ApplicationPathwayEngine().plan(_scheme(), {})
    assert result.channels[0].name == "District office"
    assert any("not a recommendation" in reason for reason in result.reasons)
    assert not any("recommend" in action.lower() for action in result.next_actions)


def test_unknown_channel_stated() -> None:
    """No channels means no channel info and no invented partners."""
    result = ApplicationPathwayEngine().plan(_scheme(application_channels=[]), {})
    assert result.channels == []


def test_no_invented_procedures() -> None:
    """Schemes without steps get no steps and an explicit gap statement."""
    result = ApplicationPathwayEngine().plan(_scheme(application_steps=[]), {})
    assert result.steps == []
    assert "Pathway information is unavailable for this scheme." in result.next_actions


def test_inputs_unchanged() -> None:
    """Scheme and status map are unchanged by planning."""
    scheme = _scheme()
    status = {"Aadhaar": True}
    before = (scheme.model_dump(), dict(status))
    ApplicationPathwayEngine().plan(scheme, status)
    assert scheme.model_dump() == before[0]
    assert status == before[1]


def test_deterministic_repeated_execution() -> None:
    """Identical runs produce identical pathway results."""
    engine = ApplicationPathwayEngine()
    assert engine.plan(_scheme(), {}) == engine.plan(_scheme(), {})


def test_no_llm_calls() -> None:
    """Pathway module never touches LLM machinery."""
    import intelligence_engine.pathway_engine as pathway_module

    for forbidden in ("LLMClient", "llm_client", "NeedExtractor", "embed", "cosine", "ocr", "OCR"):
        assert forbidden not in dir(pathway_module)


def test_no_database_or_api_calls() -> None:
    """Pathway module has no database or API surface."""
    import intelligence_engine.pathway_engine as pathway_module

    for forbidden in ("sqlite", "postgres", "psycopg", "sqlalchemy", "requests", "fastapi"):
        assert forbidden not in dir(pathway_module)
