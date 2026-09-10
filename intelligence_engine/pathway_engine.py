"""Application & pathway intelligence: documents, readiness, steps, channels.

Flow: Scheme (documents_required, application_steps, application_link,
application_channels) + user document availability -> PathwayResult.

Rules (never violated):
- Only explicitly structured data is used. Free-text descriptions never
  imply documents, steps, or channels. Nothing is invented.
- Unknown possession stays "unknown", never "missing".
- Readiness is conservative: all available -> ready; any missing ->
  not_ready; otherwise any unknown -> needs_information; no required
  documents -> ready. Ready means documentation is complete, never that
  approval is guaranteed.
- Steps execute in the order provided by verified scheme data.
- Channels are information only, never recommendations or endorsements.
- Inputs are never mutated. No eligibility logic, LLM, DB, API, or OCR.
"""

from typing import Any

from intelligence_engine.schemas import (
    ApplicationStep,
    ChannelInfo,
    DocumentCheck,
    PathwayResult,
    Scheme,
)


class PathwayError(ValueError):
    """Raised for malformed availability input (never silently fixed)."""


_AVAILABLE = {"available", "yes", "true", "have", "got"}
_MISSING = {"missing", "no", "false", "not have", "dont have"}


def _normalize_availability(value: Any, document: str) -> str:
    """Map True/False/explicit words to available/missing/unknown."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "available" if value else "missing"
    if isinstance(value, str):
        cleaned = " ".join(value.split()).lower()
        if cleaned in ("unknown", "not sure", ""):
            return "unknown"
        if cleaned in _AVAILABLE:
            return "available"
        if cleaned in _MISSING:
            return "missing"
    raise PathwayError(f"Unrecognized availability for '{document}': {value!r}.")


class ApplicationPathwayEngine:
    """Deterministic pathway builder from verified scheme data."""

    def plan(
        self, scheme: Scheme, document_status: dict[str, Any] | None = None,
    ) -> PathwayResult:
        """Build the pathway without mutating inputs.

        Args:
            scheme: Scheme with structured documents/steps/link/channels.
            document_status: Map of document name (case-insensitive) to
                availability (True/False/"available"/"missing"/None).
                Unmentioned documents stay "unknown".

        Returns:
            PathwayResult with readiness, checks, steps, channels, and
            next actions.
        """
        if document_status is not None and not isinstance(document_status, dict):
            raise PathwayError("document_status must be a dict or None.")
        lookup = {
            " ".join(str(key).split()).lower(): value
            for key, value in (document_status or {}).items()
        }

        documents: list[DocumentCheck] = []
        seen: set[str] = set()
        for raw in scheme.documents_required:
            name = " ".join(str(raw).split())
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            documents.append(
                DocumentCheck(document=name, status=_normalize_availability(lookup.get(name.lower()), name))
            )

        states = [check.status for check in documents]
        if any(state == "missing" for state in states):
            readiness = "not_ready"
        elif any(state == "unknown" for state in states):
            readiness = "needs_information"
        else:
            readiness = "ready"

        steps = [
            ApplicationStep(step_type=step.step_type, title=step.title, detail=step.detail)
            for step in scheme.application_steps
        ]
        channels = [
            ChannelInfo(
                name=channel.name,
                channel_type=channel.channel_type,
                description=channel.description,
                link=channel.link,
            )
            for channel in scheme.application_channels
        ]

        reasons: list[str] = []
        next_actions: list[str] = []
        if not documents:
            reasons.append("No documents are listed as required by this scheme.")
        for check in documents:
            if check.status == "missing":
                reasons.append(f"Required document explicitly missing: '{check.document}'.")
                next_actions.append(f"Obtain '{check.document}' (required).")
            elif check.status == "unknown":
                reasons.append(f"Possession unknown for required document: '{check.document}'.")
                next_actions.append(f"Confirm whether you have '{check.document}' (status unknown).")
        if readiness == "ready":
            reasons.append("All required documents are explicitly available.")
            reasons.append("Ready means documentation is complete, not that approval is guaranteed.")
        if steps:
            reasons.append(f"Pathway has {len(steps)} verified step(s) in listed order.")
        else:
            reasons.append("No verified pathway steps are available for this scheme.")
            next_actions.append("Pathway information is unavailable for this scheme.")
        link = (scheme.application_link or "").strip()
        if link:
            reasons.append("An explicit application link is available.")
            next_actions.append(f"Submit the application at {link}.")
        else:
            reasons.append("No application link is available in the scheme data.")
        for channel in channels:
            reasons.append(f"Channel listed by scheme data (not a recommendation): '{channel.name}'.")

        return PathwayResult(
            scheme_id=scheme.id,
            readiness=readiness,
            documents=documents,
            steps=steps,
            channels=channels,
            next_actions=next_actions,
            reasons=reasons,
        )
