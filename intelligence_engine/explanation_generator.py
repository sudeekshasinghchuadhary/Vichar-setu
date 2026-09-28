"""Grounded LLM explanation layer: trace in, checked wording out.

Flow: DecisionTrace -> build_prompt() (trace facts + fixed grounding
instructions, nothing else) -> LLMExplanationProvider -> validate
against the trace -> accepted text OR deterministic safe fallback.

Guardrails (conservative, not a perfect hallucination detector):
- non-empty string output required;
- every number token from the trace must appear verbatim;
- any trace uncertainty requires an uncertainty cue in the output;
- banned guarantee phrases rejected;
- verdict-flip rejected (e.g. "eligible" appearing after "not eligible"
  was stripped means the LLM upgraded the verdict).

No eligibility/matching/financial logic, DB, API, or embeddings here.
"""

import re

from intelligence_engine.llm_client import LLMExplanationProvider
from intelligence_engine.schemas import DecisionTrace, GeneratedExplanation

_GROUNDING_RULES = (
    "Use ONLY the facts below. Preserve every number exactly. "
    "Preserve uncertainty: never turn needs_information/unknown into a "
    "rejection or a confirmation. Do not add facts, guarantees, "
    "procedures, or documents. Do not change the decision."
)

_BANNED_PHRASES = (
    "guaranteed",
    "guarantee",
    "will be approved",
    "definitely approved",
    "definitely eligible",
    "assured approval",
    "100% approved",
)

_NEGATION_CUES = ("not ", "no ", "never ", "n't ", "without ")


def _has_unnegated(output: str, phrase: str) -> bool:
    """True when the phrase appears without a nearby negation cue."""
    for match in re.finditer(re.escape(phrase), output):
        window = output[max(0, match.start() - 12):match.start()]
        if not any(cue in window for cue in _NEGATION_CUES):
            return True
    return False

_UNCERTAINTY_CUES = (
    "unknown",
    "unavailable",
    "missing",
    "needs information",
    "not yet",
    "uncertain",
    "to be confirmed",
    "confirm",
)

_NUMBER_PATTERN = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    """Number tokens with formatting stripped for comparison."""
    return [token.replace(",", "") for token in _NUMBER_PATTERN.findall(text)]


def build_prompt(trace: DecisionTrace) -> str:
    """Render ONLY trace content plus fixed grounding rules into a prompt."""
    lines = [f"Subject: {trace.subject}", "", "Trace:"]
    for item in trace.items:
        lines.append(f"- [{item.kind}] ({item.source}) {item.text}")
    return "\n".join(lines) + "\n\nInstructions: " + _GROUNDING_RULES


def _check_output(output: object, trace: DecisionTrace) -> list[str]:
    """Return failed check names (empty means grounded)."""
    if not isinstance(output, str) or not output.strip():
        return ["non_empty_string"]
    issues: list[str] = []
    lowered = output.lower()
    trace_numbers = []
    for item in trace.items:
        trace_numbers.extend(_numbers(item.text))
    if any(token not in _numbers(output) for token in trace_numbers):
        issues.append("numbers_preserved")
    if any(item.kind == "uncertainty" for item in trace.items) and not any(
        cue in lowered for cue in _UNCERTAINTY_CUES
    ):
        issues.append("uncertainty_preserved")
    if any(_has_unnegated(lowered, phrase) for phrase in _BANNED_PHRASES):
        issues.append("no_guarantees")
    decisions = [item.text.lower() for item in trace.items if item.kind == "decision"]
    negative = any("not eligible" in text or "cannot be determined" in text for text in decisions)
    if negative:
        scrubbed = lowered.replace("not eligible", "").replace("cannot be determined", "")
        if "eligible" in scrubbed:
            issues.append("verdict_not_flipped")
    return issues


def _fallback_text(trace: DecisionTrace) -> str:
    """Deterministic rendering of the trace (safe fallback wording)."""
    lines = [f"Subject: {trace.subject}"]
    for item in trace.items:
        lines.append(f"{item.kind.capitalize()}: {item.text}")
    return "\n".join(lines)


class ExplanationGenerator:
    """Render traces through an LLM provider with grounding validation."""

    def __init__(self, provider: LLMExplanationProvider) -> None:
        """Store the injected text backend (no decision authority)."""
        self.provider = provider

    def generate(self, trace: DecisionTrace) -> GeneratedExplanation:
        """Generate checked wording, falling back to trace text on violation.

        Args:
            trace: Source of truth; never modified.

        Returns:
            GeneratedExplanation with grounded text, or the deterministic
            fallback with issues listed.
        """
        try:
            output: object = self.provider.generate_explanation(build_prompt(trace))
        except Exception as exc:
            return GeneratedExplanation(
                subject=trace.subject,
                text=_fallback_text(trace),
                grounded=False,
                fallback_used=True,
                issues=[f"provider_error: {type(exc).__name__}"],
            )
        issues = _check_output(output, trace)
        if issues:
            return GeneratedExplanation(
                subject=trace.subject,
                text=_fallback_text(trace),
                grounded=False,
                fallback_used=True,
                issues=issues,
            )
        assert isinstance(output, str)
        return GeneratedExplanation(
            subject=trace.subject, text=output.strip(), grounded=True, fallback_used=False, issues=[]
        )
