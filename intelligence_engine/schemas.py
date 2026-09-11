"""Pydantic schemas for the Core Intelligence Engine.

NOTE: These models are an internal prototype contract for the
intelligence engine. They are designed to work conceptually with
the current PostgreSQL structure (users, schemes,
eligibility_criteria tables) but do NOT copy it exactly.
FastAPI will map database fields into these models later.
Do not build backend or DB contracts on top of these yet.

This module holds data contracts only. No AI logic, no
eligibility logic, no matching logic, no database code,
and no LLM SDK code belong here.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from intelligence_engine.need_vocabulary import normalize_need_type
from intelligence_engine.profile_vocabulary import (
    normalize_social_category,
    normalize_state,
)


class UserProfile(BaseModel):
    """Validated user profile. Every field is optional (None = not provided)."""

    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = Field(default=None)
    social_category: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)
    district: Optional[str] = Field(default=None)
    occupation: Optional[str] = Field(default=None)
    annual_family_income: Optional[float] = Field(default=None, ge=0)
    income_period: Literal["monthly", "annual", "unknown"] = Field(
        default="annual",
        description=(
            "Period of annual_family_income. Default 'annual' covers legacy "
            "structured data already annualized (fixtures/backend). Extractors "
            "must write 'monthly' or 'unknown' explicitly; unknown speech must "
            "never be stored as annual. Monthly values are annualized ×12 in "
            "deterministic normalization; unknown periods cannot satisfy "
            "annual-income rules."
        ),
    )
    purpose: Optional[str] = Field(default=None)
    project_type: Optional[str] = Field(default=None)
    estimated_project_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Profile-level project estimate supplied by the user. Distinct from "
            "NeedAnalysisResult.total_requested (deterministic total of analyzed "
            "one-time needs). Never equated or copied automatically."
        ),
    )
    education_level: Optional[str] = Field(default=None)

    @field_validator("state")
    @classmethod
    def _normalize_state_field(cls, value: Optional[str]) -> Optional[str]:
        """Shared state normalization; None stays None."""
        if value is None:
            return None
        return normalize_state(value)

    @field_validator("social_category")
    @classmethod
    def _normalize_category_field(cls, value: Optional[str]) -> Optional[str]:
        """Shared social-category normalization; None stays None."""
        if value is None:
            return None
        return normalize_social_category(value)


SupportType = Literal["loan", "grant", "subsidy", "training", "marketing", "infrastructure", "other"]
NeedPriority = Literal["high", "medium", "low"]
AmountPeriod = Literal["one_time", "monthly", "annual", "unknown"]


class SchemeSupport(BaseModel):
    """One verified structured support a scheme offers (never inferred).

    support_type: Financial/support kind; loan ≠ grant ≠ subsidy.
    covers_need_types: Canonical need types this covers (entries
        normalized through the shared vocabulary; unknown wording kept
        as cleaned text, never invented).
    max_amount: Verified cap; None means unknown (never treated as 0).
    max_amount_period: Explicit period of max_amount ("unknown" for
        legacy entries that only carry max_amount — never silently
        assumed one-time). Coverage requires equal known periods.
    description: Human detail; free text implies no amount on its own.
    """

    support_type: SupportType = Field()
    covers_need_types: list[str] = Field(default_factory=list)
    max_amount: Optional[float] = Field(default=None, ge=0)
    max_amount_period: AmountPeriod = Field(default="unknown")
    description: Optional[str] = Field(default=None)

    @field_validator("covers_need_types")
    @classmethod
    def _canonicalize_covers(cls, value: list[str]) -> list[str]:
        """Collapse known aliases to canonical types; preserve the rest."""
        normalized: list[str] = []
        for entry in value:
            canonical, _ = normalize_need_type(entry)
            normalized.append(canonical if canonical is not None else entry)
        return normalized


StepType = Literal["prepare_documents", "submit_application", "verification", "follow_up", "other"]
DocumentState = Literal["available", "missing", "unknown"]
ReadinessStatus = Literal["ready", "not_ready", "needs_information"]


class ApplicationStep(BaseModel):
    """One explicitly structured pathway step (never invented).

    Steps execute in list order as provided by verified scheme data.
    """

    step_type: StepType = Field()
    title: str = Field(min_length=1)
    detail: Optional[str] = Field(default=None)


class ChannelInfo(BaseModel):
    """One explicitly structured application channel (info only, never a recommendation)."""

    name: str = Field(min_length=1)
    channel_type: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    link: Optional[str] = Field(default=None)


class DocumentCheck(BaseModel):
    """One required document with its conservative status.

    unknown means the system does not know possession — never treated
    as missing.
    """

    document: str = Field(min_length=1)
    status: DocumentState = Field()


class PathwayResult(BaseModel):
    """Actionable pathway: readiness, documents, steps, channels, next actions.

    ready means documentation is complete per known data — never a
    promise of approval.
    """

    scheme_id: str = Field()
    readiness: ReadinessStatus = Field()
    documents: list[DocumentCheck] = Field(default_factory=list)
    steps: list[ApplicationStep] = Field(default_factory=list)
    channels: list[ChannelInfo] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class Scheme(BaseModel):
    """Internal scheme view mapped from PostgreSQL schemes + eligibility_criteria.

    AUTHORITY RULE: eligibility_rules (states, categories, ...) are the SOLE
    authority for eligibility decisions. Display metadata state/category
    are frontend passthroughs and MUST NOT override or supplement rules;
    conflicting metadata is ignored by EligibilityEngine. Fixed
    display/metadata fields (ministry, benefit, documents, link)
    are optional passthroughs for the frontend. Variable eligibility
    information (age, income, occupation, education, category, state)
    lives in the extensible eligibility_rules dict, mirroring the
    separate eligibility_criteria table. Structured support information
    lives in support_options (free-text benefit never implies an amount).
    No eligibility logic here.
    """

    id: str = Field()
    name: str = Field()
    description: str = Field(default="")
    supported_purposes: list[str] = Field(default_factory=list)
    supported_project_types: list[str] = Field(default_factory=list)
    eligibility_rules: dict[str, Any] = Field(default_factory=dict)
    support_options: list[SchemeSupport] = Field(default_factory=list)
    source: Optional[str] = Field(default=None)
    ministry: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    benefit: Optional[str] = Field(default=None)
    documents_required: list[str] = Field(default_factory=list)
    application_link: Optional[str] = Field(default=None)
    application_steps: list[ApplicationStep] = Field(default_factory=list)
    application_channels: list[ChannelInfo] = Field(default_factory=list)


EligibilityStatus = Literal["eligible", "not_eligible", "needs_information"]


class EligibilityResult(BaseModel):
    """Deterministic eligibility decision with human-readable reasons.

    Use the explicit status field to distinguish a known failure from
    insufficient information. The `missing_information` list is only populated
    when the status is `needs_information`.
    """

    scheme_id: str = Field()
    status: EligibilityStatus = Field()
    reasons: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Ranked match with a validated 0-100 score.

    deterministic_score/semantic_score preserve the hybrid components
    without affecting ranking: deterministic-only results carry the
    deterministic value with semantic None; hybrid results carry both.
    """

    scheme_id: str = Field()
    score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    reasons: list[str] = Field(default_factory=list)
    deterministic_score: Optional[float] = Field(default=None, ge=0, le=100)
    semantic_score: Optional[float] = Field(default=None, ge=0, le=100)


class PipelineResult(BaseModel):
    """Final structured output combining profile, eligibility, and ranking."""

    profile: UserProfile = Field()
    eligibility_results: list[EligibilityResult] = Field(default_factory=list)
    ranked_matches: list[MatchResult] = Field(default_factory=list)


class FailedCriterion(BaseModel):
    """One failed mandatory criterion, structured for analysis and What-If reuse.

    criterion: UserProfile field name (e.g. "annual_family_income").
    user_value: The user's actual value.
    required: Human-readable requirement (e.g. "<= 300000").
    difference: Miss distance where meaningful (e.g. 20000 over the
        income limit); None for categorical mismatches, which have no
        meaningful numeric distance.
    """

    criterion: str = Field()
    user_value: Optional[Any] = Field(default=None)
    required: str = Field(default="")
    difference: Optional[float] = Field(default=None)


class NearMissResult(BaseModel):
    """Analysis of how close a not_eligible outcome is (never overrides it)."""

    scheme_id: str = Field()
    is_near_miss: bool = Field()
    failed_criteria: list[FailedCriterion] = Field(default_factory=list)
    satisfied_criteria: list[str] = Field(default_factory=list)
    total_criteria: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class SupportNeed(BaseModel):
    """One structured support need. Unknown amounts/periods stay unknown.

    need_type: Canonical need category when recognized, otherwise the
        user's own cleaned wording (never invented).
    amount: Explicitly stated amount only; None when not provided.
    amount_period: "one_time", "monthly", "annual", or "unknown".
        Never guessed: unstated periods are "unknown".
    context: Purpose detail for this need (e.g. "machines for stitching").
    priority: Only when explicitly stated ("high"/"medium"/"low").
    """

    need_type: str = Field(min_length=1)
    amount: Optional[float] = Field(default=None, ge=0)
    amount_period: AmountPeriod = Field(default="unknown")
    context: Optional[str] = Field(default=None)
    priority: Optional[NeedPriority] = Field(default=None)

    @field_validator("need_type")
    @classmethod
    def _canonicalize_need_type(cls, value: str) -> str:
        """Collapse known aliases to the canonical vocabulary type."""
        canonical, _ = normalize_need_type(value)
        if canonical is None:
            raise ValueError("need_type must be non-empty text.")
        return canonical


class NeedAnalysisResult(BaseModel):
    """Structured understanding of what the user wants to achieve and need.

    business_goal: The user's goal in their own terms (None if unstated).
    needs: Validated support needs (duplicates merged deterministically).
    total_requested: Sum of one-time amounts only; None when no one-time
        amount is known. Periods are never mixed.
    notes: Transparency flags (e.g. unrecognized types kept as stated,
        ambiguous periods left unknown).
    """

    business_goal: Optional[str] = Field(default=None)
    needs: list[SupportNeed] = Field(default_factory=list)
    total_requested: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Deterministic total of explicitly stated one-time needs. Authoritative "
            "for analyzed needs; distinct from UserProfile.estimated_project_cost "
            "(user-supplied project estimate). Never copied between the two."
        ),
    )
    notes: list[str] = Field(default_factory=list)


class SupportMapping(BaseModel):
    """One need mapped to one verified scheme support option.

    eligibility_status: Copied from EligibilityResult ("unknown" when no
        result was supplied); coverage is computed only for "eligible".
    potential_coverage: min(need amount, max_amount) when both are known,
        periods are equal and known, and status is eligible; else None.
    uncovered_amount: need amount minus coverage when computable; else None.
    coverage_known: False preserves uncertainty instead of guessing.
    """

    scheme_id: str = Field()
    support_type: SupportType = Field()
    eligibility_status: str = Field()
    max_amount: Optional[float] = Field(default=None, ge=0)
    max_amount_period: AmountPeriod = Field(default="unknown")
    potential_coverage: Optional[float] = Field(default=None, ge=0)
    uncovered_amount: Optional[float] = Field(default=None, ge=0)
    coverage_known: bool = Field(default=False)
    reasons: list[str] = Field(default_factory=list)


class NeedPlan(BaseModel):
    """Plan for one need across candidate supports (never summed across schemes).

    best_known_coverage: Highest single-scheme coverage; schemes are not
        added together without convergence data.
    fully_covered: True only when one scheme alone covers the full need.
    """

    need: SupportNeed = Field()
    mappings: list[SupportMapping] = Field(default_factory=list)
    best_known_coverage: Optional[float] = Field(default=None, ge=0)
    uncovered_amount: Optional[float] = Field(default=None, ge=0)
    fully_covered: bool = Field(default=False)
    reasons: list[str] = Field(default_factory=list)


class SupportPlan(BaseModel):
    """Structured support plan for frontend/API/finance/pathway use later.

    convergence_unknown: Always True until structured convergence data
        exists; schemes are assumed non-combinable.
    next_actions: Deterministic follow-ups (missing info, uncovered needs).
    """

    business_goal: Optional[str] = Field(default=None)
    need_plans: list[NeedPlan] = Field(default_factory=list)
    total_need_amount: Optional[float] = Field(default=None, ge=0)
    total_best_known_coverage: Optional[float] = Field(default=None, ge=0)
    total_uncovered_amount: Optional[float] = Field(default=None, ge=0)
    totals_fully_known: bool = Field(default=False)
    convergence_unknown: bool = Field(default=True)
    reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class WhatIfChange(BaseModel):
    """One requested hypothetical edit: field plus its hypothetical value.

    hypothetical_value None means "what if this were unknown".
    """

    field: str = Field(min_length=1)
    hypothetical_value: Optional[Any] = Field(default=None)


class AppliedChange(BaseModel):
    """A change with the current value attached for transparent comparison."""

    field: str = Field()
    current_value: Optional[Any] = Field(default=None)
    hypothetical_value: Optional[Any] = Field(default=None)


class WhatIfResult(BaseModel):
    """Structured current-vs-hypothetical eligibility comparison.

    changed_criteria: Eligibility dimensions whose outcome status differs.
    remaining_failures: Dimensions still failed in the hypothetical run.
    An "eligible" hypothetical means the temporary profile satisfies the
    currently defined rules — never a promise of real-world approval.
    """

    scheme_id: str = Field()
    changes: list[AppliedChange] = Field(default_factory=list)
    current: EligibilityResult = Field()
    hypothetical: EligibilityResult = Field()
    hypothetical_profile: UserProfile = Field()
    status_changed: bool = Field()
    changed_criteria: list[str] = Field(default_factory=list)
    remaining_failures: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FinancialOption(BaseModel):
    """One explicitly known funding option (never inferred).

    amount None means unknown (never treated as zero).
    """

    label: str = Field(min_length=1)
    kind: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None, ge=0)


class LoanTerms(BaseModel):
    """Explicit loan inputs. All three are required; nothing is assumed."""

    principal: float = Field(gt=0)
    annual_rate_percent: float = Field(ge=0)
    tenure_months: int = Field(ge=1)


class LoanResult(BaseModel):
    """Standard amortizing-loan outputs, rounded to 2 decimals.

    No affordability claim: affordability needs income/expense data plus
    an explicit rule, neither of which lives here.
    """

    monthly_emi: float = Field(ge=0)
    total_payable: float = Field(ge=0)
    total_interest: float = Field(ge=0)


class CoverageResult(BaseModel):
    """Deterministic requirement-vs-support arithmetic.

    covered: Best single known option capped at required (never summed
        across options without convergence data).
    uncovered/own_contribution: required minus covered when both known;
        equal by definition here (the gap is the user's share absent
        other funding). None whenever anything needed is unknown.
    coverage_known: False preserves uncertainty instead of guessing.
    """

    required: Optional[float] = Field(default=None, ge=0)
    covered: Optional[float] = Field(default=None, ge=0)
    uncovered: Optional[float] = Field(default=None, ge=0)
    own_contribution: Optional[float] = Field(default=None, ge=0)
    coverage_known: bool = Field(default=False)
    fully_covered: bool = Field(default=False)
    reasons: list[str] = Field(default_factory=list)


class OptionComparison(BaseModel):
    """Comparison on explicitly known amounts only.

    ranked_labels: Labels with known amounts, highest first (ties by
        label). Unknown-amount options are listed separately, unranked —
        never ranked on assumptions.
    """

    ranked_labels: list[str] = Field(default_factory=list)
    unknown_labels: list[str] = Field(default_factory=list)
    best_amount: Optional[float] = Field(default=None, ge=0)
    reasons: list[str] = Field(default_factory=list)


TraceItemKind = Literal["decision", "evidence", "calculation", "uncertainty", "action"]


class ExplanationItem(BaseModel):
    """One trace line with a fixed role and its source engine.

    source names the authoritative engine (e.g. "EligibilityEngine").
    Texts preserve upstream reasons/values; never paraphrased claims.
    """

    kind: TraceItemKind = Field()
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)


class DecisionTrace(BaseModel):
    """Structured explanation of one already-made decision.

    Contains exactly one "decision" item restating the verdict; all
    other items are supporting evidence, calculations, uncertainties,
    or actions. Uncertainty is never converted into a negative verdict.
    """

    subject: str = Field(min_length=1)
    items: list[ExplanationItem] = Field(default_factory=list)


class GeneratedExplanation(BaseModel):
    """LLM-rendered trace wording with its grounding verdict.

    grounded True means the draft passed all checks; otherwise text is
    the deterministic safe fallback and fallback_used is True with the
    failed check names in issues.
    """

    subject: str = Field(min_length=1)
    text: str = Field(min_length=1)
    grounded: bool = Field()
    fallback_used: bool = Field(default=False)
    issues: list[str] = Field(default_factory=list)


class SchemeInsights(BaseModel):
    """All per-scheme intelligence outputs, keyed by one scheme_id.

    Each slot reuses its authoritative result schema unchanged; None
    means that stage did not execute for this scheme (never a faked
    empty verdict). Global ranking stays in ranked_matches, associated
    by scheme_id.
    """

    scheme_id: str = Field()
    eligibility: Optional[EligibilityResult] = Field(default=None)
    near_miss: Optional[NearMissResult] = Field(default=None)
    pathway: Optional[PathwayResult] = Field(default=None)
    traces: list[DecisionTrace] = Field(default_factory=list)
    explanation: Optional[GeneratedExplanation] = Field(default=None)


class IntelligenceResult(BaseModel):
    """Product-level intelligence container (orchestration contract only).

    Sections mirror the product flow: profile understanding, needs,
    eligibility, ranking, near-miss, support, financial, pathway,
    traces, and optional LLM wording. Each section keeps its own
    authoritative schema; this container holds NO business rules,
    weights, calculations, or decisions.

    stages_completed names executed stages (e.g. "profile",
    "eligibility"); absent stages are simply None/empty, never faked.
    Non-scheme traces (needs/support/financial) live in traces;
    scheme-scoped traces live in SchemeInsights.traces, with the
    scheme's primary explanation in SchemeInsights.explanation and
    every generated wording also collected in explanations.
    What-If stays a separate on-demand operation (WhatIfResult) and is
    not embedded here. Loan/EMI comparisons remain standalone calls;
    coverage_results carries standalone coverage computations.
    """

    profile: Optional[UserProfile] = Field(default=None)
    needs: Optional[NeedAnalysisResult] = Field(default=None)
    schemes: list[SchemeInsights] = Field(default_factory=list)
    ranked_matches: list[MatchResult] = Field(default_factory=list)
    support_plan: Optional[SupportPlan] = Field(default=None)
    coverage_results: list[CoverageResult] = Field(default_factory=list)
    traces: list[DecisionTrace] = Field(default_factory=list)
    explanations: list[GeneratedExplanation] = Field(default_factory=list)
    stages_completed: list[str] = Field(default_factory=list)

    @classmethod
    def from_pipeline_result(cls, result: "PipelineResult") -> "IntelligenceResult":
        """Place a legacy PipelineResult into the product container.

        Pure slot placement (no decisions): profile, per-scheme
        eligibility association, and global ranking carry over; all
        other stages read as not executed.
        """
        return cls(
            profile=result.profile,
            schemes=[
                SchemeInsights(scheme_id=item.scheme_id, eligibility=item)
                for item in result.eligibility_results
            ],
            ranked_matches=list(result.ranked_matches),
            stages_completed=["profile", "eligibility", "matching"],
        )
