"""Hybrid input merger: form + text/voice into one canonical input.

Form, text, and voice are input modalities, not separate pipelines.
This module merges a structured form profile (and form needs) with
optionally extracted text data into one validated UserProfile plus one
needs list, which then flow through the unchanged Intelligence Engine.

Precedence: form-provided (non-None) values always win; extraction
fills only fields the form left as None. Nothing is invented, unknowns
stay unknown, and inputs are never mutated. Need merging reuses the
existing merge_duplicate_needs() contract (same type+period merges).
"""

from intelligence_engine.need_analyzer import merge_duplicate_needs, total_one_time
from intelligence_engine.schemas import (
    NeedAnalysisResult,
    SupportNeed,
    UserProfile,
    ValueConflict,
)


def merge_profiles(form: UserProfile, extracted: UserProfile | None) -> UserProfile:
    """Merge form profile with extracted profile; form non-None wins.

    Only safe when detect_conflicts() reports nothing: any genuine
    conflict must go through clarification + resolve_conflicts() first,
    never through silent form-wins merging.

    Args:
        form: Structured form profile (authoritative where provided).
        extracted: Optionally extracted profile, or None.

    Returns:
        New validated UserProfile (inputs untouched).
    """
    if extracted is None:
        return UserProfile.model_validate(form.model_dump())
    form_data = form.model_dump()
    extracted_data = extracted.model_dump()
    merged = {
        field: form_data[field] if form_data[field] is not None else extracted_data[field]
        for field in form_data
    }
    return UserProfile.model_validate(merged)


def merge_needs(
    form_needs: list[SupportNeed] | None,
    extracted_needs: list[SupportNeed] | None,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> list[SupportNeed]:
    """Combine needs from both sources with existing duplicate merging.

    Same (need_type, amount_period) entries merge per the established
    contract; distinct needs from either source are all preserved.
    Keys in exclude are skipped (used to hold disputed needs out of
    merged results until the user confirms). Inputs are never mutated.
    """
    combined = [
        need
        for need in list(form_needs or []) + list(extracted_needs or [])
        if (need.need_type, need.amount_period) not in exclude
    ]
    return merge_duplicate_needs(combined)


def _need_key(need: SupportNeed) -> tuple[str, str]:
    """Grouping key for cross-source need comparison."""
    return (need.need_type, need.amount_period)


def need_conflict_field(need_type: str, amount_period: str) -> str:
    """Machine reference for a disputed need amount."""
    return f"needs:{need_type}:{amount_period}"


def detect_need_conflicts(
    form_needs: list[SupportNeed] | None,
    extracted_needs: list[SupportNeed] | None,
) -> list[ValueConflict]:
    """Find same-need, different-known-amount disputes across sources.

    A conflict needs the same (need_type, amount_period) on both sides
    with both amounts known and different. Unknown amounts merge per
    the existing contract instead. Deterministic order: form-need order.
    """
    extracted_by_key: dict[tuple[str, str], SupportNeed] = {}
    for need in extracted_needs or []:
        extracted_by_key.setdefault(_need_key(need), need)
    conflicts: list[ValueConflict] = []
    seen: set[tuple[str, str]] = set()
    for need in form_needs or []:
        key = _need_key(need)
        if key in seen:
            continue
        seen.add(key)
        other = extracted_by_key.get(key)
        if other is None or need.amount is None or other.amount is None:
            continue
        if need.amount != other.amount:
            conflicts.append(
                ValueConflict(
                    field=need_conflict_field(*key),
                    form_value=need.amount,
                    extracted_value=other.amount,
                )
            )
    return conflicts


def resolve_need_conflicts(
    form_needs: list[SupportNeed] | None,
    extracted_needs: list[SupportNeed] | None,
    choices: dict[str, str],
) -> list[SupportNeed]:
    """Build the canonical needs list from user-confirmed amount choices.

    Choices map "needs:{type}:{period}" references to "form" or
    "extracted". Unknown references or choice values raise ValueError.
    Non-disputed needs merge per the existing contract. Inputs untouched.
    """
    form_by_key: dict[tuple[str, str], SupportNeed] = {}
    for need in form_needs or []:
        form_by_key.setdefault(_need_key(need), need)
    extracted_by_key: dict[tuple[str, str], SupportNeed] = {}
    for need in extracted_needs or []:
        extracted_by_key.setdefault(_need_key(need), need)
    chosen: list[SupportNeed] = []
    excluded: set[tuple[str, str]] = set()
    for reference, choice in choices.items():
        if choice not in ("form", "extracted"):
            raise ValueError(f"Choice for '{reference}' must be 'form' or 'extracted'.")
        parts = reference.split(":", 2)
        if len(parts) != 3 or parts[0] != "needs":
            raise ValueError(f"Unknown need reference: '{reference}'.")
        key = (parts[1], parts[2])
        source = form_by_key if choice == "form" else extracted_by_key
        if key not in source:
            raise ValueError(f"No {choice} need available for '{reference}'.")
        excluded.add(key)
        chosen.append(source[key])
    rest = merge_needs(form_needs, extracted_needs, frozenset(excluded))
    return merge_duplicate_needs(chosen + rest)


def merge_inputs(
    form_profile: UserProfile,
    extracted_profile: UserProfile | None = None,
    form_needs: list[SupportNeed] | None = None,
    extracted_needs: list[SupportNeed] | None = None,
    business_goal: str | None = None,
    exclude_need_keys: frozenset[tuple[str, str]] = frozenset(),
    notes: list[str] | None = None,
) -> tuple[UserProfile, NeedAnalysisResult | None]:
    """Merge all hybrid inputs into canonical (profile, needs-or-None.

    Returns None for needs when both sources are empty, preserving the
    existing convention that absent needs skip support/financial stages.
    Keys in exclude_need_keys stay out of merged needs (disputed amounts
    await user confirmation instead). Analyzer transparency notes are
    preserved verbatim (deduplicated, order kept); nothing is invented.
    """
    profile = merge_profiles(form_profile, extracted_profile)
    merged_needs = merge_needs(form_needs, extracted_needs, exclude_need_keys)
    if not merged_needs:
        return profile, None
    merged_notes: list[str] = []
    seen_notes: set[str] = set()
    for note in notes or []:
        if note not in seen_notes:
            seen_notes.add(note)
            merged_notes.append(note)
    return profile, NeedAnalysisResult(
        business_goal=business_goal,
        needs=merged_needs,
        total_requested=total_one_time(merged_needs),
        notes=merged_notes,
    )


def _is_blank(value: object) -> bool:
    """None, empty, or whitespace-only strings carry no assertion."""
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _canonical_amount(value: object, period: object) -> tuple[bool, float]:
    """Annualized comparable amount, or (False, 0.0) when indeterminable.

    Only equal, explicitly known periods annualize (monthly × 12);
    unknown periods never convert — equivalence cannot be established.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False, 0.0
    if period == "annual":
        return True, float(value)
    if period == "monthly":
        return True, float(value) * 12
    return False, 0.0


def _values_equal(field: str, first: object, second: object) -> bool:
    """Field-aware equivalence without fuzzy matching.

    Strings compare stripped and case-insensitively (whitespace/casing
    carry no meaning in any profile text field). Annualized income
    comparison is handled by callers via _canonical_amount; here numbers
    compare exactly.
    """
    if isinstance(first, str) and isinstance(second, str):
        return first.strip().lower() == second.strip().lower()
    return first == second


def detect_conflicts(
    form: UserProfile, extracted: UserProfile | None
) -> list[ValueConflict]:
    """Find genuinely conflicting values for the same canonical field.

    A conflict needs both sides asserting (non-blank), different after
    existing normalization, and the form value must not be a schema
    default (defaults such as income_period "annual" are not user
    assertions). Strings compare stripped and case-insensitively;
    income compares annualized when both periods are explicitly known
    (monthly ×12 per the schema conversion); unknown periods never
    convert. Deterministic field order, no fuzzy matching.

    Args:
        form: Structured form profile.
        extracted: Optionally extracted profile (None means no conflicts).

    Returns:
        ValueConflict per disputed field, both values preserved.
    """
    if extracted is None:
        return []
    form_data = form.model_dump()
    extracted_data = extracted.model_dump()
    form_ok, form_annual = _canonical_amount(
        form_data.get("annual_family_income"), form_data.get("income_period")
    )
    ext_ok, ext_annual = _canonical_amount(
        extracted_data.get("annual_family_income"), extracted_data.get("income_period")
    )
    income_equivalent = form_ok and ext_ok and form_annual == ext_annual
    conflicts: list[ValueConflict] = []
    for field in form_data:
        form_value = form_data[field]
        extracted_value = extracted_data[field]
        if _is_blank(form_value) or _is_blank(extracted_value):
            continue
        if form_value == UserProfile.model_fields[field].default:
            continue
        if field == "income_period" and income_equivalent:
            continue
        if field == "annual_family_income":
            form_ok, form_annual = _canonical_amount(
                form_value, form_data.get("income_period")
            )
            ext_ok, ext_annual = _canonical_amount(
                extracted_value, extracted_data.get("income_period")
            )
            if not form_ok or not ext_ok:
                conflicts.append(
                    ValueConflict(
                        field=field, form_value=form_value, extracted_value=extracted_value
                    )
                )
            elif form_annual != ext_annual:
                conflicts.append(
                    ValueConflict(
                        field=field, form_value=form_value, extracted_value=extracted_value
                    )
                )
            continue
        if not _values_equal(field, form_value, extracted_value):
            conflicts.append(
                ValueConflict(
                    field=field, form_value=form_value, extracted_value=extracted_value
                )
            )
    return conflicts


def resolve_conflicts(
    form: UserProfile,
    extracted: UserProfile | None,
    choices: dict[str, str],
) -> UserProfile:
    """Build the canonical profile from user-confirmed conflict choices.

    Args:
        form: Structured form profile (never mutated).
        extracted: Optionally extracted profile (never mutated).
        choices: field -> "form" or "extracted". Unknown fields,
            unknown choice values, or "extracted" without an extracted
            profile all raise ValueError.

    Returns:
        New validated UserProfile with confirmed values applied.
    """
    merged = form.model_dump()
    extracted_data = extracted.model_dump() if extracted is not None else {}
    for field, choice in choices.items():
        if field not in UserProfile.model_fields:
            raise ValueError(f"Unknown profile field: '{field}'.")
        if choice not in ("form", "extracted"):
            raise ValueError(f"Choice for '{field}' must be 'form' or 'extracted'.")
        if choice == "extracted":
            if extracted is None:
                raise ValueError(f"No extracted value available for '{field}'.")
            merged[field] = extracted_data[field]
    return UserProfile.model_validate(merged)
