"""Financial intelligence: deterministic money arithmetic on structured data.

Answers: how much is needed, covered, gapped, self-funded; what a loan
costs when rate/tenure are explicit; how known options compare.

Rules (never violated):
- Plain-Python arithmetic only; the LLM never computes.
- Only explicitly structured amounts are used. Unknown stays None and
  is never coerced to zero. Negative inputs fail validation.
- No invented rates, subsidies, limits, terms, or eligibility.
- Coverage uses the best single known option capped at required —
  options are never summed without convergence data.
- Support Planning stays the need↔scheme mapper; this layer does pure
  financial math and never decides eligibility. No DB/API/frontend.
"""

from intelligence_engine.schemas import (
    CoverageResult,
    FinancialOption,
    LoanResult,
    LoanTerms,
    OptionComparison,
    SchemeSupport,
    SupportNeed,
)


class FinancialError(ValueError):
    """Raised for inconsistent financial inputs (never silently fixed)."""


def _round_money(value: float) -> float:
    """Round calculated monetary outputs to 2 decimals."""
    return round(float(value), 2)


class FinancialIntelligence:
    """Pure deterministic financial calculations on validated models."""

    def coverage(
        self, required: float | None, options: list[FinancialOption]
    ) -> CoverageResult:
        """Cover a known requirement with the best single known option.

        Args:
            required: Explicitly known need amount (>= 0) or None.
            options: Candidate options (duplicates merge to one entry).

        Returns:
            CoverageResult; unknown whenever required or every option
            amount is unknown. covered caps at required.
        """
        if required is not None and (not isinstance(required, (int, float)) or required < 0):
            raise FinancialError("required must be a non-negative number or None.")
        unique = self._dedupe(options)
        known = [option.amount for option in unique if option.amount is not None]
        if required is None or not known:
            return CoverageResult(
                required=required,
                coverage_known=False,
                reasons=["Coverage unknown: requirement or all option amounts unknown."],
            )
        covered = _round_money(min(float(required), max(known)))
        uncovered = _round_money(float(required) - covered)
        return CoverageResult(
            required=float(required),
            covered=covered,
            uncovered=uncovered,
            own_contribution=uncovered,
            coverage_known=True,
            fully_covered=uncovered == 0,
            reasons=[f"Best single option covers {_round_money(covered)} of {float(required)}."],
        )

    def coverage_for_need(
        self, need: SupportNeed, supports: list[SchemeSupport]
    ) -> CoverageResult:
        """Bridge Support Planning data into pure financial arithmetic.

        Uses only needs with known amounts and known periods, matched to
        supports with the same explicit period (never converted).
        Eligibility is the caller's job: pass supports already
        filtered to eligible schemes — this method never checks it.
        """
        if need.amount is None or need.amount_period == "unknown":
            return CoverageResult(
                coverage_known=False,
                reasons=["Need amount or period unknown: coverage cannot be calculated."],
            )
        options = [
            FinancialOption(label=f"support-{index}", amount=support.max_amount)
            for index, support in enumerate(supports)
            if need.need_type in support.covers_need_types
            and support.max_amount_period == need.amount_period
        ]
        if not options:
            return CoverageResult(
                coverage_known=False,
                reasons=["No support with a matching known period: coverage cannot be calculated."],
            )
        return self.coverage(float(need.amount), options)

    def loan_schedule(self, terms: LoanTerms) -> LoanResult:
        """Standard amortizing EMI: P*r*(1+r)^n/((1+r)^n-1), r = annual/1200.

        Zero interest yields principal/tenure exactly. No affordability
        claim is made from these outputs alone.
        """
        months = terms.tenure_months
        if terms.annual_rate_percent == 0:
            emi = _round_money(terms.principal / months)
            return LoanResult(monthly_emi=emi, total_payable=_round_money(emi * months), total_interest=0.0)
        rate = terms.annual_rate_percent / 1200.0
        factor = (1 + rate) ** months
        emi = _round_money(terms.principal * rate * factor / (factor - 1))
        total = _round_money(emi * months)
        return LoanResult(monthly_emi=emi, total_payable=total, total_interest=_round_money(total - terms.principal))

    def compare(self, options: list[FinancialOption]) -> OptionComparison:
        """Rank known-amount options highest-first; unknowns stay unranked."""
        unique = self._dedupe(options)
        known = sorted(
            ((option.label, option.amount) for option in unique if option.amount is not None),
            key=lambda item: (-item[1], item[0]),
        )
        unknown = sorted(option.label for option in unique if option.amount is None)
        reasons = [f"{label}: {amount}" for label, amount in known]
        reasons.extend(f"{label}: amount unknown (unranked)" for label in unknown)
        return OptionComparison(
            ranked_labels=[label for label, _ in known],
            unknown_labels=unknown,
            best_amount=known[0][1] if known else None,
            reasons=reasons,
        )

    @staticmethod
    def _dedupe(options: list[FinancialOption]) -> list[FinancialOption]:
        """Collapse identical (label, kind, amount) options, order kept."""
        unique: list[FinancialOption] = []
        seen: set[tuple[str, str | None, float | None]] = set()
        for option in options:
            key = (option.label, option.kind, option.amount)
            if key not in seen:
                seen.add(key)
                unique.append(option)
        return unique
