"""Pipeline orchestration: single entry point for the FastAPI team.

Flow: natural-language text -> ProfileProcessor -> UserProfile
-> EligibilityEngine (every scheme, three-state statuses preserved)
-> MatchingEngine (only "eligible" schemes ranked)
-> PipelineResult.

The pipeline is an ORCHESTRATOR: it coordinates components and
assembles their outputs. It contains no eligibility rules, no
matching weights, no LLM calls, no database/FastAPI access, no
embeddings, and no near-miss/what-if/financial logic.

Components are injected (ProfileProcessor, EligibilityEngine,
MatchingEngine) so tests use fakes and production wires the reals.
Component errors (ProfileProcessingError, EligibilityEngineError,
pydantic ValidationError) propagate unchanged — never hidden or
converted into fabricated data.
"""

from intelligence_engine.eligibility_engine import EligibilityEngine
from intelligence_engine.matching_engine import MatchingEngine
from intelligence_engine.profile_processor import ProfileProcessor
from intelligence_engine.schemas import PipelineResult, Scheme


class IntelligencePipeline:
    """Coordinate the P0 intelligence flow end to end."""

    def __init__(
        self,
        profile_processor: ProfileProcessor,
        eligibility_engine: EligibilityEngine,
        matching_engine: MatchingEngine,
    ) -> None:
        """Store injected components (no service construction here)."""
        self.profile_processor = profile_processor
        self.eligibility_engine = eligibility_engine
        self.matching_engine = matching_engine

    def run(self, user_text: str, schemes: list[Scheme]) -> PipelineResult:
        """Run ProfileProcessor -> EligibilityEngine -> MatchingEngine.

        Args:
            user_text: Natural-language user description.
            schemes: Candidate schemes to evaluate and rank.

        Returns:
            PipelineResult with the validated profile, all eligibility
            results (statuses preserved as-is), and ranked matches for
            eligible schemes only.
        """
        profile = self.profile_processor.process_profile(user_text)
        eligibility_results = self.eligibility_engine.evaluate_many(profile, schemes)
        ranked_matches = self.matching_engine.rank_schemes(profile, schemes, eligibility_results)
        return PipelineResult(
            profile=profile,
            eligibility_results=eligibility_results,
            ranked_matches=ranked_matches,
        )
