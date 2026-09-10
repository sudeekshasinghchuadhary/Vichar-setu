"""Contract between Raj/Anusha's FastAPI backend and Ananya's intelligence engine.

Do NOT put the core AI/rule/matching implementation here. Ananya can provide a
module exposing the two functions below. The backend will call them through the
adapter in intelligence_adapter.py.
"""
from typing import Any, Protocol

class IntelligenceEngine(Protocol):
    def check_eligibility(self, profile: dict[str, Any], scheme: dict[str, Any]) -> dict[str, Any]: ...
    def match_schemes(self, profile: dict[str, Any], schemes: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
