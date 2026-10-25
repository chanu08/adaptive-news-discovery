"""
Explanation Generator Service — PLACEHOLDER

Phase 5 Implementation Plan:
─────────────────────────────
Each recommendation will be accompanied by a human-readable
explanation describing *why* it was recommended.

Explanation types (planned):
  1. Relevance explanation   — "Recommended because you searched for X"
  2. Profile explanation     — "Matches your long-term interest in Technology"
  3. Drift explanation       — "Based on your recent interest shift toward Finance"
  4. Diversity explanation   — "Included to broaden your news perspective"

Template approach:
  - Rule-based templates filled with top TF-IDF terms and topic labels
  - No LLM used — explanations derived directly from IR signals

Public API (planned):
  generate_explanation(article_id, user_id, signals: ExplanationSignals) -> str
"""


def generate_explanation(article_id: int, user_id: int, signals: dict) -> str:
    """[PLACEHOLDER] Generate a natural-language explanation for a recommendation."""
    raise NotImplementedError("Explanation generation will be implemented in Phase 5.")
