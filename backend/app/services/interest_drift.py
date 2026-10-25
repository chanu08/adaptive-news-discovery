"""
Interest Drift Service — PLACEHOLDER

Phase 4 Implementation Plan:
─────────────────────────────
Interest drift is the core novelty of this IR system.

It detects when a user's recent interests have shifted relative
to their long-term profile and adjusts the retrieval weights accordingly.

Algorithm (planned):
  1. Compute cosine similarity between recent_vector and long_term_vector.
  2. If similarity drops below drift_threshold → drift detected.
  3. Blend retrieval weights:
       score = α × long_term_score + (1-α) × recent_score
     where α decreases when drift is detected (favour recent interests).
  4. Gradually merge recent_vector into long_term_vector over time.

Public API (planned):
  detect_drift(user_id: int) -> DriftResult
  compute_blend_weight(user_id: int) -> float   # returns α ∈ [0, 1]
  apply_drift_adjustment(candidates, user_id) -> List[ScoredArticle]
"""


def detect_drift(user_id: int) -> dict:
    """[PLACEHOLDER] Detect interest drift for a user."""
    raise NotImplementedError("Interest drift detection will be implemented in Phase 4.")


def compute_blend_weight(user_id: int) -> float:
    """[PLACEHOLDER] Compute the long-term vs. recent blend weight α."""
    raise NotImplementedError("Blend weight computation will be implemented in Phase 4.")
