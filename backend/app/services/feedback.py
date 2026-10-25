"""
Feedback Service — PLACEHOLDER

Phase 6 Implementation Plan:
─────────────────────────────
User feedback (like, dislike, read, skip) closes the recommendation loop
by updating both the user profile and the IR system's relevance signals.

Feedback types (planned):
  - READ     → strong positive signal  (weight: +1.0)
  - LIKE     → strong positive signal  (weight: +1.5)
  - DISLIKE  → negative signal         (weight: -1.0)
  - SKIP     → weak negative signal    (weight: -0.2)

Processing pipeline:
  1. Record interaction in database (InteractionLog table)
  2. Extract article's topic/TF-IDF vector
  3. Update user's recent_profile vector
  4. Trigger drift detection if recent_profile has changed significantly
  5. Schedule periodic long_term_profile merge (e.g., nightly batch)

Public API (planned):
  record_feedback(user_id, article_id, feedback_type) -> None
  process_feedback_batch(interactions: List[Interaction]) -> None
"""


def record_feedback(user_id: int, article_id: int, feedback_type: str) -> None:
    """[PLACEHOLDER] Record a single user feedback event."""
    raise NotImplementedError("Feedback recording will be implemented in Phase 6.")


def process_feedback_batch(interactions: list) -> None:
    """[PLACEHOLDER] Process a batch of interactions to update user profiles."""
    raise NotImplementedError("Batch feedback processing will be implemented in Phase 6.")
