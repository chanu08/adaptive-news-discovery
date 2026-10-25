"""
User Profiling Service — PLACEHOLDER

Phase 3 Implementation Plan:
─────────────────────────────
This module maintains a two-component interest model per user:

  Long-term Profile:
    - Aggregated interest vector from all historical interactions
    - Weighted by interaction type: read > like > dislike
    - Updated on every interaction with small learning rate

  Recent Profile (sliding window):
    - Interest vector from the last N interactions
    - Captures current interests and supports drift detection
    - Window size is configurable (e.g., last 20 interactions)

Public API (planned):
  build_user_profile(user_id: int) -> UserProfile
  update_profile(user_id: int, article_id: int, interaction: str) -> None
  get_long_term_vector(user_id: int) -> np.ndarray
  get_recent_vector(user_id: int, window: int) -> np.ndarray
"""


def build_user_profile(user_id: int) -> dict:
    """[PLACEHOLDER] Build initial interest profile for a user."""
    raise NotImplementedError("User profiling will be implemented in Phase 3.")


def update_profile(user_id: int, article_id: int, interaction: str) -> None:
    """[PLACEHOLDER] Update user profile based on an interaction event."""
    raise NotImplementedError("Profile update will be implemented in Phase 3.")
