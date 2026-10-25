"""
Feedback Constants — Phase 9

Centralised configuration for feedback event weights and decay parameters.
These values are DESIGN CHOICES, not psychologically optimal constants.
They must not be hardcoded elsewhere in the codebase.

Weights are signed floating-point values:
    Positive → article reinforces user interest in this category/subcategory
    Negative → article suppresses user interest in this category/subcategory

Temporal decay reuses the same exponential model from Phase 3 weighting.py.
The half-life governs how quickly older feedback events lose influence.

HALF_LIFE_DAYS = 14 means feedback loses half its influence after 14 days.
    lambda = ln(2) / HALF_LIFE_DAYS
"""

import math

# ---------------------------------------------------------------------------
# Event type registry
# ---------------------------------------------------------------------------
VALID_EVENT_TYPES = {"READ", "LIKE", "DISLIKE"}

# ---------------------------------------------------------------------------
# Signed weights per event type
# ---------------------------------------------------------------------------
EVENT_WEIGHTS: dict[str, float] = {
    "LIKE":    +1.0,
    "READ":    +0.5,
    "DISLIKE": -1.0,
}

# ---------------------------------------------------------------------------
# Temporal decay for online feedback
# Half-life: 14 days
# ---------------------------------------------------------------------------
FEEDBACK_HALF_LIFE_DAYS: float = 14.0
FEEDBACK_DECAY_LAMBDA: float = math.log(2) / FEEDBACK_HALF_LIFE_DAYS

# ---------------------------------------------------------------------------
# Maximum feedback events returned in history queries
# ---------------------------------------------------------------------------
FEEDBACK_HISTORY_LIMIT: int = 50

# ---------------------------------------------------------------------------
# Duplicate/state-change policy constants
# ---------------------------------------------------------------------------
# Event types treated as state-setters (last-write-wins per user/article pair)
STATE_EVENT_TYPES = {"LIKE", "DISLIKE"}

# Event types treated as additive (each occurrence is stored independently)
ADDITIVE_EVENT_TYPES = {"READ"}
