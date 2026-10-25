"""
Profile Update Service — Phase 9

Updates the RECENT interest profile to incorporate online feedback signals.

DESIGN:
    Only the RECENT profile is updated — the LONG_TERM profile remains
    grounded purely in historical MIND interaction data.

UPDATE FORMULA:
    The recent profile is a normalized distribution over categories/subcategories.
    We maintain a separate "positive evidence" and "negative evidence" dict,
    then combine as:

        net[feature] = positive[feature] - negative[feature]

    Negative values are clamped to 0 before normalization (rather than silently)
    because negative feature values in a probability distribution are undefined.
    This clamping is DOCUMENTED: it means strong DISLIKE signals compress
    affinity toward 0 rather than flipping it to a negative probability.

    Finally, the net distribution is L1-normalized (sums to 1.0) with
    the constraint that all values ≥ 0.

    The base historical profile is mixed with feedback evidence using a
    configurable mixing weight to prevent a single feedback event from
    completely dominating the recent profile:

        combined = (1 - mix_alpha) * base_profile
                 + mix_alpha       * feedback_distribution

    mix_alpha = 0.3 (experimental default; 30% feedback, 70% history)

TEMPORAL DECAY:
    Each feedback event's weight is decayed by:
        w_decayed = event_weight × exp(-lambda × age_days)

    where age_days is measured from now (UTC) to the event timestamp.
    This reuses the same exponential decay used in Phase 3 profile_builder.
    Decay lambda = ln(2) / FEEDBACK_HALF_LIFE_DAYS ≈ 0.0495 (14-day half-life).

ISOLATION GUARANTEE:
    This function reads the existing RECENT profile from the DB and returns
    an updated profile dict. The caller is responsible for persisting it.
    This function NEVER modifies MIND behavior tables.
"""

import json
import math
from datetime import datetime
from collections import defaultdict
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feedback_event import FeedbackEvent
from app.services.feedback.constants import FEEDBACK_DECAY_LAMBDA
from app.services.interest_modeling.profile_store import get_user_profile, save_user_profile

# Mixing weight: fraction of the final profile that comes from feedback evidence.
# 0.3 = 30% feedback, 70% historical recent profile.
# This is an experimental value, not a universally optimal constant.
FEEDBACK_MIX_ALPHA: float = 0.3


def _apply_decayed_feedback_weight(
    events: list[FeedbackEvent],
    reference_time: Optional[datetime] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Computes temporally decayed positive and negative category/subcategory
    evidence from a list of feedback events.

    Returns:
        (positive_evidence, negative_evidence) — both dicts of feature → weight
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    positive: dict[str, float] = defaultdict(float)
    negative: dict[str, float] = defaultdict(float)

    for evt in events:
        age_days = (reference_time - evt.event_timestamp).total_seconds() / 86400.0
        age_days = max(0.0, age_days)
        decay = math.exp(-FEEDBACK_DECAY_LAMBDA * age_days)
        decayed_weight = evt.event_weight * decay  # signed

        # The article's category/subcategory is stored in metadata_json
        # when the feedback was created via the API (see feedback_service.py).
        if evt.metadata_json:
            try:
                meta = json.loads(evt.metadata_json)
                cat = meta.get("category", "")
                subcat = meta.get("subcategory", "")
            except (json.JSONDecodeError, AttributeError):
                cat, subcat = "", ""
        else:
            cat, subcat = "", ""

        for feature in [f for f in [cat, subcat] if f]:
            if decayed_weight >= 0:
                positive[feature] += decayed_weight
            else:
                negative[feature] += abs(decayed_weight)

    return dict(positive), dict(negative)


def _normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    """L1-normalizes a non-negative feature distribution. Returns {} if all zeros."""
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in dist.items()}


def _mix_profiles(
    base: dict[str, float],
    feedback: dict[str, float],
    alpha: float = FEEDBACK_MIX_ALPHA,
) -> dict[str, float]:
    """
    Linearly mixes base historical profile with feedback distribution.

        combined[f] = (1 - alpha) * base[f] + alpha * feedback[f]

    Features appearing only in one distribution get weight 0 in the other.
    """
    all_features = set(base) | set(feedback)
    combined = {}
    for f in all_features:
        combined[f] = (1 - alpha) * base.get(f, 0.0) + alpha * feedback.get(f, 0.0)
    # Re-normalize
    return _normalize_distribution(combined)


async def update_recent_profile_from_feedback(
    session: AsyncSession,
    user_mind_id: str,
    feedback_events: list[FeedbackEvent],
) -> dict:
    """
    Reads the existing RECENT profile, incorporates decayed feedback signals,
    and writes the updated profile back to the database.

    Returns the updated profile dict (categories + subcategories).

    Caller must commit the session after this returns.
    """
    # 1. Fetch existing RECENT profile (base)
    recent_profile = await get_user_profile(session, user_mind_id, "RECENT")
    if not recent_profile:
        # Cold user — no base profile; start from empty
        base_cats: dict[str, float] = {}
        base_subs: dict[str, float] = {}
    else:
        base_cats = recent_profile.get("categories", {})
        base_subs = recent_profile.get("subcategories", {})

    if not feedback_events:
        return {"categories": base_cats, "subcategories": base_subs}

    # 2. Compute decayed positive/negative evidence per feature
    pos_cats: dict[str, float] = defaultdict(float)
    neg_cats: dict[str, float] = defaultdict(float)
    pos_subs: dict[str, float] = defaultdict(float)
    neg_subs: dict[str, float] = defaultdict(float)

    now = datetime.utcnow()
    for evt in feedback_events:
        age_days = max(0.0, (now - evt.event_timestamp).total_seconds() / 86400.0)
        decay = math.exp(-FEEDBACK_DECAY_LAMBDA * age_days)
        decayed = evt.event_weight * decay  # signed

        if evt.metadata_json:
            try:
                meta = json.loads(evt.metadata_json)
                cat = meta.get("category", "")
                subcat = meta.get("subcategory", "")
            except (json.JSONDecodeError, AttributeError):
                cat, subcat = "", ""
        else:
            cat, subcat = "", ""

        if decayed >= 0:
            if cat:
                pos_cats[cat] += decayed
            if subcat:
                pos_subs[subcat] += decayed
        else:
            if cat:
                neg_cats[cat] += abs(decayed)
            if subcat:
                neg_subs[subcat] += abs(decayed)

    # 3. Net evidence = positive − negative, clamped to ≥ 0
    # Documented reason for clamping: probability distributions cannot be negative.
    # Strong negative signals compress affinity toward 0.
    all_cats = set(pos_cats) | set(neg_cats)
    net_cats = {f: max(0.0, pos_cats.get(f, 0.0) - neg_cats.get(f, 0.0)) for f in all_cats}
    all_subs = set(pos_subs) | set(neg_subs)
    net_subs = {f: max(0.0, pos_subs.get(f, 0.0) - neg_subs.get(f, 0.0)) for f in all_subs}

    # 4. Normalize feedback distribution
    fb_cats = _normalize_distribution(net_cats)
    fb_subs = _normalize_distribution(net_subs)

    # 5. Mix with base profile
    new_cats = _mix_profiles(base_cats, fb_cats)
    new_subs = _mix_profiles(base_subs, fb_subs)

    # 6. Persist updated RECENT profile
    await save_user_profile(session, user_mind_id, "RECENT", new_cats, new_subs)

    return {"categories": new_cats, "subcategories": new_subs}
