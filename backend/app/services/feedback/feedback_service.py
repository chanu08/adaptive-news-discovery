"""
Feedback Service — Phase 9

High-level orchestration of the feedback pipeline:
    1. Validate the user and article exist in the database
    2. Record the feedback event
    3. Update the RECENT interest profile
    4. Recalculate Interest Drift using the updated profile

This is the only module that should be imported by the API route.
"""

from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mind_user import MindUser
from app.models.news_article import NewsArticle
from app.models.feedback_event import FeedbackEvent
from app.services.feedback.constants import VALID_EVENT_TYPES, EVENT_WEIGHTS
from app.services.feedback.feedback_store import (
    record_feedback,
    get_all_feedback_for_profile_update,
    get_recent_feedback_events,
)
from app.services.feedback.profile_updater import update_recent_profile_from_feedback
from app.services.interest_modeling.drift_service import calculate_drift_for_user


class FeedbackValidationError(ValueError):
    """Raised when a feedback request fails validation."""


async def submit_feedback(
    session: AsyncSession,
    user_mind_id: str,
    article_mind_id: str,
    event_type: str,
    source: str = "recommendation",
) -> dict:
    """
    Full feedback pipeline:
        validate → record → update profile → recalculate drift

    Returns a response dict suitable for the API endpoint.
    """
    # --- Validation ---
    event_type = event_type.upper()
    if event_type not in VALID_EVENT_TYPES:
        raise FeedbackValidationError(
            f"Invalid event_type '{event_type}'. Must be one of: {sorted(VALID_EVENT_TYPES)}"
        )

    user = await session.scalar(
        select(MindUser).where(MindUser.mind_user_id == user_mind_id)
    )
    if not user:
        raise FeedbackValidationError(f"User '{user_mind_id}' not found.")

    article = await session.scalar(
        select(NewsArticle).where(NewsArticle.mind_id == article_mind_id)
    )
    if not article:
        raise FeedbackValidationError(f"Article '{article_mind_id}' not found.")

    # Embed article metadata into the event for profile update use
    metadata = {
        "category": article.category or "",
        "subcategory": article.subcategory or "",
        "title": article.title or "",
    }

    # --- Record event ---
    event = await record_feedback(
        session=session,
        user_mind_id=user_mind_id,
        article_mind_id=article_mind_id,
        event_type=event_type,
        source=source,
        metadata=metadata,
    )

    # --- Update RECENT profile ---
    all_feedback = await get_all_feedback_for_profile_update(session, user_mind_id)
    updated_profile = await update_recent_profile_from_feedback(
        session, user_mind_id, all_feedback
    )

    # --- Recalculate drift ---
    await calculate_drift_for_user(session, user_mind_id)

    # Commit everything in one transaction
    await session.commit()

    return {
        "status": "recorded",
        "user_id": user_mind_id,
        "mind_id": article_mind_id,
        "event_type": event_type,
        "event_weight": event.event_weight,
        "timestamp": event.event_timestamp.isoformat() + "Z",
        "article_category": metadata["category"],
        "article_subcategory": metadata["subcategory"],
        "profile_updated": True,
        "drift_recalculated": True,
    }


async def get_feedback_history(
    session: AsyncSession,
    user_mind_id: str,
    limit: int = 50,
) -> list[dict]:
    """
    Returns recent feedback events for a user (newest first).
    """
    events = await get_recent_feedback_events(session, user_mind_id, limit)
    return [
        {
            "mind_id": e.article_mind_id,
            "event_type": e.event_type,
            "event_weight": e.event_weight,
            "timestamp": e.event_timestamp.isoformat() + "Z",
            "source": e.source,
        }
        for e in events
    ]
