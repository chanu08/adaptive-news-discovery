"""
Feedback Store — Phase 9

Handles all database read/write operations for FeedbackEvent records.
Implements the deduplication/state-change policy:

    READ  : additive — every READ event is stored as a new row
    LIKE  : state-setter — when a new LIKE or DISLIKE is recorded, the
            *previous* opposing state event for the same (user, article)
            is retained in the DB for audit purposes but the new event
            supersedes it as the authoritative preference signal.

IMPORTANT:
    This module must NOT touch MIND behavior tables (BehaviorLog, ImpressionItem).
    All writes are strictly limited to the feedback_events table.
"""

import json
from datetime import datetime
from typing import Optional
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feedback_event import FeedbackEvent
from app.services.feedback.constants import (
    EVENT_WEIGHTS,
    STATE_EVENT_TYPES,
    ADDITIVE_EVENT_TYPES,
    FEEDBACK_HISTORY_LIMIT,
)


async def record_feedback(
    session: AsyncSession,
    user_mind_id: str,
    article_mind_id: str,
    event_type: str,
    source: str = "recommendation",
    metadata: Optional[dict] = None,
) -> FeedbackEvent:
    """
    Persists a feedback event following the state-change policy.

    For STATE events (LIKE / DISLIKE):
        Previous events of the same type for this (user, article) pair
        are NOT deleted, but the new event is the authoritative state.

    For ADDITIVE events (READ):
        Always creates a new row.

    Returns the newly created FeedbackEvent.
    """
    weight = EVENT_WEIGHTS[event_type]
    now = datetime.utcnow()

    event = FeedbackEvent(
        user_mind_id=user_mind_id,
        article_mind_id=article_mind_id,
        event_type=event_type,
        event_weight=weight,
        event_timestamp=now,
        source=source,
        metadata_json=json.dumps(metadata) if metadata else None,
        created_at=now,
    )
    session.add(event)
    # Flush so we have an id before the caller commits
    await session.flush()
    return event


async def get_recent_feedback_events(
    session: AsyncSession,
    user_mind_id: str,
    limit: int = FEEDBACK_HISTORY_LIMIT,
) -> list[FeedbackEvent]:
    """
    Returns the most recent feedback events for a user (newest first).
    """
    result = await session.execute(
        select(FeedbackEvent)
        .where(FeedbackEvent.user_mind_id == user_mind_id)
        .order_by(desc(FeedbackEvent.event_timestamp))
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_authoritative_state(
    session: AsyncSession,
    user_mind_id: str,
    article_mind_id: str,
) -> Optional[FeedbackEvent]:
    """
    Returns the latest LIKE or DISLIKE event for a (user, article) pair,
    or None if no state event has been recorded.
    This is used to determine the current explicit preference state.
    """
    result = await session.execute(
        select(FeedbackEvent)
        .where(FeedbackEvent.user_mind_id == user_mind_id)
        .where(FeedbackEvent.article_mind_id == article_mind_id)
        .where(FeedbackEvent.event_type.in_(STATE_EVENT_TYPES))
        .order_by(desc(FeedbackEvent.event_timestamp))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_all_feedback_for_profile_update(
    session: AsyncSession,
    user_mind_id: str,
) -> list[FeedbackEvent]:
    """
    Returns all feedback events for a user ordered by timestamp (oldest first).
    Used by the profile updater to build the feedback-weighted interest distribution.

    For STATE events, only the latest LIKE/DISLIKE per (user, article) pair
    is included — earlier opposing events are filtered out so that a
    LIKE → DISLIKE transition doesn't double-count.
    """
    all_events = await session.execute(
        select(FeedbackEvent)
        .where(FeedbackEvent.user_mind_id == user_mind_id)
        .order_by(FeedbackEvent.event_timestamp)
    )
    rows = list(all_events.scalars().all())

    # For STATE events: keep only the latest per (article, event_type) is NOT enough.
    # We need to keep only the latest STATE event per ARTICLE (ignoring type),
    # so that LIKE → DISLIKE replaces rather than accumulates.
    # We do a two-pass filter:
    latest_state: dict[str, FeedbackEvent] = {}
    additive_events: list[FeedbackEvent] = []

    for evt in rows:
        if evt.event_type in STATE_EVENT_TYPES:
            # Overwrite — last one seen wins (rows already sorted oldest→newest)
            latest_state[evt.article_mind_id] = evt
        else:
            additive_events.append(evt)

    # Merge and sort by timestamp
    merged = list(latest_state.values()) + additive_events
    merged.sort(key=lambda e: e.event_timestamp)
    return merged
