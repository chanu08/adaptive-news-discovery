"""
FeedbackEvent ORM Model — Phase 9

Stores online user feedback events SEPARATELY from historical MIND data.
This table must never be confused with or merged into the original
MIND behavior records. It represents post-recommendation user signals.

Event Types:
    READ    : User viewed/opened the article        (+0.5)
    LIKE    : Explicit positive signal              (+1.0)
    DISLIKE : Explicit negative signal              (-1.0)

State-change policy:
    - READ  : Additive — each READ event is stored independently
    - LIKE  : Upsert-style — the latest LIKE/DISLIKE for a given
              (user, article) pair is the authoritative state.
              Previous opposing events are NOT deleted (preserved for
              audit/history) but are superseded by the latest event.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # The MIND user ID (e.g. "U536528") — links to MindUser without FK constraint
    # to avoid coupling the feedback table to MIND ingestion lifecycle.
    user_mind_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # The MIND article ID (e.g. "N12345") — links to NewsArticle without FK constraint
    article_mind_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # One of: READ, LIKE, DISLIKE
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # Signed numeric weight at time of event (design choice, not psychological truth)
    event_weight: Mapped[float] = mapped_column(Float, nullable=False)

    # Server-assigned event creation time (UTC).
    # We do NOT accept raw client timestamps to avoid fabricated temporal data.
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Source tag: 'recommendation' if sourced from our pipeline, 'manual' for test events
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="recommendation")

    # Free-form JSON metadata (e.g. session ID, recommendation rank at time of feedback)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Composite index for quick per-user and per-user/article lookups
        Index("ix_feedback_user_article", "user_mind_id", "article_mind_id"),
        Index("ix_feedback_user_ts", "user_mind_id", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackEvent user={self.user_mind_id} article={self.article_mind_id} "
            f"type={self.event_type} w={self.event_weight}>"
        )
