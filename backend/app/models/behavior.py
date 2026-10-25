"""
Behavior / Impression Log ORM model — Phase 2

One row per impression log from MIND behaviors.tsv.
Each row can contain multiple articles in History and Impressions.

Relationships:
  - behaviors.user_mind_id  → mind_users.mind_user_id
  - impression_items.article_mind_id → news_articles.mind_id
"""

from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class BehaviorLog(Base):
    """One impression session per row from behaviors.tsv."""

    __tablename__ = "behavior_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # MIND source fields
    impression_id: Mapped[int]  = mapped_column(Integer, nullable=False, index=True)
    user_mind_id: Mapped[str]   = mapped_column(String(20), nullable=False, index=True)
    time: Mapped[datetime]       = mapped_column(DateTime, nullable=True)

    # Raw history and impressions stored as space-separated strings
    # (parsed article-level data goes into impression_items)
    history_raw: Mapped[str]     = mapped_column(Text, nullable=True)
    impressions_raw: Mapped[str] = mapped_column(Text, nullable=True)

    # Derived counts (computed during ingestion)
    history_length: Mapped[int]  = mapped_column(Integer, default=0, nullable=False)
    n_impressions: Mapped[int]   = mapped_column(Integer, default=0, nullable=False)
    n_clicks: Mapped[int]        = mapped_column(Integer, default=0, nullable=False)

    split: Mapped[str]           = mapped_column(String(10), nullable=False)  # train/dev/test
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_behavior_logs_user_split", "user_mind_id", "split"),
        Index("ix_behavior_logs_time", "time"),
    )

    def __repr__(self) -> str:
        return (
            f"<BehaviorLog impression_id={self.impression_id} "
            f"user={self.user_mind_id} clicks={self.n_clicks}/{self.n_impressions}>"
        )


class ImpressionItem(Base):
    """
    One row per article shown in a single impression session.
    Extracted from behaviors.tsv Impressions column.

    Allows efficient querying: which articles were clicked by which users.
    """

    __tablename__ = "impression_items"

    id: Mapped[int]              = mapped_column(Integer, primary_key=True, autoincrement=True)
    behavior_log_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("behavior_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    article_mind_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    clicked: Mapped[int]         = mapped_column(Integer, nullable=False)  # 1 = clicked, 0 = not
    split: Mapped[str]           = mapped_column(String(10), nullable=False)

    __table_args__ = (
        Index("ix_impression_items_article_clicked", "article_mind_id", "clicked"),
    )

    def __repr__(self) -> str:
        return (
            f"<ImpressionItem article={self.article_mind_id} clicked={self.clicked}>"
        )
