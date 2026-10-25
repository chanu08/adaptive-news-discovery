"""
MIND User ORM model — Phase 2

Maps MIND user IDs to internal user records.

MIND behaviors.tsv columns (tab-separated, no header):
  0  ImpressionID   integer (unique per log row)
  1  UserID         e.g. U13740
  2  Time           e.g. 11/11/2019 9:43:18 AM
  3  History        space-separated NewsIDs that user clicked before
  4  Impressions    space-separated "NewsID-Label" where Label=1 click, 0=no-click
"""

from datetime import datetime
from sqlalchemy import String, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class MindUser(Base):
    __tablename__ = "mind_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # MIND source identifier
    mind_user_id: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )

    # Aggregated stats (updated during ingestion)
    total_impressions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_clicks: Mapped[int]      = mapped_column(Integer, default=0, nullable=False)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<MindUser mind_user_id={self.mind_user_id!r}>"
