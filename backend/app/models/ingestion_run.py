"""
Ingestion metadata / run log — Phase 2

Tracks each ingestion run so we can:
  - know when data was last ingested
  - safely re-run without duplicating
  - diagnose past ingestion failures
"""

from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str]      = mapped_column(String(40), unique=True, nullable=False)
    split: Mapped[str]       = mapped_column(String(10), nullable=False)  # train/dev/test
    started_at: Mapped[datetime]  = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Counts
    news_read: Mapped[int]        = mapped_column(Integer, default=0)
    news_inserted: Mapped[int]    = mapped_column(Integer, default=0)
    news_skipped: Mapped[int]     = mapped_column(Integer, default=0)
    behaviors_read: Mapped[int]   = mapped_column(Integer, default=0)
    behaviors_inserted: Mapped[int] = mapped_column(Integer, default=0)
    behaviors_skipped: Mapped[int]  = mapped_column(Integer, default=0)
    items_inserted: Mapped[int]   = mapped_column(Integer, default=0)
    users_inserted: Mapped[int]   = mapped_column(Integer, default=0)

    duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str]   = mapped_column(String(20), default="running")  # running/success/error
    error_log: Mapped[str] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<IngestionRun run_id={self.run_id!r} status={self.status!r}>"
