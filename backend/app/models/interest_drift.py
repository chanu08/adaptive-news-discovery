"""
User Interest Drift Model — Phase 4
Stores the results of comparing a user's LONG_TERM and RECENT profiles.
"""

from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base

class UserInterestDrift(Base):
    __tablename__ = "user_interest_drifts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_mind_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    
    jsd_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence_level: Mapped[str] = mapped_column(String(20), nullable=False)  # INSUFFICIENT, LIMITED, ADEQUATE
    
    # JSON strings
    category_drift_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    subcategory_drift_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    emerging_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    declining_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    stable_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_user_interest_drifts_jsd", "jsd_score"),
    )

    def __repr__(self) -> str:
        return f"<UserInterestDrift user={self.user_mind_id} jsd={self.jsd_score} ev={self.evidence_level}>"
