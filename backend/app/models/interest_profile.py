"""
User Interest Profile ORM model — Phase 3

Stores the computed long-term and recent interest profiles for a user.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class UserInterestProfile(Base):
    __tablename__ = "user_interest_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_mind_id: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # "LONG_TERM" or "RECENT"
    profile_type: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # JSON encoded strings representing normalized feature weights
    # e.g. {"sports": 0.45, "technology": 0.3}
    categories_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    subcategories_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_user_interest_profiles_user_type", "user_mind_id", "profile_type", unique=True),
    )

    def __repr__(self) -> str:
        return f"<UserInterestProfile user={self.user_mind_id} type={self.profile_type}>"
