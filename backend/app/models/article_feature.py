"""
Article Text Feature Model — Phase 5
Stores metadata and topic information from the TF-IDF and NMF pipeline.
"""

from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base

class ArticleTextFeature(Base):
    __tablename__ = "article_text_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mind_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    
    preprocessing_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    dominant_topic: Mapped[int] = mapped_column(Integer, nullable=True)
    topic_weights_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ArticleTextFeature mind_id={self.mind_id} dominant_topic={self.dominant_topic}>"
