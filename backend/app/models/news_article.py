"""
MIND News Article ORM model — Phase 2

Extends the basic article model to store all MIND news.tsv fields.
Original MIND IDs are preserved as `mind_id`.

MIND news.tsv columns (tab-separated, no header):
  0  NewsID        e.g. N55689
  1  Category      e.g. lifestyle
  2  SubCategory   e.g. lifestyleroyals
  3  Title         plain text
  4  Abstract      plain text (may be empty)
  5  URL           full MSN article URL
  6  TitleEntities JSON array of named entities in title
  7  AbstractEntities JSON array of named entities in abstract
"""

from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Index, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class NewsArticle(Base):
    __tablename__ = "news_articles"

    # ── Primary key ─────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── MIND source identifier (preserved from raw data) ─────────────────────
    mind_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)

    # ── Content fields ───────────────────────────────────────────────────────
    category: Mapped[str]       = mapped_column(String(100), nullable=False, index=True)
    subcategory: Mapped[str]    = mapped_column(String(100), nullable=True)
    title: Mapped[str]          = mapped_column(String(1000), nullable=False)
    abstract: Mapped[str]       = mapped_column(Text, nullable=True)
    url: Mapped[str]            = mapped_column(String(2000), nullable=True)

    # ── Entity metadata (stored as raw JSON strings) ──────────────────────────
    title_entities: Mapped[str]    = mapped_column(Text, nullable=True)   # JSON
    abstract_entities: Mapped[str] = mapped_column(Text, nullable=True)   # JSON

    # ── Record metadata ───────────────────────────────────────────────────────
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    
    article_splits: Mapped[list["ArticleSplit"]] = relationship(
        "ArticleSplit", back_populates="article", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<NewsArticle mind_id={self.mind_id!r} title={self.title[:40]!r}>"


class ArticleSplit(Base):
    """
    Many-to-many representation mapping canonical articles to the datasets they appeared in.
    """
    __tablename__ = "article_splits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    split: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    article: Mapped["NewsArticle"] = relationship("NewsArticle", back_populates="article_splits")

    __table_args__ = (
        UniqueConstraint("article_id", "split", name="uix_article_split"),
    )

    def __repr__(self) -> str:
        return f"<ArticleSplit article_id={self.article_id} split={self.split!r}>"
