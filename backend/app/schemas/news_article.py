"""
Pydantic schemas for MIND news articles — Phase 2.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict
import json


class NewsArticleResponse(BaseModel):
    id: int
    mind_id: str
    category: str
    subcategory: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    url: Optional[str] = None
    splits: list[str] = []
    ingested_at: datetime

    model_config = ConfigDict(from_attributes=True)


class NewsArticleDetailResponse(NewsArticleResponse):
    """Detailed view — includes parsed entities."""
    title_entities: Optional[Any] = None
    abstract_entities: Optional[Any] = None

    @classmethod
    def from_orm_with_entities(cls, article) -> "NewsArticleDetailResponse":
        data = NewsArticleDetailResponse.model_validate(article)
        # Parse JSON entity strings
        for field in ("title_entities", "abstract_entities"):
            raw = getattr(article, field, None)
            if raw:
                try:
                    setattr(data, field, json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    setattr(data, field, [])
        # Extract splits from the relationship if loaded
        if hasattr(article, "article_splits"):
            data.splits = [s.split for s in article.article_splits]
        return data


class NewsArticleListResponse(BaseModel):
    items: list[NewsArticleResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
