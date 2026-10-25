"""
Pydantic schemas for Article — request / response validation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ArticleBase(BaseModel):
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    """Schema used when creating a new article."""
    pass


class ArticleUpdate(BaseModel):
    """Schema used when partially updating an article."""
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None


class ArticleResponse(ArticleBase):
    """Schema returned to the client."""
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
