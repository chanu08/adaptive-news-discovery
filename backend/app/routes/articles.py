"""
News Articles API Routes — Phase 2

Read-only endpoints for inspecting ingested MIND articles.

GET /api/articles          — paginated list with optional filters
GET /api/articles/{id}     — single article by internal DB id
GET /api/articles/mind/{mind_id} — lookup by original MIND NewsID
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.models.news_article import NewsArticle
from app.schemas.news_article import (
    NewsArticleDetailResponse,
    NewsArticleListResponse,
    NewsArticleResponse,
)

router = APIRouter()


@router.get("/", response_model=NewsArticleListResponse)
async def list_news_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=200, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    split: Optional[str] = Query(None, description="Filter by split (train/dev/test)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return a paginated list of MIND news articles.
    Supports filtering by category and split.
    """
    stmt = select(NewsArticle)
    count_stmt = select(func.count()).select_from(NewsArticle)

    if category:
        stmt = stmt.where(NewsArticle.category == category)
        count_stmt = count_stmt.where(NewsArticle.category == category)
    if split:
        from app.models.news_article import ArticleSplit
        stmt = stmt.join(ArticleSplit).where(ArticleSplit.split == split)
        count_stmt = count_stmt.join(ArticleSplit).where(ArticleSplit.split == split)

    total = (await db.execute(count_stmt)).scalar_one()
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size).order_by(NewsArticle.id)

    result = await db.execute(stmt)
    articles = result.scalars().all()

    return NewsArticleListResponse(
        items=[NewsArticleResponse.model_validate(a) for a in articles],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, -(-total // page_size)),  # ceiling division
    )


@router.get("/mind/{mind_id}", response_model=NewsArticleDetailResponse)
async def get_article_by_mind_id(
    mind_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return a news article by its original MIND NewsID (e.g. N55689)."""
    result = await db.execute(
        select(NewsArticle).where(NewsArticle.mind_id == mind_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with MIND ID '{mind_id}' not found.",
        )
    return NewsArticleDetailResponse.from_orm_with_entities(article)


@router.get("/{article_id}", response_model=NewsArticleDetailResponse)
async def get_article_by_id(
    article_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a news article by internal database ID."""
    result = await db.execute(
        select(NewsArticle).where(NewsArticle.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with ID {article_id} not found.",
        )
    return NewsArticleDetailResponse.from_orm_with_entities(article)

@router.get("/mind/{mind_id}/representation")
async def get_article_representation_endpoint(mind_id: str, db: AsyncSession = Depends(get_db)):
    """Return the TF-IDF and NMF topic representation for an article."""
    from app.services.ir.representation import get_article_representation
    
    return await get_article_representation(db, mind_id)
