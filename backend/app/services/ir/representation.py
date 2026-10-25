"""
IR Representation Service — Phase 5
Retrieves TF-IDF and NMF metadata for an article.
"""

import os
import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from app.models.news_article import NewsArticle
from app.models.article_feature import ArticleTextFeature
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data", "processed", "tfidf"))
METADATA_FILE = os.path.join(INDEX_DIR, "topics_metadata.json")

async def get_article_representation(session: AsyncSession, mind_id: str) -> dict:
    """
    Returns the textual/topical representation of an article.
    """
    # 1. Fetch metadata
    result = await session.execute(
        select(NewsArticle, ArticleTextFeature)
        .join(ArticleTextFeature, NewsArticle.mind_id == ArticleTextFeature.mind_id)
        .where(NewsArticle.mind_id == mind_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Article {mind_id} not found or not indexed.")
        
    article, feature = row
    
    # 2. Load topics mapping
    topics = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            topics = json.load(f)
            
    dominant_topic = feature.dominant_topic
    top_terms = topics.get(str(dominant_topic), []) if dominant_topic is not None else []
    
    return {
        "mind_id": article.mind_id,
        "title": article.title,
        "category": article.category,
        "subcategory": article.subcategory,
        "preprocessing_version": feature.preprocessing_version,
        "dominant_topic": dominant_topic,
        "top_terms": top_terms,
        "topic_weights": json.loads(feature.topic_weights_json),
        "indexed_at": feature.indexed_at.isoformat()
    }
