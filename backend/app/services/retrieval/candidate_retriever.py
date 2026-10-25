"""
Candidate Retriever — Phase 6
Handles filtering candidates by split and excluding already clicked articles.
"""

from typing import List, Set
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.news_article import NewsArticle, ArticleSplit
from app.services.retrieval.index_loader import ir_store

async def get_candidate_mask(session: AsyncSession, candidate_split: str, clicked_mind_ids: List[str]) -> np.ndarray:
    """
    Returns a boolean mask of shape (num_articles,) where True means the article is a valid candidate.
    """
    ir_store.load()
    num_articles = ir_store.tfidf_matrix.shape[0]
    mask = np.ones(num_articles, dtype=bool)
    
    # 1. Exclude clicked articles
    for mid in set(clicked_mind_ids):
        row_idx = ir_store.mind_id_to_row.get(mid)
        if row_idx is not None:
            mask[row_idx] = False
            
    # 2. Filter by split
    if candidate_split and candidate_split != "all":
        # Fetch valid mind_ids for this split
        stmt = select(NewsArticle.mind_id).join(ArticleSplit).where(ArticleSplit.split == candidate_split)
        result = await session.execute(stmt)
        valid_mids = set(result.scalars().all())
        
        for mid, row_idx in ir_store.mind_id_to_row.items():
            if mid not in valid_mids:
                mask[row_idx] = False
                
    return mask
