"""
Similarity Ranker — Phase 6
Ranks candidates based on cosine similarity to the user vector.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_article import NewsArticle
from app.models.article_feature import ArticleTextFeature
from app.services.retrieval.index_loader import ir_store

async def rank_candidates(
    session: AsyncSession, 
    user_vector: sp.csr_matrix, 
    candidate_mask: np.ndarray, 
    top_n: int = 100, 
    retrieval_method: str = "tfidf", 
    hybrid_alpha: float = 0.7, 
    user_topic_vector: np.ndarray = None,
    ranking_method: str = "content",
    popularity_beta: float = 0.7
) -> List[dict]:
    """
    Computes similarity, filters by mask, applies popularity bias, and returns top-N ranked candidates.
    Supports ranking_method: 'content', 'popularity', 'content_popularity'.
    """
    ir_store.load()
    from app.services.retrieval.popularity_loader import popularity_store
    
    # 1. Base Content Score
    if retrieval_method == "tfidf":
        content_scores = ir_store.compute_similarity(user_vector)
    elif retrieval_method == "topic":
        if user_topic_vector is None:
            raise ValueError("user_topic_vector must be provided for topic retrieval")
        content_scores = ir_store.compute_topic_similarity(user_topic_vector)
    elif retrieval_method == "hybrid":
        if user_topic_vector is None:
            raise ValueError("user_topic_vector must be provided for hybrid retrieval")
        tfidf_scores = ir_store.compute_similarity(user_vector)
        topic_scores = ir_store.compute_topic_similarity(user_topic_vector)
        content_scores = (hybrid_alpha * tfidf_scores) + ((1.0 - hybrid_alpha) * topic_scores)
    else:
        raise ValueError(f"Unknown retrieval_method: {retrieval_method}")
        
    # 2. Popularity Blending
    if ranking_method == "content":
        sim_scores = content_scores
    elif ranking_method == "popularity":
        popularity_store.load()
        sim_scores = popularity_store.get_popularity_array().copy()
    elif ranking_method == "content_popularity":
        popularity_store.load()
        pop_scores = popularity_store.get_popularity_array()
        sim_scores = (popularity_beta * content_scores) + ((1.0 - popularity_beta) * pop_scores)
    else:
        raise ValueError(f"Unknown ranking_method: {ranking_method}")
    
    # Apply mask
    sim_scores[~candidate_mask] = -1.0
    
    # Get top N indices
    effective_n = min(top_n, np.count_nonzero(candidate_mask))
    if effective_n == 0:
        return []
        
    top_indices = np.argpartition(sim_scores, -effective_n)[-effective_n:]
    
    # Sort the top indices in descending order
    top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
    
    # Get mind_ids
    candidate_mids = [ir_store.row_to_mind_id[idx] for idx in top_indices]
    
    # Fetch metadata
    stmt = select(NewsArticle, ArticleTextFeature).outerjoin(
        ArticleTextFeature, NewsArticle.mind_id == ArticleTextFeature.mind_id
    ).where(NewsArticle.mind_id.in_(candidate_mids))
    
    result = await session.execute(stmt)
    rows = result.all()
    
    metadata_map = {}
    for article, feature in rows:
        metadata_map[article.mind_id] = {
            "mind_id": article.mind_id,
            "title": article.title,
            "category": article.category,
            "subcategory": article.subcategory,
            "dominant_topic": feature.dominant_topic if feature else None
        }
        
    ranked_candidates = []
    for idx in top_indices:
        mid = ir_store.row_to_mind_id[idx]
        score = float(sim_scores[idx])
        meta = metadata_map.get(mid, {})
        meta["similarity"] = score
        
        # For hybrid/topic debugging in inspection
        if retrieval_method == "hybrid":
            meta["tfidf_similarity"] = float(tfidf_scores[idx])
            meta["topic_similarity"] = float(topic_scores[idx])
        elif retrieval_method == "topic":
            meta["topic_similarity"] = float(sim_scores[idx])
            
        ranked_candidates.append(meta)
        
    return ranked_candidates
