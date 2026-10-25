"""
Recommendation Baselines — Phase 11
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

Implements:
- BASELINE A: Training-period Popularity (strictly no future clicks)
- BASELINE B: Long-Term Profile Only (Phase 6 retrieval, alpha = 1.0)
- BASELINE C: Recent Profile Only (Phase 6 retrieval, alpha = 0.0)
- SYSTEM D: Full Profile Cosine Similarity without MMR
- SYSTEM E: Full System (Long-Term + Recent + Drift Adjustment + Cosine + MMR)
"""

import os
import sqlite3
from collections import Counter
from typing import List, Dict, Any, Set, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.index_loader import ir_store
from app.services.retrieval.user_vector import _compute_article_weights, _build_weighted_vector
from app.services.interest_modeling.profile_builder import LONG_TERM_DECAY, RECENT_DECAY, RECENT_WINDOW_SIZE
from app.services.interest_modeling.drift_detector import compute_jsd
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.mmr_ranker import mmr_rerank

class TrainingPopularityRanker:
    """
    Computes global article popularity based strictly on training period clicks.
    Provides recommendation by taking the most popular unclicked eligible articles.
    """

    def __init__(self, db_path: str, max_logs_to_scan: int = 100000):
        self.db_path = os.path.abspath(db_path)
        self.max_logs_to_scan = max_logs_to_scan
        self.article_click_counts: Counter = Counter()
        self.ranked_train_articles: List[str] = []
        self._is_initialized = False

    def initialize(self):
        if self._is_initialized:
            return

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        print(f"Initializing Training Popularity baseline from train behaviors (scanning up to {self.max_logs_to_scan} logs)...")
        cur.execute(f"SELECT impressions_raw FROM behavior_logs WHERE split='train' LIMIT {self.max_logs_to_scan}")
        
        counter = Counter()
        for row in cur.fetchall():
            if row[0]:
                for token in row[0].split():
                    if token.endswith("-1"):
                        aid = token.split("-")[0]
                        counter[aid] += 1

        con.close()
        self.article_click_counts = counter
        self.ranked_train_articles = [aid for aid, cnt in counter.most_common()]
        self._is_initialized = True
        print(f"Training Popularity initialized with {len(self.ranked_train_articles)} distinct clicked train articles.")

    async def recommend(
        self,
        session: AsyncSession,
        train_clicked_aids: List[str],
        candidate_split: str = "dev",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommends the top-K most popular articles from the candidate pool,
        excluding any articles already clicked by the user during training.
        """
        self.initialize()
        ir_store.load()

        # Get candidate mask for candidate_split
        mask = await get_candidate_mask(session, candidate_split=candidate_split, clicked_mind_ids=train_clicked_aids)
        clicked_set = set(train_clicked_aids)

        # Filter the globally ranked train articles by eligibility
        selected_mids = []
        for aid in self.ranked_train_articles:
            if aid in clicked_set:
                continue
            row_idx = ir_store.mind_id_to_row.get(aid)
            if row_idx is not None and mask[row_idx]:
                selected_mids.append(aid)
                if len(selected_mids) >= top_k:
                    break

        # Fallback: if fewer than top_k, fill from candidate pool
        if len(selected_mids) < top_k:
            valid_indices = np.where(mask)[0]
            for idx in valid_indices:
                aid = ir_store.row_to_mind_id.get(idx)
                if aid and aid not in clicked_set and aid not in selected_mids:
                    selected_mids.append(aid)
                    if len(selected_mids) >= top_k:
                        break

        # Fetch metadata
        if not selected_mids:
            return []

        from app.models.news_article import NewsArticle
        from sqlalchemy import select
        stmt = select(NewsArticle).where(NewsArticle.mind_id.in_(selected_mids))
        res = await session.execute(stmt)
        articles_map = {a.mind_id: a for a in res.scalars().all()}

        results = []
        for rank, mid in enumerate(selected_mids, 1):
            art = articles_map.get(mid)
            results.append({
                "mind_id": mid,
                "title": art.title if art else "",
                "category": art.category if art else "",
                "subcategory": art.subcategory if art else "",
                "similarity": float(self.article_click_counts.get(mid, 0)),
                "rank": rank
            })
        return results


def build_user_vector_from_sequence(
    train_sequence: List[Tuple[str, Optional[Any]]],
    mode: str = "full",
    alpha: float = 0.8,
    apply_drift: bool = True
) -> Tuple[Optional[sp.csr_matrix], Optional[np.ndarray], Dict[str, Any]]:
    """
    Constructs the user's vector representation strictly from their training click sequence.
    Also constructs the user's topic vector in the same manner.
    """
    ir_store.load()

    if not train_sequence:
        return None, None, {"status": "INSUFFICIENT"}

    # Compute weights
    lt_weights = _compute_article_weights(train_sequence, LONG_TERM_DECAY)
    rt_sequence = train_sequence[-RECENT_WINDOW_SIZE:]
    rt_weights = _compute_article_weights(rt_sequence, RECENT_DECAY)

    # TF-IDF vectors
    lt_vec = _build_weighted_vector(lt_weights)
    rt_vec = _build_weighted_vector(rt_weights)

    # Topic vectors
    def _build_weighted_topic_vector(weights: List[Tuple[str, float]]) -> Optional[np.ndarray]:
        if not weights:
            return None
        t_vec = None
        total_w = 0.0
        for aid, w in weights:
            vec = ir_store.get_article_topic_vector(aid)
            if vec is not None:
                if t_vec is None:
                    t_vec = vec * w
                else:
                    t_vec += vec * w
                total_w += w
        if t_vec is None or total_w == 0:
            return None
        t_vec = t_vec / total_w
        return normalize(t_vec.reshape(1, -1), norm="l2", axis=1).flatten()

    lt_topic = _build_weighted_topic_vector(lt_weights)
    rt_topic = _build_weighted_topic_vector(rt_weights)

    if lt_vec is None and rt_vec is None:
        return None, None, {"status": "INSUFFICIENT"}

    if mode == "long_term":
        vec = lt_vec if lt_vec is not None else rt_vec
        vec = normalize(vec, norm="l2", axis=1)
        topic = lt_topic if lt_topic is not None else rt_topic
        if topic is not None:
            topic = normalize(topic.reshape(1, -1), norm="l2", axis=1).flatten()
        return vec, topic, {"mode": "long_term", "lt_weight": 1.0, "rt_weight": 0.0}

    elif mode == "recent":
        vec = rt_vec if rt_vec is not None else lt_vec
        vec = normalize(vec, norm="l2", axis=1)
        topic = rt_topic if rt_topic is not None else lt_topic
        if topic is not None:
            topic = normalize(topic.reshape(1, -1), norm="l2", axis=1).flatten()
        return vec, topic, {"mode": "recent", "lt_weight": 0.0, "rt_weight": 1.0}

    # Mode: 'full' or 'full_nodrift'
    base_recent_weight = 1.0 - alpha
    effective_recent_weight = base_recent_weight

    if apply_drift and lt_vec is not None and rt_vec is not None:
        cos_sim = float((lt_vec @ rt_vec.T).toarray()[0, 0])
        drift_proxy = max(0.0, min(1.0, 1.0 - cos_sim))
        drift_adjustment = min(drift_proxy, 0.5)
        effective_recent_weight = min(1.0, base_recent_weight + drift_adjustment)

    effective_lt_weight = 1.0 - effective_recent_weight

    if lt_vec is not None and rt_vec is not None:
        combined_vec = (lt_vec * effective_lt_weight) + (rt_vec * effective_recent_weight)
    elif lt_vec is not None:
        combined_vec = lt_vec
    else:
        combined_vec = rt_vec

    if lt_topic is not None and rt_topic is not None:
        combined_topic = (lt_topic * effective_lt_weight) + (rt_topic * effective_recent_weight)
    elif lt_topic is not None:
        combined_topic = lt_topic
    else:
        combined_topic = rt_topic

    combined_vec = normalize(combined_vec, norm="l2", axis=1)
    if combined_topic is not None:
        combined_topic = normalize(combined_topic.reshape(1, -1), norm="l2", axis=1).flatten()

    return combined_vec, combined_topic, {
        "mode": mode,
        "lt_weight": round(effective_lt_weight, 4),
        "rt_weight": round(effective_recent_weight, 4)
    }

