"""
Evaluation Protocols — Phase 11
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

Implements:
1. CatalogCandidateRetrievalProtocol (Section 7):
   Candidate retrieval from the full candidate split catalog (e.g. 72,023 dev articles),
   evaluating recommendation discovery against held-out future clicks.
2. ImpressionLevelRerankingProtocol (Section 8):
   Re-ranking within the impressed articles of actual behavior sessions.
3. Zero-division and edge-case handling (Section 10).
"""

import time
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.index_loader import ir_store
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.mmr_ranker import mmr_rerank
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    category_diversity,
    subcategory_diversity,
)

class CatalogCandidateRetrievalProtocol:
    """
    Evaluates recommendations generated from the full candidate catalog (e.g., dev split).
    Measures Precision@K, Recall@K, NDCG@K, HitRate@K, ILD@10, and Category Diversity.
    """

    def __init__(self, candidate_split: str = "dev", candidate_pool_size: int = 50, final_k: int = 10):
        self.candidate_split = candidate_split
        self.candidate_pool_size = candidate_pool_size
        self.final_k = final_k

    async def evaluate_user(
        self,
        session: AsyncSession,
        user_data: Dict[str, Any],
        user_vector: Any,
        user_topic_vector: Any = None,
        apply_mmr: bool = True,
        mmr_lambda: float = 0.7,
        retrieval_method: str = "tfidf",
        hybrid_alpha: float = 0.7,
        ranking_method: str = "content",
        popularity_beta: float = 0.7
    ) -> Tuple[Optional[Dict[str, float]], Optional[str], List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs candidate retrieval and ranking for a single user, returns:
        (metrics_dict, skip_reason, final_recommendations, timing_dict)
        """
        timings = {}
        ground_truth = user_data["dev_ground_truth"]
        train_clicked_aids = user_data["train_clicked_aids"]

        # Check edge cases
        if not ground_truth:
            return None, "NO_HELDOUT_CLICKS", [], timings

        if user_vector is None and retrieval_method in ["tfidf", "hybrid"]:
            return None, "EMPTY_USER_VECTOR", [], timings
            
        if user_topic_vector is None and retrieval_method in ["topic", "hybrid"]:
            return None, "EMPTY_USER_TOPIC_VECTOR", [], timings

        ir_store.load()

        # Step 1: Candidate Mask
        t0 = time.perf_counter()
        mask = await get_candidate_mask(
            session,
            candidate_split=self.candidate_split,
            clicked_mind_ids=train_clicked_aids
        )
        if not np.any(mask):
            return None, "EMPTY_CANDIDATE_POOL", [], timings

        # Step 2: Relevance Ranking (Cosine Retrieval)
        t_retrieval_0 = time.perf_counter()
        candidates = await rank_candidates(
            session,
            user_vector,
            mask,
            top_n=self.candidate_pool_size,
            retrieval_method=retrieval_method,
            hybrid_alpha=hybrid_alpha,
            user_topic_vector=user_topic_vector,
            ranking_method=ranking_method,
            popularity_beta=popularity_beta
        )
        timings["retrieval_sec"] = time.perf_counter() - t_retrieval_0

        if not candidates:
            return None, "NO_CANDIDATES_RETRIEVED", [], timings

        # Step 3: MMR Re-ranking
        t_mmr_0 = time.perf_counter()
        if apply_mmr:
            mids = [c["mind_id"] for c in candidates]
            scores = [c["similarity"] for c in candidates]
            selected_indices = mmr_rerank(mids, scores, self.final_k, mmr_lambda)
            final_recs = [candidates[i] for i in selected_indices]
        else:
            final_recs = candidates[:self.final_k]
        timings["mmr_sec"] = time.perf_counter() - t_mmr_0
        timings["total_sec"] = time.perf_counter() - t0

        rec_mids = [r["mind_id"] for r in final_recs]

        # Step 4: Metric Computations
        user_metrics = {
            "p_at_5": precision_at_k(rec_mids, ground_truth, 5),
            "r_at_5": recall_at_k(rec_mids, ground_truth, 5),
            "ndcg_at_5": ndcg_at_k(rec_mids, ground_truth, 5),
            "hit_rate_at_5": hit_rate_at_k(rec_mids, ground_truth, 5),
            "p_at_10": precision_at_k(rec_mids, ground_truth, 10),
            "r_at_10": recall_at_k(rec_mids, ground_truth, 10),
            "ndcg_at_10": ndcg_at_k(rec_mids, ground_truth, 10),
            "hit_rate_at_10": hit_rate_at_k(rec_mids, ground_truth, 10),
            "ild_at_10": intra_list_diversity(rec_mids, ir_store, 10),
            "category_diversity_at_10": float(category_diversity(final_recs, 10)),
            "subcategory_diversity_at_10": float(subcategory_diversity(final_recs, 10))
        }

        return user_metrics, None, final_recs, timings


class ImpressionLevelRerankingProtocol:
    """
    Evaluates ranking within the actual impressed sets of dev behavior logs (Section 8).
    Candidate set = articles impressed in that behavior session.
    Ground truth = articles clicked in that behavior session.
    """

    def __init__(self, final_k: int = 10):
        self.final_k = final_k

    def evaluate_user_impressions(
        self,
        user_data: Dict[str, Any],
        user_vector: Any,
        user_topic_vector: Any = None,
        apply_mmr: bool = False,
        mmr_lambda: float = 0.7,
        retrieval_method: str = "tfidf",
        hybrid_alpha: float = 0.7,
        ranking_method: str = "content",
        popularity_beta: float = 0.7
    ) -> Tuple[Optional[Dict[str, float]], Optional[str], int, List[List[Dict[str, Any]]]]:
        """
        Averages ranking metrics across all dev impression sessions for this user.
        Also returns the list of recommended items per session for coverage analysis.
        """
        sessions = user_data.get("dev_sessions", [])
        if not sessions:
            return None, "NO_DEV_SESSIONS", 0, []

        if user_vector is None and retrieval_method in ["tfidf", "hybrid"]:
            return None, "EMPTY_USER_VECTOR", 0, []

        if user_topic_vector is None and retrieval_method in ["topic", "hybrid"]:
            return None, "EMPTY_USER_TOPIC_VECTOR", 0, []

        ir_store.load()
        from app.services.retrieval.popularity_loader import popularity_store
        popularity_store.load()

        session_metrics_list = []
        valid_sessions = 0
        all_session_recs = []

        for sess in sessions:
            impressed_aids = sess["impressed_articles"]
            clicked_aids = sess["clicked_articles"]

            if not clicked_aids or not impressed_aids:
                continue

            # Compute similarity for each impressed article
            scored_candidates = []
            for aid in impressed_aids:
                # 1. Content Score
                c_score = 0.0
                row_idx = ir_store.mind_id_to_row.get(aid)
                if row_idx is not None:
                    if retrieval_method == "tfidf":
                        c_score = float((user_vector @ ir_store.tfidf_matrix[row_idx].T).toarray()[0, 0])
                    elif retrieval_method == "topic":
                        t_vec = ir_store.topic_matrix[row_idx]
                        c_score = float((user_topic_vector @ t_vec.T))
                    elif retrieval_method == "hybrid":
                        tfidf_score = float((user_vector @ ir_store.tfidf_matrix[row_idx].T).toarray()[0, 0])
                        t_vec = ir_store.topic_matrix[row_idx]
                        topic_score = float((user_topic_vector @ t_vec.T))
                        c_score = (hybrid_alpha * tfidf_score) + ((1.0 - hybrid_alpha) * topic_score)

                # 2. Popularity Score
                pop_score = popularity_store.get_article_popularity(aid)

                # 3. Final Score
                if ranking_method == "content":
                    sim = c_score
                elif ranking_method == "popularity":
                    sim = pop_score
                elif ranking_method == "content_popularity":
                    sim = (popularity_beta * c_score) + ((1.0 - popularity_beta) * pop_score)
                else:
                    sim = c_score

                scored_candidates.append({
                    "mind_id": aid, 
                    "similarity": sim, 
                    "content_sim": c_score,
                    "pop_score": pop_score
                })

            # Sort by similarity descending
            scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)

            if apply_mmr and len(scored_candidates) > 1:
                mids = [c["mind_id"] for c in scored_candidates]
                scores = [c["similarity"] for c in scored_candidates]
                selected_indices = mmr_rerank(mids, scores, min(self.final_k, len(mids)), mmr_lambda)
                rec_mids = [mids[i] for i in selected_indices]
                final_recs = [scored_candidates[i] for i in selected_indices]
            else:
                rec_mids = [c["mind_id"] for c in scored_candidates[:self.final_k]]
                final_recs = scored_candidates[:self.final_k]

            all_session_recs.append(final_recs)

            m = {
                "p_at_5": precision_at_k(rec_mids, clicked_aids, 5),
                "r_at_5": recall_at_k(rec_mids, clicked_aids, 5),
                "ndcg_at_5": ndcg_at_k(rec_mids, clicked_aids, 5),
                "hit_rate_at_5": hit_rate_at_k(rec_mids, clicked_aids, 5),
                "p_at_10": precision_at_k(rec_mids, clicked_aids, 10),
                "r_at_10": recall_at_k(rec_mids, clicked_aids, 10),
                "ndcg_at_10": ndcg_at_k(rec_mids, clicked_aids, 10),
                "hit_rate_at_10": hit_rate_at_k(rec_mids, clicked_aids, 10)
            }
            session_metrics_list.append(m)
            valid_sessions += 1

        if not session_metrics_list:
            return None, "NO_VALID_SESSIONS", 0, []

        # Average across sessions
        avg_metrics = {}
        for key in session_metrics_list[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in session_metrics_list]))

        return avg_metrics, None, valid_sessions, all_session_recs
