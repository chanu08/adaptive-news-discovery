"""
Recommendation Orchestration Service — Phase 10

Orchestrates the entire recommendation pipeline by tying together
previously validated modules:
- Phase 3/9 User Interest Profiles
- Phase 4/9 Interest Drift
- Phase 6 Candidate Retrieval & Cosine Similarity
- Phase 7 MMR Diversity Re-ranking
- Phase 8 Explanation Generator

Does NOT re-implement mathematical algorithms. It only calls them.
"""

import time
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.user_vector import get_user_vector
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.mmr_ranker import mmr_rerank
from app.services.explanation.explanation_generator import generate_explanation
from app.services.interest_modeling.profile_store import get_all_user_profiles
from app.services.interest_modeling.drift_service import get_drift_report

class RecommendationPipelineError(Exception):
    """Raised when the pipeline fails to generate recommendations."""

async def generate_recommendations(
    session: AsyncSession,
    user_mind_id: str,
    candidate_pool_size: int = 50,
    final_k: int = 10,
    mmr_lambda: float = 0.7,
    apply_mmr: bool = True,
    include_explanations: bool = True,
    split: str = "all",
    alpha: float = 0.8
) -> Dict[str, Any]:
    """
    Executes the unified recommendation pipeline.
    """
    t0 = time.perf_counter()

    # 1 & 2 & 3. Load user vector (which internally fetches profiles)
    user_vec, stats = await get_user_vector(session, user_mind_id, alpha)
    if stats["status"] == "INSUFFICIENT" or user_vec is None:
        raise RecommendationPipelineError("User profile has insufficient data.")

    # Fetch raw profiles and drift for explanation/diagnostic purposes
    profiles = await get_all_user_profiles(session, user_mind_id)
    drift = await get_drift_report(session, user_mind_id)

    # 4. Get candidate pool mask (exclude clicked)
    mask = await get_candidate_mask(session, split, stats["clicked_articles"])

    # 5. Compute relevance ranking
    t_retrieval_0 = time.perf_counter()
    candidates = await rank_candidates(session, user_vec, mask, candidate_pool_size)
    t_retrieval = time.perf_counter() - t_retrieval_0

    if not candidates:
        raise RecommendationPipelineError("No candidates available for retrieval.")

    # 6. Apply MMR Diversity Re-ranking
    t_mmr_0 = time.perf_counter()
    if apply_mmr:
        mids = [c["mind_id"] for c in candidates]
        scores = [c["similarity"] for c in candidates]
        selected_indices = mmr_rerank(mids, scores, final_k, mmr_lambda)
        final_candidates = [candidates[i] for i in selected_indices]
    else:
        final_candidates = candidates[:final_k]
    t_mmr = time.perf_counter() - t_mmr_0

    # 7. Generate Explanations
    t_exp_0 = time.perf_counter()
    if include_explanations:
        for i, c in enumerate(final_candidates):
            c["orig_rank"] = candidates.index(c) + 1
            c["mmr_rank"] = i + 1 if apply_mmr else c["orig_rank"]
            c["explanation"] = generate_explanation(c, profiles, drift, apply_mmr)
    else:
        for i, c in enumerate(final_candidates):
            c["orig_rank"] = candidates.index(c) + 1
            c["mmr_rank"] = i + 1 if apply_mmr else c["orig_rank"]
    t_exp = time.perf_counter() - t_exp_0

    t1 = time.perf_counter()
    total_latency = t1 - t0

    # Map to final output format
    recommendations = []
    for c in final_candidates:
        rec = {
            "mind_id": c["mind_id"],
            "title": c["title"],
            "category": c["category"],
            "subcategory": c["subcategory"],
            "relevance_score": c["similarity"],
            "mmr_score": c.get("mmr_score", c["similarity"]),  # fallback if MMR off
            "rank": c["mmr_rank"]
        }
        if include_explanations and "explanation" in c:
            rec["explanation"] = c["explanation"]
        recommendations.append(rec)

    response = {
        "user_id": user_mind_id,
        "pipeline": {
            "candidate_pool_size": candidate_pool_size,
            "final_k": final_k,
            "mmr_lambda": mmr_lambda,
            "mmr_enabled": apply_mmr,
            "explanations_enabled": include_explanations,
            "split": split
        },
        "profile": {
            "long_term_available": profiles["long_term"] is not None,
            "recent_available": profiles["recent"] is not None
        },
        "drift": {
            "jsd": drift.get("jsd_score") if drift else 0.0,
            "evidence_level": drift.get("evidence_level") if drift else "NONE"
        },
        "recommendations": recommendations,
        "diagnostics": {
            "candidate_count": len(candidates),
            "recommendation_count": len(recommendations),
            "clicked_items_excluded": True,
            "mmr_applied": apply_mmr,
            "explanations_generated": include_explanations,
            "latency_ms": round(total_latency * 1000, 2),
            "retrieval_ms": round(t_retrieval * 1000, 2),
            "mmr_ms": round(t_mmr * 1000, 2),
            "explanation_ms": round(t_exp * 1000, 2)
        }
    }
    
    return response
