"""
Users routes — Phase 2

GET /api/users           — list MIND users (paginated)
GET /api/users/{id}      — single user by internal DB id
GET /api/users/mind/{mind_user_id} — lookup by original MIND UserID
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

from app.services.interest_modeling.profile_store import get_user_profile, get_all_user_profiles
from app.services.interest_modeling.profile_builder import build_user_profiles

from app.database.connection import get_db
from app.models.mind_user import MindUser

router = APIRouter()


class MindUserResponse(BaseModel):
    id: int
    mind_user_id: str
    total_impressions: int
    total_clicks: int
    ingested_at: datetime

    class Config:
        from_attributes = True


@router.get("/", response_model=list[MindUserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return paginated MIND users."""
    result = await db.execute(
        select(MindUser).offset(skip).limit(limit).order_by(MindUser.id)
    )
    return result.scalars().all()


@router.get("/mind/{mind_user_id}", response_model=MindUserResponse)
async def get_user_by_mind_id(
    mind_user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return a MIND user by their original UserID (e.g. U13740)."""
    result = await db.execute(
        select(MindUser).where(MindUser.mind_user_id == mind_user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with MIND ID '{mind_user_id}' not found.",
        )
    return user


@router.get("/{user_id}", response_model=MindUserResponse)
async def get_user_by_id(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a MIND user by internal database ID."""
    result = await db.execute(
        select(MindUser).where(MindUser.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )
    return user

@router.post("/mind/{mind_user_id}/profile/build", status_code=status.HTTP_202_ACCEPTED)
async def trigger_profile_build(mind_user_id: str, db: AsyncSession = Depends(get_db)):
    """Manually triggers building the user's profile from interaction logs."""
    success = await build_user_profiles(db, mind_user_id)
    await db.commit()
    if not success:
        return {"message": "User has no clicks, empty profile created.", "status": "cold_user"}
    return {"message": "Profile built successfully.", "status": "success"}

@router.get("/mind/{mind_user_id}/profile", response_model=Dict[str, Any])
async def get_user_full_profile(mind_user_id: str, db: AsyncSession = Depends(get_db)):
    """Returns both long-term and recent profiles."""
    profiles = await get_all_user_profiles(db, mind_user_id)
    if not profiles["long_term"] and not profiles["recent"]:
        # Build dynamically if empty (lazy loading)
        success = await build_user_profiles(db, mind_user_id)
        await db.commit()
        profiles = await get_all_user_profiles(db, mind_user_id)
    if not profiles["long_term"] and not profiles["recent"]:
         raise HTTPException(status_code=404, detail="User profile not found or empty.")
    return profiles

@router.get("/mind/{mind_user_id}/profile/long-term", response_model=Dict[str, Any])
async def get_user_long_term_profile(mind_user_id: str, db: AsyncSession = Depends(get_db)):
    """Returns the long-term profile."""
    profile = await get_user_profile(db, mind_user_id, "LONG_TERM")
    if not profile:
        raise HTTPException(status_code=404, detail="Long-term profile not found.")
    return profile

@router.get("/mind/{mind_user_id}/profile/recent", response_model=Dict[str, Any])
async def get_user_recent_profile(mind_user_id: str, db: AsyncSession = Depends(get_db)):
    """Returns the recent profile."""
    profile = await get_user_profile(db, mind_user_id, "RECENT")
    if not profile:
        raise HTTPException(status_code=404, detail="Recent profile not found.")
    return profile

@router.get("/mind/{mind_user_id}/drift", response_model=Dict[str, Any])
async def get_user_interest_drift(mind_user_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves the Interest Drift report for a MIND user."""
    from app.services.interest_modeling.drift_service import get_drift_report
    
    drift = await get_drift_report(db, mind_user_id)
    if not drift:
        raise HTTPException(status_code=404, detail="Drift data not available. Ensure user has a profile generated.")
        
    return drift

@router.get("/mind/{mind_user_id}/candidates", response_model=Dict[str, Any])
async def get_user_candidates(
    mind_user_id: str, 
    top_n: int = Query(50, ge=1, le=500, description="Candidate pool size"),
    candidate_split: str = Query("all", description="train, dev, test, all"),
    alpha: float = Query(0.8, ge=0.0, le=1.0),
    apply_mmr: bool = Query(False, description="Apply MMR diversity re-ranking"),
    final_k: int = Query(10, ge=1, le=100, description="Final number of candidates after MMR"),
    mmr_lambda: float = Query(0.7, ge=0.0, le=1.0, description="MMR lambda (relevance weight)"),
    include_explanations: bool = Query(False, description="Include Phase 8 explanations"),
    db: AsyncSession = Depends(get_db)
):
    """Returns candidate articles via TF-IDF cosine similarity, MMR re-ranking, and optional explanations."""
    from app.services.retrieval.user_vector import get_user_vector
    from app.services.retrieval.candidate_retriever import get_candidate_mask
    from app.services.retrieval.similarity_ranker import rank_candidates
    from app.services.retrieval.mmr_ranker import mmr_rerank
    from app.services.explanation.explanation_generator import generate_explanation
    import time
    
    t0 = time.perf_counter()
    user_vec, stats = await get_user_vector(db, mind_user_id, alpha)
    
    if stats["status"] == "INSUFFICIENT" or user_vec is None:
        return {
            "user_id": mind_user_id,
            "candidate_split": candidate_split,
            "profile_status": "INSUFFICIENT",
            "profile_weights": {},
            "candidates": []
        }
        
    mask = await get_candidate_mask(db, candidate_split, stats["clicked_articles"])
    candidates = await rank_candidates(db, user_vec, mask, top_n)
    
    # Store original rank for explanation
    for i, c in enumerate(candidates):
        c["orig_rank"] = i + 1
    
    # Apply MMR Re-ranking if requested
    if apply_mmr and candidates:
        mids = [c["mind_id"] for c in candidates]
        scores = [c["similarity"] for c in candidates]
        selected_indices = mmr_rerank(mids, scores, final_k, mmr_lambda)
        candidates = [candidates[i] for i in selected_indices]
        
    # Apply Explanations if requested
    if include_explanations and candidates:
        from app.services.interest_modeling.drift_service import get_drift_report
        profiles = await get_all_user_profiles(db, mind_user_id)
        drift = await get_drift_report(db, mind_user_id)
        
        for i, c in enumerate(candidates):
            c["mmr_rank"] = i + 1 if apply_mmr else c["orig_rank"]
            c["explanation"] = generate_explanation(c, profiles, drift, apply_mmr)
        
    t1 = time.perf_counter()
    
    return {
        "user_id": mind_user_id,
        "candidate_split": candidate_split,
        "profile_status": "OK",
        "profile_weights": {
            "long_term": stats["long_term_weight"],
            "recent": stats["recent_weight"],
            "base_alpha": stats["base_alpha"]
        },
        "latency_ms": round((t1 - t0) * 1000, 2),
        "mmr_applied": apply_mmr,
        "candidates": candidates
    }

@router.get("/mind/{mind_user_id}/recommendations", response_model=Dict[str, Any])
async def get_user_recommendations(
    mind_user_id: str,
    final_k: int = Query(10, ge=1, le=100, description="Final number of recommendations"),
    candidate_pool_size: int = Query(50, ge=1, le=500, description="Candidate pool size"),
    mmr_lambda: float = Query(0.7, ge=0.0, le=1.0, description="MMR lambda (relevance weight)"),
    apply_mmr: bool = Query(True, description="Apply MMR diversity re-ranking"),
    include_explanations: bool = Query(True, description="Include explanations"),
    split: str = Query("all", description="train, dev, test, all"),
    db: AsyncSession = Depends(get_db)
):
    """
    Unified endpoint for Phase 10 end-to-end recommendation pipeline.
    Orchestrates profile loading, drift calculation, retrieval, MMR, and explanation generation.
    """
    from app.services.recommendation.recommendation_service import generate_recommendations, RecommendationPipelineError
    
    try:
        response = await generate_recommendations(
            session=db,
            user_mind_id=mind_user_id,
            candidate_pool_size=candidate_pool_size,
            final_k=final_k,
            mmr_lambda=mmr_lambda,
            apply_mmr=apply_mmr,
            include_explanations=include_explanations,
            split=split
        )
        return response
    except RecommendationPipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
