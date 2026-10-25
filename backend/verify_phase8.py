"""
Phase 8 Verification Script
Tests Explanation Generation for Recommended Articles.
"""

import asyncio
import json
from app.database.connection import AsyncSessionLocal
from app.services.retrieval.user_vector import get_user_vector
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.mmr_ranker import mmr_rerank
from app.services.interest_modeling.profile_store import get_all_user_profiles
from app.services.interest_modeling.drift_service import get_drift_report
from app.services.explanation.explanation_generator import generate_explanation
import time

async def verify():
    test_users = ["U536528", "U13740", "U91836"]
    pool_size = 50
    final_k = 10
    mmr_lambda = 0.7
    
    async with AsyncSessionLocal() as session:
        print("=== PHASE 8 FINAL RESULTS ===\n")
        
        for uid in test_users:
            print(f"User: {uid}")
            t0 = time.perf_counter()
            
            user_vec, stats = await get_user_vector(session, uid, alpha=0.8)
            
            if stats["status"] == "INSUFFICIENT":
                print("Profile status: INSUFFICIENT\n")
                continue
                
            mask = await get_candidate_mask(session, "all", stats["clicked_articles"])
            candidates = await rank_candidates(session, user_vec, mask, top_n=pool_size)
            
            for i, c in enumerate(candidates):
                c["orig_rank"] = i + 1
            
            mids = [c["mind_id"] for c in candidates]
            scores = [c["similarity"] for c in candidates]
            
            selected_indices = mmr_rerank(mids, scores, final_k, mmr_lambda)
            mmr_candidates = [candidates[i] for i in selected_indices]
            
            profiles = await get_all_user_profiles(session, uid)
            drift = await get_drift_report(session, uid)
            
            for i, c in enumerate(mmr_candidates):
                c["mmr_rank"] = i + 1
                c["explanation"] = generate_explanation(c, profiles, drift, True)
                
            t1 = time.perf_counter()
            
            print(f"Candidates returned with explanations: {len(mmr_candidates)}")
            print("\nTop 3 Candidate Explanations:")
            for i, c in enumerate(mmr_candidates[:3]):
                print(f"  {i+1}. [{c['mind_id']}] {c['title']} ({c['category']}/{c['subcategory']})")
                print(f"     Sim: {c['similarity']:.4f} | Orig Rank: {c['orig_rank']} | MMR Rank: {c['mmr_rank']}")
                print(f"     Why? {c['explanation']['primary_reason']}")
                if c['explanation']['supporting_reasons']:
                    print(f"     Supporting: {', '.join(c['explanation']['supporting_reasons'])}")
                print()
                
            print(f"Latency: {(t1 - t0)*1000:.2f} ms")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(verify())
