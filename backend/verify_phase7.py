"""
Phase 7 Verification Script
Applies MMR Diversity re-ranking to candidate pools for real MIND users.
"""

import asyncio
from app.database.connection import AsyncSessionLocal
from app.services.retrieval.user_vector import get_user_vector
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.mmr_ranker import mmr_rerank
import time
import numpy as np

async def verify():
    test_users = ["U536528", "U13740", "U91836"]
    pool_size = 50
    final_k = 10
    mmr_lambda = 0.7
    
    async with AsyncSessionLocal() as session:
        print("=== PHASE 7 FINAL RESULTS ===\n")
        
        for uid in test_users:
            print(f"User: {uid}")
            t0 = time.perf_counter()
            user_vec, stats = await get_user_vector(session, uid, alpha=0.8)
            
            if stats["status"] == "INSUFFICIENT":
                print("Profile status: INSUFFICIENT\n")
                continue
                
            # 1. Candidate Retrieval
            mask = await get_candidate_mask(session, "all", stats["clicked_articles"])
            candidates = await rank_candidates(session, user_vec, mask, top_n=pool_size)
            
            # 2. MMR Re-ranking
            mids = [c["mind_id"] for c in candidates]
            scores = [c["similarity"] for c in candidates]
            
            # Record top 3 before MMR for comparison
            print(f"\nBefore MMR (Top 3 of {pool_size} candidates):")
            for i, c in enumerate(candidates[:3]):
                print(f"  {i+1}. [{c['mind_id']}] {c['title']} ({c['category']}/{c['subcategory']}) - Sim: {c['similarity']:.4f}")
                
            selected_indices = mmr_rerank(mids, scores, final_k, mmr_lambda)
            mmr_candidates = [candidates[i] for i in selected_indices]
            
            t1 = time.perf_counter()
            
            print(f"\nAfter MMR (Top 10):")
            for i, c in enumerate(mmr_candidates):
                # Also lookup where it was in original ranking
                orig_rank = mids.index(c["mind_id"]) + 1
                print(f"  {i+1}. [{c['mind_id']}] {c['title']} ({c['category']}/{c['subcategory']}) - Sim: {c['similarity']:.4f} (Orig Rank: {orig_rank})")
                
            print(f"\nLatency: {(t1 - t0)*1000:.2f} ms")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(verify())
