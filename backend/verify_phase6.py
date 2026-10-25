"""
Phase 6 Verification Script
Computes User TF-IDF Vector and ranks candidates for real MIND users.
"""

import asyncio
from app.database.connection import AsyncSessionLocal
from app.services.retrieval.user_vector import get_user_vector
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
import time
import numpy as np

async def verify():
    test_users = ["U536528", "U13740", "U91836"]
    
    async with AsyncSessionLocal() as session:
        print("=== PHASE 6 FINAL RESULTS ===\n")
        
        for uid in test_users:
            print(f"User: {uid}")
            t0 = time.perf_counter()
            user_vec, stats = await get_user_vector(session, uid, alpha=0.8)
            
            if stats["status"] == "INSUFFICIENT":
                print("Profile status: INSUFFICIENT")
                print("Candidates: 0")
                print("Latency: 0 ms\n")
                continue
                
            mask = await get_candidate_mask(session, "all", stats["clicked_articles"])
            candidates = await rank_candidates(session, user_vec, mask, top_n=10)
            t1 = time.perf_counter()
            
            print(f"User vector:")
            print(f"  Dimension: {user_vec.shape}")
            print(f"  Normalization: L2 (Norm = {np.linalg.norm(user_vec.toarray()):.4f})")
            print(f"  Profile Weights: LT={stats['long_term_weight']}, RT={stats['recent_weight']}")
            
            print(f"\nCandidate corpus:")
            print(f"  Split: all")
            print(f"  Available candidates (after exclusions): {np.count_nonzero(mask)}")
            
            print(f"\nCandidates returned: {len(candidates)}")
            if candidates:
                print(f"Top similarity: {candidates[0]['similarity']:.4f}")
                print("\nTop 3 Candidate Articles:")
                for i, c in enumerate(candidates[:3]):
                    print(f"  {i+1}. [{c['mind_id']}] {c['title']} ({c['category']}/{c['subcategory']}) - Sim: {c['similarity']:.4f}")
                    
            print(f"\nLatency: {(t1 - t0)*1000:.2f} ms")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(verify())
