import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from evaluation.datasets import EvaluationCohort
from evaluation.baselines import build_user_vector_from_sequence
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.index_loader import ir_store
from evaluation.protocols import CatalogCandidateRetrievalProtocol

async def sanity_check():
    db_path = "./data/news_discovery.db"
    
    # 1. Load users
    cohort = EvaluationCohort(db_path, max_users=5, seed=42, split="dev")
    users = cohort.load_cohort()
    
    # Init IR Store
    ir_store.load()
    
    async for session in get_db():
        for i, user in enumerate(users):
            print(f"\n{'='*50}")
            print(f"USER {i+1} | ID: {user['user_id']}")
            print(f"{'='*50}")
            
            train_clicked = user["train_clicked_aids"]
            ground_truth = user["dev_ground_truth"]
            
            print(f"Train Clicks: {len(train_clicked)}")
            print(f"Future Dev Clicks (Ground Truth): {len(ground_truth)}")
            
            # 2. Vector
            vec, _ = build_user_vector_from_sequence(user["train_sequence"], mode="full")
            if vec is None:
                print("Could not build user vector.")
                continue
                
            # 3. Candidate Mask
            mask = await get_candidate_mask(session, candidate_split="dev", clicked_mind_ids=train_clicked)
            print(f"Candidates available in dev split: {mask.sum()}")
            
            # 4. Analyze Ground Truth vs Mask
            gt_in_mask = 0
            gt_not_in_mask = 0
            gt_not_in_index = 0
            
            gt_sims = []
            
            for gt_id in ground_truth:
                row_idx = ir_store.mind_id_to_row.get(gt_id)
                if row_idx is not None:
                    if mask[row_idx]:
                        gt_in_mask += 1
                    else:
                        gt_not_in_mask += 1
                        
                    # Calculate similarity
                    art_vec = ir_store.get_article_vector(gt_id)
                    sim = float((vec @ art_vec.T).toarray()[0, 0])
                    gt_sims.append((gt_id, sim))
                else:
                    gt_not_in_index += 1
                    
            print(f"\nGround Truth Analysis:")
            print(f"  - In candidate mask (eligible for retrieval): {gt_in_mask}")
            print(f"  - In index but NOT in candidate mask: {gt_not_in_mask}")
            print(f"  - Not in index at all: {gt_not_in_index}")
            
            if gt_sims:
                gt_sims.sort(key=lambda x: x[1], reverse=True)
                print(f"  - Top similarities of ground truth items: {[round(s, 4) for _, s in gt_sims[:5]]}")
                print(f"  - Avg similarity of ground truth items: {round(sum(s for _, s in gt_sims)/len(gt_sims), 4)}")
                
            # 5. Top Retrieved Candidates
            candidates = await rank_candidates(session, vec, mask, top_n=10)
            print(f"\nTop 5 Retrieved Candidates by System:")
            for rank, c in enumerate(candidates[:5]):
                is_gt = "YES" if c["mind_id"] in ground_truth else "NO"
                print(f"  {rank+1}. {c['mind_id']} | Sim: {round(c['similarity'], 4)} | Is Ground Truth: {is_gt}")

        break # Only need one db session

if __name__ == "__main__":
    asyncio.run(sanity_check())
