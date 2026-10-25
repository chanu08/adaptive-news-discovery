import asyncio
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from evaluation.datasets import EvaluationCohort
from evaluation.baselines import build_user_vector_from_sequence
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.index_loader import ir_store
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k

async def run_inspection(session):
    print("\n" + "="*50)
    print("SMALL REAL-USER INSPECTION")
    print("="*50)
    
    db_path = "./data/news_discovery.db"
    
    # We just need to load these specific users.
    # EvaluationCohort might not be able to fetch specific users directly easily without modifying it.
    # So we'll fetch them manually.
    target_users = ["U536528", "U13740", "U91836"]
    
    # Actually, EvaluationCohort loads all eligible users and random samples. We can just query db directly for their sequences.
    import sqlite3
    from datetime import datetime
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    ir_store.load()
    
    for uid in target_users:
        print(f"\n--- USER: {uid} ---")
        # Get train history
        cur.execute("SELECT history_raw FROM behavior_logs WHERE user_mind_id = ? AND split = 'train' ORDER BY id ASC LIMIT 1", (uid,))
        hist_row = cur.fetchone()
        hist_raw = hist_row[0] if hist_row and hist_row[0] else ""
        hist_clicks = [aid for aid in hist_raw.split() if aid]
        
        cur.execute("""
            SELECT i.article_mind_id, b.time, i.id
            FROM impression_items i
            JOIN behavior_logs b ON i.behavior_log_id = b.id
            WHERE b.user_mind_id = ? AND b.split = 'train' AND i.clicked = 1
            ORDER BY i.id ASC
        """, (uid,))
        train_imp_clicks = []
        for aid, dt_str, iid in cur.fetchall():
            dt = None
            if dt_str:
                try:
                    dt = datetime.strptime(dt_str, "%m/%d/%Y %I:%M:%S %p")
                except Exception:
                    pass
            train_imp_clicks.append((aid, dt, iid))
            
        train_imp_clicks.sort(key=lambda x: (x[1] if x[1] else datetime.min, x[2]))
        train_sequence = [(aid, None) for aid in hist_clicks] + [(aid, dt) for aid, dt, _ in train_imp_clicks]
        train_clicked_aids = [aid for aid, _ in train_sequence]
        
        vec, topic_vec, _ = build_user_vector_from_sequence(train_sequence, mode="full")
        mask = await get_candidate_mask(session, "dev", train_clicked_aids)
        
        tfidf_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="tfidf")
        topic_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="topic", user_topic_vector=topic_vec)
        hybrid_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="hybrid", hybrid_alpha=0.7, user_topic_vector=topic_vec)
        
        print("TF-IDF Top 10:")
        for r, c in enumerate(tfidf_cands):
            print(f"  {r+1}. {c['mind_id']} | {c['title'][:40]}... | {c['category']} | Sim: {c['similarity']:.4f}")
            
        print("\nTopic-Only Top 10:")
        for r, c in enumerate(topic_cands):
            print(f"  {r+1}. {c['mind_id']} | {c['title'][:40]}... | {c['category']} | Sim: {c['similarity']:.4f}")
            
        print("\nHybrid (alpha=0.7) Top 10:")
        for r, c in enumerate(hybrid_cands):
            print(f"  {r+1}. {c['mind_id']} | {c['title'][:40]}... | {c['category']} | Hybrid: {c['similarity']:.4f} (TFIDF: {c['tfidf_similarity']:.4f}, Topic: {c['topic_similarity']:.4f})")
            
    con.close()

async def run_experiment(session):
    print("\n" + "="*50)
    print("SMALL OFFLINE EXPERIMENT")
    print("="*50)
    
    db_path = "./data/news_discovery.db"
    cohort = EvaluationCohort(db_path, max_users=20, seed=42, split="dev")
    users = cohort.load_cohort()
    
    configs = [
        {"name": "TF-IDF Baseline", "method": "tfidf", "alpha": 1.0},
        {"name": "Topic-only", "method": "topic", "alpha": 0.0},
        {"name": "Hybrid alpha=0.8", "method": "hybrid", "alpha": 0.8},
        {"name": "Hybrid alpha=0.7", "method": "hybrid", "alpha": 0.7},
        {"name": "Hybrid alpha=0.5", "method": "hybrid", "alpha": 0.5},
        {"name": "Hybrid alpha=0.3", "method": "hybrid", "alpha": 0.3},
    ]
    
    results = {c["name"]: [] for c in configs}
    
    ir_store.load()
    
    users_with_gt = 0
    t0 = time.time()
    
    for u in users:
        train_clicked = u["train_clicked_aids"]
        gt = u["dev_ground_truth"]
        
        if not gt:
            continue
            
        users_with_gt += 1
        vec, topic_vec, _ = build_user_vector_from_sequence(u["train_sequence"], mode="full")
        if vec is None:
            continue
            
        mask = await get_candidate_mask(session, "dev", train_clicked)
        
        for c in configs:
            cands = await rank_candidates(
                session, vec, mask, top_n=10, 
                retrieval_method=c["method"], 
                hybrid_alpha=c["alpha"], 
                user_topic_vector=topic_vec
            )
            rec_mids = [cand["mind_id"] for cand in cands]
            
            m = {
                "p_at_5": precision_at_k(rec_mids, gt, 5),
                "p_at_10": precision_at_k(rec_mids, gt, 10),
                "r_at_5": recall_at_k(rec_mids, gt, 5),
                "r_at_10": recall_at_k(rec_mids, gt, 10),
                "ndcg_at_5": ndcg_at_k(rec_mids, gt, 5),
                "ndcg_at_10": ndcg_at_k(rec_mids, gt, 10),
                "hit_at_5": hit_rate_at_k(rec_mids, gt, 5),
                "hit_at_10": hit_rate_at_k(rec_mids, gt, 10),
            }
            results[c["name"]].append(m)
            
    total_time = time.time() - t0
    
    print(f"Evaluated {len(users)} users ({users_with_gt} with ground truth). Runtime: {total_time:.2f}s")
    
    for c in configs:
        name = c["name"]
        user_metrics = results[name]
        
        if not user_metrics:
            continue
            
        avg = {k: np.mean([m[k] for m in user_metrics]) for k in user_metrics[0].keys()}
        
        print(f"\n{name}:")
        print(f"  P@5:      {avg['p_at_5']:.4f}")
        print(f"  P@10:     {avg['p_at_10']:.4f}")
        print(f"  Recall@5: {avg['r_at_5']:.4f}")
        print(f"  Recall@10:{avg['r_at_10']:.4f}")
        print(f"  NDCG@5:   {avg['ndcg_at_5']:.4f}")
        print(f"  NDCG@10:  {avg['ndcg_at_10']:.4f}")
        print(f"  Hit@5:    {avg['hit_at_5']:.4f}")
        print(f"  Hit@10:   {avg['hit_at_10']:.4f}")


async def main():
    async for session in get_db():
        await run_inspection(session)
        await run_experiment(session)
        break

if __name__ == "__main__":
    asyncio.run(main())
