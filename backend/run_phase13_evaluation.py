import asyncio
import os
import sys
import time
import numpy as np
import sqlite3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.connection import get_db
from evaluation.datasets import EvaluationCohort
from evaluation.baselines import build_user_vector_from_sequence
from app.services.retrieval.candidate_retriever import get_candidate_mask
from app.services.retrieval.similarity_ranker import rank_candidates
from app.services.retrieval.index_loader import ir_store
from app.services.retrieval.popularity_loader import popularity_store
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k

async def run_inspection(session):
    print("\n" + "="*50)
    print("SMALL REAL-USER INSPECTION")
    print("="*50)
    
    db_path = "./data/news_discovery.db"
    target_users = ["U536528", "U13740", "U91836"]
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    ir_store.load()
    popularity_store.load(db_path)
    pop_array = popularity_store.get_popularity_array()
    
    for uid in target_users:
        print(f"\n--- USER: {uid} ---")
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
        
        vec, _, _ = build_user_vector_from_sequence(train_sequence, mode="full")
        mask = await get_candidate_mask(session, "dev", train_clicked_aids)
        
        # 1. Content-only top 10
        content_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="tfidf", ranking_method="content")
        # 2. Popularity-only top 10
        pop_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="tfidf", ranking_method="popularity")
        # 3. Content + popularity beta=0.7 top 10
        hybrid_cands = await rank_candidates(session, vec, mask, top_n=10, retrieval_method="tfidf", ranking_method="content_popularity", popularity_beta=0.7)
        
        content_mids = set(c['mind_id'] for c in content_cands)
        pop_mids = set(c['mind_id'] for c in pop_cands)
        hybrid_mids = set(c['mind_id'] for c in hybrid_cands)
        
        print("Overlap: Content & Hybrid:", len(content_mids & hybrid_mids))
        print("Overlap: Pop & Hybrid:", len(pop_mids & hybrid_mids))
        
        def print_cands(cands, label):
            print(f"\n{label}:")
            for r, c in enumerate(cands):
                mid = c['mind_id']
                row_idx = ir_store.mind_id_to_row.get(mid)
                pop_score = pop_array[row_idx] if row_idx is not None else 0.0
                
                # To get the raw count we inverse the normalization:
                # norm_pop = log(1+count) / max_log_count
                # log(1+count) = norm_pop * max_log_count
                # count = exp(norm_pop * max_log_count) - 1
                # But we can just use popularity_store internals if needed.
                # Just estimating count roughly:
                max_log_count = np.log(1 + 66449)
                raw_count = int(np.round(np.exp(pop_score * max_log_count) - 1))
                
                # Find content sim
                content_sim = 0.0
                if row_idx is not None and vec is not None:
                    # dot product is fast
                    content_sim = float(vec.dot(ir_store.tfidf_matrix[row_idx].T).toarray()[0,0])
                
                print(f"  {r+1}. {mid} | {c.get('title', '')[:30]}... | {c.get('category', '')} "
                      f"| ContentSim: {content_sim:.4f} | PopCnt: {raw_count} | PopScore: {pop_score:.4f} | Final: {c.get('similarity', 0.0):.4f}")

        print_cands(content_cands, "Content-Only Top 10")
        print_cands(pop_cands, "Popularity-Only Top 10")
        print_cands(hybrid_cands, "Content + Popularity (beta=0.7) Top 10")
            
    con.close()

async def run_experiment(session):
    print("\n" + "="*50)
    print("SMALL OFFLINE EXPERIMENT")
    print("="*50)
    
    db_path = "./data/news_discovery.db"
    cohort = EvaluationCohort(db_path, max_users=20, seed=42, split="dev")
    users = cohort.load_cohort()
    
    configs = [
        {"name": "Content-only", "method": "content", "beta": 1.0},
        {"name": "Popularity-only", "method": "popularity", "beta": 0.0},
        {"name": "beta=0.9", "method": "content_popularity", "beta": 0.9},
        {"name": "beta=0.8", "method": "content_popularity", "beta": 0.8},
        {"name": "beta=0.7", "method": "content_popularity", "beta": 0.7},
        {"name": "beta=0.5", "method": "content_popularity", "beta": 0.5},
        {"name": "beta=0.3", "method": "content_popularity", "beta": 0.3},
    ]
    
    results = {c["name"]: [] for c in configs}
    rec_stats = {c["name"]: {"avg_pop": [], "zero_pop": []} for c in configs}
    
    ir_store.load()
    popularity_store.load(db_path)
    pop_array = popularity_store.get_popularity_array()
    
    users_with_gt = 0
    t0 = time.time()
    
    for u in users:
        train_clicked = u["train_clicked_aids"]
        gt = u["dev_ground_truth"]
        
        if not gt:
            continue
            
        users_with_gt += 1
        vec, _, _ = build_user_vector_from_sequence(u["train_sequence"], mode="full")
        if vec is None:
            continue
            
        mask = await get_candidate_mask(session, "dev", train_clicked)
        
        for c in configs:
            cands = await rank_candidates(
                session, vec, mask, top_n=10, 
                retrieval_method="tfidf", 
                ranking_method=c["method"], 
                popularity_beta=c["beta"]
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
            
            pops = []
            zeros = 0
            for mid in rec_mids:
                row_idx = ir_store.mind_id_to_row.get(mid)
                p = pop_array[row_idx] if row_idx is not None else 0.0
                pops.append(p)
                if p == 0.0:
                    zeros += 1
            if pops:
                rec_stats[c["name"]]["avg_pop"].append(np.mean(pops))
                rec_stats[c["name"]]["zero_pop"].append(zeros / len(pops))
            
    total_time = time.time() - t0
    
    print(f"Evaluated {len(users)} users ({users_with_gt} with ground truth). Runtime: {total_time:.2f}s")
    
    # Calculate global popularities
    num_candidates = np.count_nonzero(pop_array > -1.0) # all articles
    sorted_pops = np.sort(pop_array)[::-1]
    top_1_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.01)]
    top_5_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.05)]
    top_10_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.10)]
    
    for c in configs:
        name = c["name"]
        user_metrics = results[name]
        stats = rec_stats[name]
        
        if not user_metrics:
            continue
            
        avg = {k: np.mean([m[k] for m in user_metrics]) for k in user_metrics[0].keys()}
        avg_pop = np.mean(stats["avg_pop"]) if stats["avg_pop"] else 0.0
        avg_zero = np.mean(stats["zero_pop"]) if stats["zero_pop"] else 0.0
        
        print(f"\n{name}:")
        print(f"  P@10:     {avg['p_at_10']:.4f}")
        print(f"  Recall@10:{avg['r_at_10']:.4f}")
        print(f"  NDCG@10:  {avg['ndcg_at_10']:.4f}")
        print(f"  Hit@10:   {avg['hit_at_10']:.4f}")
        print(f"  Avg Pop Score: {avg_pop:.4f}")
        print(f"  Zero Pop Recs: {avg_zero*100:.1f}%")

async def main():
    async for session in get_db():
        await run_inspection(session)
        await run_experiment(session)
        break

if __name__ == "__main__":
    asyncio.run(main())
