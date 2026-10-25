import asyncio
import os
import sys
import time
import json
import numpy as np
import sqlite3
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.baselines import build_user_vector_from_sequence
from app.services.retrieval.index_loader import ir_store
from app.services.retrieval.popularity_loader import popularity_store
from evaluation.protocols import ImpressionLevelRerankingProtocol
from evaluation.metrics import intra_list_diversity

RESULTS_DIR = "./data/evaluation/phase15"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "raw_session_results.jsonl")

configs = [
    {"name": "A. TF-IDF", "retrieval_method": "tfidf", "ranking_method": "content", "apply_mmr": False},
    {"name": "B. Topic-only", "retrieval_method": "topic", "ranking_method": "content", "apply_mmr": False},
    {"name": "C. Hybrid alpha=0.8", "retrieval_method": "hybrid", "hybrid_alpha": 0.8, "ranking_method": "content", "apply_mmr": False},
    {"name": "C. Hybrid alpha=0.7", "retrieval_method": "hybrid", "hybrid_alpha": 0.7, "ranking_method": "content", "apply_mmr": False},
    {"name": "C. Hybrid alpha=0.5", "retrieval_method": "hybrid", "hybrid_alpha": 0.5, "ranking_method": "content", "apply_mmr": False},
    {"name": "C. Hybrid alpha=0.3", "retrieval_method": "hybrid", "hybrid_alpha": 0.3, "ranking_method": "content", "apply_mmr": False},
    {"name": "D. TF-IDF + MMR", "retrieval_method": "tfidf", "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
    {"name": "E. Hybrid alpha=0.8 + MMR", "retrieval_method": "hybrid", "hybrid_alpha": 0.8, "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
    {"name": "E. Hybrid alpha=0.7 + MMR", "retrieval_method": "hybrid", "hybrid_alpha": 0.7, "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
    {"name": "E. Hybrid alpha=0.5 + MMR", "retrieval_method": "hybrid", "hybrid_alpha": 0.5, "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
    {"name": "E. Hybrid alpha=0.3 + MMR", "retrieval_method": "hybrid", "hybrid_alpha": 0.3, "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
    {"name": "F. TF-IDF+Pop beta=0.7", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.7, "apply_mmr": False},
    {"name": "F. TF-IDF+Pop beta=0.7 + MMR", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.7, "apply_mmr": True, "mmr_lambda": 0.7},
]

def yield_users(db_path, seed=42):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    cur.execute("""
        SELECT DISTINCT b1.user_mind_id
        FROM behavior_logs b1
        JOIN behavior_logs b2 ON b1.user_mind_id = b2.user_mind_id
        JOIN impression_items i ON b2.id = i.behavior_log_id
        WHERE b1.split = 'train' 
        AND b2.split = 'dev'
        AND i.clicked = 1
    """)
    eligible_user_ids = [r[0] for r in cur.fetchall()]
    rng = random.Random(seed)
    eligible_user_ids.sort()
    rng.shuffle(eligible_user_ids) # We can process in any order as long as they are distinct.
    
    for uid in eligible_user_ids:
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
                    try: dt = datetime.fromisoformat(dt_str)
                    except: pass
            train_imp_clicks.append((aid, dt, iid))

        train_imp_clicks.sort(key=lambda x: (x[1] if x[1] else datetime.min, x[2]))
        train_sequence = [(aid, None) for aid in hist_clicks] + [(aid, dt) for aid, dt, _ in train_imp_clicks]

        cur.execute("""
            SELECT b.id, b.time, b.impressions_raw
            FROM behavior_logs b
            WHERE b.user_mind_id = ? AND b.split = 'dev'
            ORDER BY b.id ASC
        """, (uid,))
        dev_sessions = []
        for b_id, b_time, imp_raw in cur.fetchall():
            if not imp_raw: continue
            impressed = []
            clicked = set()
            for token in imp_raw.split():
                if '-' in token:
                    aid, c = token.split('-')
                    impressed.append(aid)
                    if c == '1': clicked.add(aid)
            if impressed:
                dev_sessions.append({"session_id": b_id, "time": b_time, "impressed_articles": impressed, "clicked_articles": list(clicked)})

        yield {"user_id": uid, "train_sequence": train_sequence, "dev_sessions": dev_sessions}
    con.close()

async def run_experiment():
    print("="*50)
    print("PHASE 15: FULL-SCALE EVALUATION (SYNCHRONOUS GENERATOR)")
    print("="*50)
    
    ir_store.load()
    popularity_store.load("./data/news_discovery.db")
    protocol = ImpressionLevelRerankingProtocol(final_k=10)
    
    pop_array = popularity_store.get_popularity_array()
    valid_pops = pop_array[pop_array > 0.0]
    sorted_pops = np.sort(valid_pops)[::-1]
    top_1 = sorted_pops[int(len(sorted_pops) * 0.01)] if len(sorted_pops) > 0 else 1.0
    top_5 = sorted_pops[int(len(sorted_pops) * 0.05)] if len(sorted_pops) > 0 else 1.0
    top_10 = sorted_pops[int(len(sorted_pops) * 0.10)] if len(sorted_pops) > 0 else 1.0
    
    processed_users = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            for line in f:
                if line.strip(): processed_users.add(json.loads(line)["user_mind_id"])
        print(f"Resuming. {len(processed_users)} users already processed.")
        
    t0 = time.time()
    processed_count = 0
    skipped_count = 0
    
    with open(CHECKPOINT_FILE, "a") as f_out:
        for u in yield_users("./data/news_discovery.db"):
            uid = u["user_id"]
            if uid in processed_users:
                skipped_count += 1
                continue
                
            sessions = u.get("dev_sessions", [])
            if not sessions:
                f_out.write(json.dumps({"user_mind_id": uid, "has_sessions": False}) + "\n")
                f_out.flush()
                continue
                
            t_user_start = time.perf_counter()
            vec, topic_vec, _ = build_user_vector_from_sequence(u["train_sequence"], mode="full")
            
            user_res = {"user_mind_id": uid, "has_sessions": True, "num_sessions": len(sessions), "configs": {}}
            
            for c in configs:
                t_method_start = time.perf_counter()
                metrics, _, v_sess, session_recs = protocol.evaluate_user_impressions(
                    u, vec, user_topic_vector=topic_vec, 
                    retrieval_method=c.get("retrieval_method", "tfidf"),
                    hybrid_alpha=c.get("hybrid_alpha", 0.7),
                    ranking_method=c.get("ranking_method", "content"),
                    popularity_beta=c.get("popularity_beta", 0.7),
                    apply_mmr=c.get("apply_mmr", False), mmr_lambda=c.get("mmr_lambda", 0.7)
                )
                
                c_data = {"metrics": metrics, "valid_sess_count": v_sess, "runtime": time.perf_counter() - t_method_start}
                
                if metrics and session_recs:
                    ild_list, p_list, t1_list, t5_list, t10_list, z_list = [], [], [], [], [], []
                    for sr in session_recs:
                        if not sr: continue
                        mids = [r["mind_id"] for r in sr]
                        pops = [r["pop_score"] for r in sr]
                        ild_list.append(intra_list_diversity(mids, ir_store, 10))
                        p_list.append(np.mean(pops))
                        t1_list.append(sum(1 for p in pops if p >= top_1) / len(pops))
                        t5_list.append(sum(1 for p in pops if p >= top_5) / len(pops))
                        t10_list.append(sum(1 for p in pops if p >= top_10) / len(pops))
                        z_list.append(sum(1 for p in pops if p == 0.0) / len(pops))
                        
                    if ild_list: c_data["ild"] = float(np.mean(ild_list))
                    if p_list: c_data["avg_pop"] = float(np.mean(p_list))
                    if t1_list: c_data["top_1"] = float(np.mean(t1_list))
                    if t5_list: c_data["top_5"] = float(np.mean(t5_list))
                    if t10_list: c_data["top_10"] = float(np.mean(t10_list))
                    if z_list: c_data["zero_pop"] = float(np.mean(z_list))
                    
                user_res["configs"][c["name"]] = c_data
                
            user_res["runtime"] = time.perf_counter() - t_user_start
            f_out.write(json.dumps(user_res) + "\n")
            f_out.flush()
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} users (+{skipped_count} skipped) in {time.time() - t0:.2f}s...")
                
    print(f"\nEvaluation finished. Processed {processed_count} users in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    asyncio.run(run_experiment())
