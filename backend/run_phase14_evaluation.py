import asyncio
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.datasets import EvaluationCohort
from evaluation.baselines import build_user_vector_from_sequence
from app.services.retrieval.index_loader import ir_store
from app.services.retrieval.popularity_loader import popularity_store
from evaluation.protocols import ImpressionLevelRerankingProtocol
from evaluation.metrics import intra_list_diversity, category_diversity, subcategory_diversity

async def run_experiment():
    print("="*50)
    print("PHASE 14: IMPRESSION-LEVEL RERANKING EXPERIMENT")
    print("="*50)
    
    db_path = "./data/news_discovery.db"
    
    # 1. Load the fixed evaluation cohort (same 100-user dev sample as Phase 11)
    cohort = EvaluationCohort(db_path, max_users=100, seed=42, split="dev")
    users = cohort.load_cohort()
    
    # Load stores
    ir_store.load()
    popularity_store.load(db_path)
    
    protocol = ImpressionLevelRerankingProtocol(final_k=10)
    
    configs = [
        {"name": "A. TF-IDF", "retrieval_method": "tfidf", "ranking_method": "content", "apply_mmr": False},
        {"name": "B. Topic-only", "retrieval_method": "topic", "ranking_method": "content", "apply_mmr": False},
        {"name": "C. Hybrid alpha=0.8", "retrieval_method": "hybrid", "hybrid_alpha": 0.8, "ranking_method": "content", "apply_mmr": False},
        {"name": "C. Hybrid alpha=0.7", "retrieval_method": "hybrid", "hybrid_alpha": 0.7, "ranking_method": "content", "apply_mmr": False},
        {"name": "C. Hybrid alpha=0.5", "retrieval_method": "hybrid", "hybrid_alpha": 0.5, "ranking_method": "content", "apply_mmr": False},
        {"name": "C. Hybrid alpha=0.3", "retrieval_method": "hybrid", "hybrid_alpha": 0.3, "ranking_method": "content", "apply_mmr": False},
        {"name": "D. Popularity-only", "retrieval_method": "tfidf", "ranking_method": "popularity", "apply_mmr": False},
        {"name": "E. TF-IDF+Pop beta=0.9", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.9, "apply_mmr": False},
        {"name": "E. TF-IDF+Pop beta=0.8", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.8, "apply_mmr": False},
        {"name": "E. TF-IDF+Pop beta=0.7", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.7, "apply_mmr": False},
        {"name": "E. TF-IDF+Pop beta=0.5", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.5, "apply_mmr": False},
        {"name": "F. TF-IDF + MMR", "retrieval_method": "tfidf", "ranking_method": "content", "apply_mmr": True, "mmr_lambda": 0.7},
        {"name": "G. TF-IDF+Pop beta=0.7 + MMR", "retrieval_method": "tfidf", "ranking_method": "content_popularity", "popularity_beta": 0.7, "apply_mmr": True, "mmr_lambda": 0.7},
    ]
    
    # Track stats
    results = {c["name"]: [] for c in configs}
    pop_stats = {c["name"]: {"avg_pop": [], "zero_pop": [], "top_1": [], "top_5": [], "top_10": []} for c in configs}
    div_stats = {c["name"]: {"ild": [], "cat": [], "subcat": []} for c in configs}
    
    # Pre-calculate top-K popularity thresholds
    pop_array = popularity_store.get_popularity_array()
    valid_pops = pop_array[pop_array > 0.0]
    sorted_pops = np.sort(valid_pops)[::-1]
    top_1_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.01)] if len(sorted_pops) > 0 else 1.0
    top_5_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.05)] if len(sorted_pops) > 0 else 1.0
    top_10_pct_thresh = sorted_pops[int(len(sorted_pops) * 0.10)] if len(sorted_pops) > 0 else 1.0
    
    users_with_sessions = 0
    t0 = time.time()
    
    for u in users:
        sessions = u.get("dev_sessions", [])
        if not sessions:
            continue
            
        users_with_sessions += 1
        vec, topic_vec, _ = build_user_vector_from_sequence(u["train_sequence"], mode="full")
        
        for c in configs:
            metrics, _, _, session_recs = protocol.evaluate_user_impressions(
                u, 
                vec, 
                user_topic_vector=topic_vec, 
                retrieval_method=c.get("retrieval_method", "tfidf"),
                hybrid_alpha=c.get("hybrid_alpha", 0.7),
                ranking_method=c.get("ranking_method", "content"),
                popularity_beta=c.get("popularity_beta", 0.7),
                apply_mmr=c.get("apply_mmr", False),
                mmr_lambda=c.get("mmr_lambda", 0.7)
            )
            
            if metrics:
                results[c["name"]].append(metrics)
                
            # Pop and Diversity tracking per session
            for sess_recs in session_recs:
                if not sess_recs:
                    continue
                
                # Div
                mids = [r["mind_id"] for r in sess_recs]
                div_stats[c["name"]]["ild"].append(intra_list_diversity(mids, ir_store, 10))
                div_stats[c["name"]]["cat"].append(category_diversity(sess_recs, 10))
                div_stats[c["name"]]["subcat"].append(subcategory_diversity(sess_recs, 10))
                
                # Pop
                pops = [r["pop_score"] for r in sess_recs]
                pop_stats[c["name"]]["avg_pop"].append(np.mean(pops))
                pop_stats[c["name"]]["zero_pop"].append(sum(1 for p in pops if p == 0.0) / len(pops))
                pop_stats[c["name"]]["top_1"].append(sum(1 for p in pops if p >= top_1_pct_thresh) / len(pops))
                pop_stats[c["name"]]["top_5"].append(sum(1 for p in pops if p >= top_5_pct_thresh) / len(pops))
                pop_stats[c["name"]]["top_10"].append(sum(1 for p in pops if p >= top_10_pct_thresh) / len(pops))
                
    total_time = time.time() - t0
    
    print(f"\nEvaluated {users_with_sessions} users (148 total sessions). Runtime: {total_time:.2f}s")
    
    # Save results
    os.makedirs("./data/evaluation/phase14", exist_ok=True)
    
    with open("./data/evaluation/phase14/metrics.txt", "w") as f:
        f.write(f"Phase 14 Impression-Level Reranking\n")
        f.write(f"Users: {users_with_sessions}\n")
        f.write(f"Runtime: {total_time:.2f}s\n\n")
        
        f.write(f"{'Method':<30} | {'P@5':<6} | {'P@10':<6} | {'NDCG@5':<7} | {'NDCG@10':<7} | {'Hit@5':<6} | {'Hit@10':<6} | {'ILD@10':<6} | {'AvgPop':<6} | {'Top1%Pop':<8} | {'ZeroPop':<7}\n")
        f.write("-" * 125 + "\n")
        
        for c in configs:
            name = c["name"]
            metrics_list = results[name]
            p_stats = pop_stats[name]
            d_stats = div_stats[name]
            
            if not metrics_list:
                continue
                
            avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
            avg_pop = np.mean(p_stats["avg_pop"])
            top_1 = np.mean(p_stats["top_1"])
            zero_pop = np.mean(p_stats["zero_pop"])
            ild = np.mean(d_stats["ild"])
            
            f.write(f"{name:<30} | {avg['p_at_5']:.4f} | {avg['p_at_10']:.4f} | {avg['ndcg_at_5']:.4f} | {avg['ndcg_at_10']:.4f} | {avg['hit_rate_at_5']:.4f} | {avg['hit_rate_at_10']:.4f} | {ild:.4f} | {avg_pop:.4f} | {top_1*100:>5.1f}% | {zero_pop*100:>4.1f}%\n")
            
            # Print to stdout
            print(f"{name:<30} | P@10: {avg['p_at_10']:.4f} | NDCG@10: {avg['ndcg_at_10']:.4f} | Hit@10: {avg['hit_rate_at_10']:.4f} | ILD: {ild:.4f} | Pop: {avg_pop:.4f}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
