import os
import json
import numpy as np

CHECKPOINT_PATH = "data/evaluation/phase15/raw_session_results.jsonl"

def main():
    user_metrics = {}
    with open(CHECKPOINT_PATH, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if not rec.get("has_sessions", False):
                continue
            
            for c_name, c_data in rec.get("configs", {}).items():
                if c_name not in user_metrics:
                    user_metrics[c_name] = {"ndcg": [], "hit": [], "sessions": 0, "missing_ndcg": 0, "zero_ndcg": 0}
                
                m = c_data.get("metrics")
                if not m:
                    user_metrics[c_name]["missing_ndcg"] += 1
                    continue
                
                ndcg = m.get("ndcg_at_10")
                if ndcg is None:
                    user_metrics[c_name]["missing_ndcg"] += 1
                    continue
                
                if ndcg == 0.0:
                    user_metrics[c_name]["zero_ndcg"] += 1
                    
                user_metrics[c_name]["ndcg"].append(ndcg)
                user_metrics[c_name]["hit"].append(m.get("hit_rate_at_10", 0.0))
                user_metrics[c_name]["sessions"] += 1
                
    for c_name, stats in user_metrics.items():
        arr = np.array(stats["ndcg"])
        hit_arr = np.array(stats["hit"])
        print(f"Method: {c_name}")
        print(f"  Sessions: {stats['sessions']}")
        print(f"  Valid observations: {len(arr)}")
        print(f"  Missing NDCG: {stats['missing_ndcg']}")
        print(f"  Zero NDCG: {stats['zero_ndcg']}")
        print(f"  Min NDCG: {np.min(arr) if len(arr) else 0}")
        print(f"  Max NDCG: {np.max(arr) if len(arr) else 0}")
        print(f"  Mean NDCG: {np.mean(arr) if len(arr) else 0:.4f}")
        print(f"  Median NDCG: {np.median(arr) if len(arr) else 0:.4f}")
        print(f"  Mean Hit@10: {np.mean(hit_arr) if len(hit_arr) else 0:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
