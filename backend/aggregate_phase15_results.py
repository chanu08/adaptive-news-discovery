import os
import json
import numpy as np

RESULTS_DIR = "./data/evaluation/phase15"
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "raw_session_results.jsonl")

def bootstrap_ci(data, iterations=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(data)
    means = []
    for _ in range(iterations):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return lower, upper

def main():
    if not os.path.exists(CHECKPOINT_FILE):
        print("No results found.")
        return
        
    print("Loading data...")
    user_metrics = {}
    
    users_processed = 0
    users_with_sessions = 0
    total_sessions = 0
    total_runtime = 0.0
    
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            users_processed += 1
            if not rec.get("has_sessions", False):
                continue
            
            users_with_sessions += 1
            total_sessions += rec.get("num_sessions", 0)
            total_runtime += rec.get("runtime", 0.0)
            
            for c_name, c_data in rec.get("configs", {}).items():
                if c_name not in user_metrics:
                    user_metrics[c_name] = {
                        "p_at_5": [], "p_at_10": [], 
                        "ndcg_at_5": [], "ndcg_at_10": [],
                        "hit_at_5": [], "hit_at_10": [],
                        "ild": [], "avg_pop": [], "top_1": [], "top_5": [], "top_10": [], "zero_pop": []
                    }
                
                m = c_data.get("metrics")
                if not m: continue
                
                user_metrics[c_name]["p_at_5"].append(m.get("p_at_5", 0))
                user_metrics[c_name]["p_at_10"].append(m.get("p_at_10", 0))
                user_metrics[c_name]["ndcg_at_5"].append(m.get("ndcg_at_5", 0))
                user_metrics[c_name]["ndcg_at_10"].append(m.get("ndcg_at_10", 0))
                user_metrics[c_name]["hit_at_5"].append(m.get("hit_rate_at_5", 0))
                user_metrics[c_name]["hit_at_10"].append(m.get("hit_rate_at_10", 0))
                
                user_metrics[c_name]["ild"].append(c_data.get("ild", 0))
                user_metrics[c_name]["avg_pop"].append(c_data.get("avg_pop", 0))
                user_metrics[c_name]["top_1"].append(c_data.get("top_1", 0))
                user_metrics[c_name]["top_5"].append(c_data.get("top_5", 0))
                user_metrics[c_name]["top_10"].append(c_data.get("top_10", 0))
                user_metrics[c_name]["zero_pop"].append(c_data.get("zero_pop", 0))

    print(f"Total Evaluated Users: {users_processed}")
    print(f"Users with Dev Sessions: {users_with_sessions}")
    print(f"Total Dev Sessions: {total_sessions}")
    
    out_lines = []
    out_lines.append(f"Phase 15 Full-Scale Impression-Level Reranking")
    out_lines.append(f"Users with Sessions: {users_with_sessions}")
    out_lines.append(f"Total Dev Sessions: {total_sessions}")
    out_lines.append(f"Total Ranking Runtime: {total_runtime:.2f}s (across all cores/nodes)\n")
    
    # TF-IDF Baseline
    baseline = "A. TF-IDF"
    baseline_ndcg = user_metrics[baseline]["ndcg_at_10"] if baseline in user_metrics else None
    
    out_lines.append(f"{'Method':<30} | {'P@10':<6} | {'NDCG@10':<7} | {'NDCG@10 CI (95%)':<21} | {'Hit@10':<6} | {'ILD@10':<6} | {'AvgPop':<6} | {'Top1%Pop':<8} | {'ZeroPop':<7}")
    out_lines.append("-" * 140)
    
    for name, m_dict in user_metrics.items():
        if not m_dict["ndcg_at_10"]: continue
        
        p10 = np.mean(m_dict["p_at_10"])
        ndcg10 = np.mean(m_dict["ndcg_at_10"])
        hit10 = np.mean(m_dict["hit_at_10"])
        ild = np.mean(m_dict["ild"])
        avg_pop = np.mean(m_dict["avg_pop"])
        t1 = np.mean(m_dict["top_1"])
        z_pop = np.mean(m_dict["zero_pop"])
        
        ci_lower, ci_upper = bootstrap_ci(m_dict["ndcg_at_10"])
        
        out_lines.append(f"{name:<30} | {p10:.4f} | {ndcg10:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] | {hit10:.4f} | {ild:.4f} | {avg_pop:.4f} | {t1*100:>5.1f}% | {z_pop*100:>4.1f}%")
        
    out_lines.append("\n" + "="*50)
    out_lines.append("RELATIVE DIFFERENCES VS TF-IDF BASELINE")
    out_lines.append("="*50)
    
    if baseline_ndcg:
        b_mean = np.mean(baseline_ndcg)
        for name, m_dict in user_metrics.items():
            if name == baseline or not m_dict["ndcg_at_10"]: continue
            diffs = [x - y for x, y in zip(m_dict["ndcg_at_10"], baseline_ndcg)]
            d_mean = np.mean(diffs)
            ci_l, ci_u = bootstrap_ci(diffs)
            rel_diff = (d_mean / b_mean) * 100 if b_mean > 0 else 0
            
            sig = "*" if (ci_l > 0 or ci_u < 0) else ""
            out_lines.append(f"{name:<30} diff: {d_mean:+.4f} ({rel_diff:+.1f}%)  CI: [{ci_l:+.4f}, {ci_u:+.4f}] {sig}")

    summary = "\n".join(out_lines)
    print(summary)
    with open(os.path.join(RESULTS_DIR, "aggregated_metrics.txt"), "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
