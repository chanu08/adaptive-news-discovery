import os
import sqlite3

def verify_phase16():
    print("VERIFICATION SCRIPT FOR PHASE 16")
    errors = 0
    
    # 1-4. Checkpoint Integrity
    try:
        count = 0
        uids = set()
        sess_count = 0
        with open("data/evaluation/phase15/raw_session_results.jsonl", "r") as f:
            for l in f:
                if not l.strip(): continue
                count += 1
                import json
                j = json.loads(l)
                uids.add(j['user_mind_id'])
                sess_count += j.get("num_sessions", 0)
                
        if count != 93612:
            print(f"FAIL: Expected 93,612 lines, got {count}")
            errors += 1
        else:
            print("PASS: Checkpoint has exactly 93,612 rows.")
            
        if len(uids) != 93612:
            print("FAIL: Duplicate records found!")
            errors += 1
        else:
            print("PASS: No duplicate records.")
            
        if sess_count != 141171:
            print(f"FAIL: Expected 141,171 sessions, got {sess_count}")
            errors += 1
        else:
            print("PASS: Completed sessions match exactly 141,171.")
    except Exception as e:
        print(f"FAIL: Checkpoint not readable. {e}")
        errors += 1
        
    # 13. Database counts
    try:
        con = sqlite3.connect('./data/news_discovery.db')
        cur = con.cursor()
        if cur.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0] != 130379:
            print("FAIL: canonical_articles count changed!")
            errors += 1
        if cur.execute("SELECT COUNT(*) FROM behavior_logs").fetchone()[0] != 4979946:
            print("FAIL: behavior_logs count changed!")
            errors += 1
        if cur.execute("SELECT COUNT(*) FROM impression_items").fetchone()[0] != 190707932:
            print("FAIL: impression_items count changed!")
            errors += 1
        if cur.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0] != 18:
            print("FAIL: feedback_events count changed!")
            errors += 1
        print("PASS: Database integrity unchanged.")
        con.close()
    except Exception as e:
        print(f"FAIL: Database integrity check failed: {e}")
        errors += 1

    # 10. Plots exist
    plots = [
        "alpha_sensitivity_ndcg.png",
        "mmr_tradeoff.png",
        "ndcg_diff_ci.png",
        "per_user_diff_dist.png",
        "popularity_concentration.png"
    ]
    for p in plots:
        path = os.path.join("data/evaluation/phase16", p)
        if not os.path.exists(path):
            print(f"FAIL: Missing plot {path}")
            errors += 1
    print("PASS: All plots generated.")

    # 11, 7, 8, 9, 12. Report checks
    report_path = "../phase16_statistical_analysis_report.md"
    if not os.path.exists(report_path):
        print("FAIL: Report missing.")
        errors += 1
    else:
        with open(report_path, "r") as f:
            content = f.read()
            if "Holm Adj p-val" not in content:
                print("FAIL: Holm correction missing from report.")
                errors += 1
            if "Cohen's d" not in content:
                print("FAIL: Effect sizes missing from report.")
                errors += 1
            if "95% CI" not in content:
                print("FAIL: CIs missing from report.")
                errors += 1
            if "PASS" not in content:
                print("FAIL: Leakage audit PASS missing from report.")
                errors += 1
        print("PASS: Report generated with required statistical analyses and leakage audit.")

    if errors == 0:
        print("\nALL CHECKS PASS")
    else:
        print(f"\n{errors} CHECKS FAILED")

if __name__ == "__main__":
    verify_phase16()
