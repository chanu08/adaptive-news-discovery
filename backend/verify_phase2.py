import sqlite3
import os
import urllib.request, json, sys

def get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"FAILED to fetch {url}: {e}")
        return None

def verify():
    print("========================================")
    print("      PHASE 2 FINAL VERIFICATION SCRIPT")
    print("========================================\n")
    
    passed = True
    db_path = "data/news_discovery.db"
    
    # Database exists
    if not os.path.exists(db_path):
        print("[FAIL] Database does not exist!")
        sys.exit(1)
    else:
        print("[PASS] Database exists.")
        
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    print("\n--- Database Checks ---")
    
    # Canonical articles
    cur.execute("SELECT COUNT(*) FROM news_articles")
    articles = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT mind_id) FROM news_articles")
    unique_mind = cur.fetchone()[0]
    
    if articles == unique_mind and articles > 0:
        print(f"[PASS] Article uniqueness verified ({articles} canonical articles, 0 duplicates).")
    else:
        print(f"[FAIL] Article uniqueness violated! rows={articles}, unique={unique_mind}")
        passed = False
        
    # Split memberships
    cur.execute("SELECT split, COUNT(*) FROM article_splits GROUP BY split")
    splits = {r[0]: r[1] for r in cur.fetchall()}
    if splits.get('train', 0) > 0 and splits.get('dev', 0) > 0 and splits.get('test', 0) > 0:
        print(f"[PASS] Split memberships verified (Train: {splits.get('train')}, Dev: {splits.get('dev')}, Test: {splits.get('test')}).")
    else:
        print(f"[FAIL] Missing split memberships: {splits}")
        passed = False
        
    cur.execute("SELECT COUNT(*) FROM (SELECT article_id FROM article_splits GROUP BY article_id HAVING COUNT(*) > 1)")
    multi_split = cur.fetchone()[0]
    print(f"       Multi-split articles: {multi_split}")
    
    # Users & Behaviors
    cur.execute("SELECT COUNT(*) FROM mind_users")
    users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM behavior_logs")
    logs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM impression_items")
    items = cur.fetchone()[0]
    cur.execute("SELECT SUM(clicked) FROM impression_items")
    clicks = cur.fetchone()[0] or 0
    
    if users > 0 and logs > 0 and items > 0:
        print(f"[PASS] Users ({users}), logs ({logs}), items ({items}), clicks ({clicks}) populated.")
    else:
        print("[FAIL] Missing users/logs/items.")
        passed = False
        
    # Orphan references
    cur.execute("SELECT COUNT(*) FROM impression_items WHERE article_mind_id NOT IN (SELECT mind_id FROM news_articles)")
    orphans = cur.fetchone()[0]
    if orphans == 0:
        print("[PASS] Zero orphan references from impressions to articles.")
    else:
        print(f"[FAIL] Found {orphans} orphan impression items!")
        passed = False
        
    print("\n--- API Checks ---")
    
    health = get('http://localhost:8000/api/health')
    if health and health.get("status") == "ok":
        print("[PASS] Backend health is OK.")
    else:
        print("[FAIL] Backend health failed.")
        passed = False
        
    stats = get('http://localhost:8000/api/dataset/stats')
    if stats and stats.get("total_articles") == articles:
        print("[PASS] API /api/dataset/stats returns consistent data.")
    else:
        print(f"[FAIL] API stats mismatch or failed.")
        passed = False

    arts = get('http://localhost:8000/api/articles/?limit=1')
    if arts and arts.get("items"):
        art_id = arts["items"][0]["id"]
        single = get(f'http://localhost:8000/api/articles/{art_id}')
        if single and single.get("id") == art_id:
            print("[PASS] API /api/articles and single retrieval work.")
        else:
            print("[FAIL] API single article failed.")
            passed = False
    else:
        print("[FAIL] API articles endpoint failed.")
        passed = False
        


    print("\n========================================")
    if passed:
        print("  OVERALL VERIFICATION: PASS")
        sys.exit(0)
    else:
        print("  OVERALL VERIFICATION: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    verify()
