import urllib.request
import json
import sqlite3

def run_smoke_test():
    base_url = "http://localhost:8000"
    
    # Get actual IDs from DB
    con = sqlite3.connect("./data/news_discovery.db")
    cur = con.cursor()
    user_id = "U13740"
    article_id = "N55528"
    
    # Actually, I'll just pick a known user from the database directly by mind_user_id.
    con = sqlite3.connect("./data/news_discovery.db")
    user_row = con.cursor().execute("SELECT mind_user_id FROM mind_users WHERE mind_user_id LIKE 'U%' LIMIT 1").fetchone()
    user_id = user_row[0] if user_row else "U0"
    article_row = con.cursor().execute("SELECT mind_id FROM news_articles WHERE mind_id LIKE 'N%' LIMIT 1").fetchone()
    article_id = article_row[0] if article_row else "N1"
    con.close()
    article_row = cur.execute("SELECT mind_id FROM news_articles LIMIT 1").fetchone()
    if not article_row:
        print("FAIL: No articles found.")
        return
    article_id = article_row[0]
    con.close()
    
    print(f"Using User: {user_id}, Article: {article_id}")
    
    endpoints = [
        f"/api/health",
        f"/api/dataset/stats",
        f"/api/articles/mind/{article_id}/representation",
        f"/api/users/mind/{user_id}/profile",
        f"/api/users/mind/{user_id}/drift",
        f"/api/users/mind/{user_id}/candidates",
        f"/api/users/mind/{user_id}/recommendations",
        f"/api/users/mind/{user_id}/feedback"
    ]
    
    for ep in endpoints:
        url = base_url + ep
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                data = json.loads(response.read().decode())
                print(f"PASS: {ep} (Status {status})")
                # Just checking it returns JSON successfully
        except Exception as e:
            print(f"FAIL: {ep} - {e}")

if __name__ == "__main__":
    run_smoke_test()
