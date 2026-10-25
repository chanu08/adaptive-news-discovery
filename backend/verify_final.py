import sqlite3
import time
import os

db_path = "data/news_discovery.db"

# Wait for fast_ingest.py to finish (size stops changing or takes locks off)
print("Waiting for ingestion to complete...")
last_size = -1
while True:
    if os.path.exists(db_path):
        current_size = os.path.getsize(db_path)
        if current_size == last_size and current_size > 0:
            # Maybe finished, try to read
            try:
                conn = sqlite3.connect(db_path, timeout=5)
                # Ensure it's not locked by writing
                conn.execute("PRAGMA journal_mode")
                conn.close()
                time.sleep(2)  # Wait just a bit to be sure it's completely closed
                
                # Double check size didn't change
                if os.path.getsize(db_path) == current_size:
                    break
            except Exception:
                pass
        last_size = current_size
    time.sleep(5)

print("Ingestion completed! Running verification...\n")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

splits = ["train", "dev", "test"]

print("=== FINAL MINDLARGE INGESTION ===\n")

for split in splits:
    print(split.upper())
    
    cur.execute("SELECT COUNT(*) FROM news_articles WHERE split = ?", (split,))
    articles = cur.fetchone()[0]
    print(f"Articles: {articles}")
    
    cur.execute("SELECT COUNT(DISTINCT user_mind_id) FROM behavior_logs WHERE split = ?", (split,))
    users = cur.fetchone()[0]
    print(f"Users: {users}")
    
    cur.execute("SELECT COUNT(*) FROM behavior_logs WHERE split = ?", (split,))
    logs = cur.fetchone()[0]
    print(f"Impression Logs: {logs}")
    
    cur.execute("SELECT SUM(clicked) FROM impression_items WHERE split = ?", (split,))
    clicks = cur.fetchone()[0] or 0
    print(f"Clicks: {clicks}")
    print()

cur.execute("SELECT COUNT(*) FROM mind_users")
total_users = cur.fetchone()[0]
print("OVERALL")
print(f"Total Unique Users in DB: {total_users}")

conn.close()
