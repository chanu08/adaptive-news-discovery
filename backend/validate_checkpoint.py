import json
import sqlite3

def validate_checkpoint():
    file_path = 'data/evaluation/phase15/raw_session_results.jsonl'
    lines = 0
    bad = 0
    uids = set()
    has_sess = 0
    duplicate = 0
    
    with open(file_path, 'r') as f:
        for l in f:
            lines += 1
            if not l.strip(): continue
            try:
                j = json.loads(l)
                uid = j['user_mind_id']
                if uid in uids:
                    duplicate += 1
                uids.add(uid)
                if j.get('has_sessions'): 
                    has_sess += 1
            except Exception:
                bad += 1
                
    print(f"Total records: {lines}")
    print(f"Valid unique users: {len(uids)}")
    print(f"Users with sessions: {has_sess}")
    print(f"Duplicate records: {duplicate}")
    print(f"Malformed records: {bad}")

def check_db_integrity():
    con = sqlite3.connect('./data/news_discovery.db')
    cur = con.cursor()
    
    counts = {}
    for table in ['canonical_articles', 'behavior_logs', 'impression_items', 'feedback_events']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cur.fetchone()[0]
        
    con.close()
    
    print("\nDatabase Integrity:")
    for k, v in counts.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    validate_checkpoint()
    check_db_integrity()
