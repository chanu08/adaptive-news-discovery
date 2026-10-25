import sqlite3
import time
from pathlib import Path
import os

def ingest():
    db_path = "data/news_discovery.db"
    if not os.path.exists("data"):
        os.makedirs("data")
        
    import sys
    sys.path.append(os.path.dirname(__file__))
    from app.database.connection import engine, Base
    from app.models.mind_user import MindUser
    from app.models.news_article import NewsArticle, ArticleSplit
    from app.models.behavior import BehaviorLog, ImpressionItem
    import asyncio
    
    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
    asyncio.run(init_db())
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    
    cur = conn.cursor()
    
    cur.execute("SELECT mind_user_id, id FROM mind_users")
    user_cache = {row[0]: row[1] for row in cur.fetchall()}
    
    cur.execute("SELECT mind_id, id FROM news_articles")
    article_cache = {row[0]: row[1] for row in cur.fetchall()}
    
    splits = ["dev", "train", "test"]
    base_dir = Path("../data/extracted/mind")
    
    for split in splits:
        split_dir = base_dir / split
        data_dir = split_dir
        for cand in [split_dir, split_dir / f"MINDlarge_{split}"]:
            if cand.is_dir() and (cand / "news.tsv").exists():
                data_dir = cand
                break
                
        print(f"Ingesting {split} from {data_dir}")
        news_file = data_dir / "news.tsv"
        beh_file = data_dir / "behaviors.tsv"
        
        cur.execute("SELECT news_articles.mind_id FROM article_splits JOIN news_articles ON article_splits.article_id = news_articles.id WHERE article_splits.split = ?", (split,))
        existing_splits = {r[0] for r in cur.fetchall()}
        
        news_batch = []
        split_batch = []
        
        def flush_news():
            if news_batch:
                cur.executemany("INSERT INTO news_articles (mind_id, category, subcategory, title, abstract, url, title_entities, abstract_entities, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", news_batch)
                conn.commit()
                # Update article_cache with newly inserted
                mind_ids = [row[0] for row in news_batch]
                # Chunk the select if it's too large, but 10000 is fine
                cur.execute("SELECT mind_id, id FROM news_articles WHERE mind_id IN (" + ",".join("?" for _ in mind_ids) + ")", mind_ids)
                for r in cur.fetchall():
                    article_cache[r[0]] = r[1]
                news_batch.clear()
            if split_batch:
                # Map mind_ids to internal article_ids
                split_insert = [(article_cache[m_id], split) for m_id in split_batch]
                cur.executemany("INSERT INTO article_splits (article_id, split) VALUES (?, ?)", split_insert)
                conn.commit()
                split_batch.clear()
        
        if news_file.exists():
            with open(news_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) < 4: continue
                    mind_id = parts[0]
                    
                    if mind_id not in article_cache:
                        cat = parts[1]
                        subcat = parts[2]
                        title = parts[3]
                        abs_ = parts[4] if len(parts) > 4 else None
                        url = parts[5] if len(parts) > 5 else None
                        t_ent = parts[6] if len(parts) > 6 else None
                        a_ent = parts[7] if len(parts) > 7 else None
                        
                        news_batch.append((mind_id, cat, subcat, title, abs_, url, t_ent, a_ent))
                        # Temporarily mark as seen so we don't insert duplicate mind_ids in the same batch
                        article_cache[mind_id] = -1 
                        
                    if mind_id not in existing_splits:
                        split_batch.append(mind_id)
                        existing_splits.add(mind_id)
                        
                    if len(news_batch) >= 10000 or len(split_batch) >= 10000:
                        flush_news()
                        
            flush_news()
            print(f"  News for {split} inserted.")
        
        if not beh_file.exists():
            continue
            
        cur.execute("SELECT impression_id FROM behavior_logs WHERE split = ?", (split,))
        existing_beh = {r[0] for r in cur.fetchall()}
        
        beh_batch = []
        item_batch = []
        new_users = {}
        
        def flush_behaviors():
            if new_users:
                cur.executemany("INSERT INTO mind_users (mind_user_id, total_impressions, total_clicks, ingested_at) VALUES (?, 0, 0, CURRENT_TIMESTAMP)", [(u,) for u in new_users.keys()])
                conn.commit()
                cur.execute("SELECT mind_user_id, id FROM mind_users WHERE mind_user_id IN (" + ",".join("?" for _ in new_users) + ")", list(new_users.keys()))
                for r in cur.fetchall():
                    user_cache[r[0]] = r[1]
                new_users.clear()
            if beh_batch:
                cur.executemany("INSERT INTO behavior_logs (impression_id, user_mind_id, time, history_raw, impressions_raw, history_length, n_impressions, n_clicks, split, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", beh_batch)
                conn.commit()
                cur.execute("SELECT id, impression_id FROM behavior_logs ORDER BY id DESC LIMIT ?", (len(beh_batch),))
                id_map = {r[1]: r[0] for r in cur.fetchall()}
                for row in beh_batch:
                    imp_id_val = row[0]
                    imp_raw = row[4]
                    db_log_id = id_map[imp_id_val]
                    if imp_raw:
                        for item in imp_raw.split():
                            if '-' in item:
                                a, c = item.rsplit('-', 1)
                                item_batch.append((db_log_id, a, int(c), split))
                            else: # Test set doesn't have click labels
                                item_batch.append((db_log_id, item, 0, split))
                cur.executemany("INSERT INTO impression_items (behavior_log_id, article_mind_id, clicked, split) VALUES (?, ?, ?, ?)", item_batch)
                conn.commit()
                item_batch.clear()
                beh_batch.clear()

        with open(beh_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) < 4: continue
                
                imp_id = int(parts[0])
                if imp_id in existing_beh: continue
                
                uid = parts[1]
                if uid not in user_cache and uid not in new_users:
                    new_users[uid] = len(user_cache) + len(new_users) + 1
                    
                time_str = parts[2]
                history = parts[3]
                impressions = parts[4] if len(parts) > 4 else ""
                
                clicks = impressions.count('-1') if impressions else 0
                
                beh_batch.append((imp_id, uid, time_str, history, impressions, len(history.split()) if history else 0, len(impressions.split()), clicks, split))
                existing_beh.add(imp_id)
                
                if len(new_users) >= 10000 or len(beh_batch) >= 10000:
                    flush_behaviors()
                    
        flush_behaviors()
        print(f"  Behaviors for {split} inserted.")
        
    print("Done!")

if __name__ == "__main__":
    start = time.time()
    ingest()
    print("Total time:", time.time() - start)
