import os
import sqlite3
import numpy as np
from typing import Dict
from app.config.settings import settings
from app.services.retrieval.index_loader import ir_store

class PopularityStore:
    _instance = None

    def __init__(self):
        self.article_popularity: Dict[str, float] = {}
        self.popularity_array = None  # Aligned with ir_store row mapping
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> "PopularityStore":
        if cls._instance is None:
            cls._instance = PopularityStore()
        return cls._instance

    def load(self, db_path: str = None):
        if self.is_loaded:
            return

        if db_path is None:
            db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        print("Loading TRAIN-only popularity scores...")
        
        # User-unique click counts for TRAIN split ONLY.
        # History clicks are also TRAIN interactions from before the impression time.
        # However, to be strict and match the training popularity definition:
        # "number of TRAIN users who clicked the article"
        
        # We can scan train behavior_logs, parse impressions_raw for "-1" clicks, 
        # and also parse history_raw. We'll build a set of users for each article.
        
        article_users = {} # aid -> set of int
        uid_map = {}
        next_uid = 0
        
        cur.execute("SELECT user_mind_id, history_raw, impressions_raw FROM behavior_logs WHERE split = 'train'")
        for uid, hist, imp in cur.fetchall():
            if uid not in uid_map:
                uid_map[uid] = next_uid
                next_uid += 1
            u_int = uid_map[uid]
            
            # History clicks
            if hist:
                for aid in hist.split():
                    if aid:
                        if aid not in article_users:
                            article_users[aid] = set()
                        article_users[aid].add(u_int)
            
            # Impression clicks
            if imp:
                for token in imp.split():
                    if token.endswith("-1"):
                        aid = token.split("-")[0]
                        if aid not in article_users:
                            article_users[aid] = set()
                        article_users[aid].add(u_int)

        con.close()

        # Compute raw user-unique counts
        counts = {aid: len(users) for aid, users in article_users.items()}
        
        if not counts:
            max_log_count = 1.0
        else:
            max_count = max(counts.values())
            max_log_count = np.log(1 + max_count)

        self.article_popularity = {}
        
        ir_store.load()
        num_articles = len(ir_store.row_to_mind_id)
        self.popularity_array = np.zeros(num_articles)

        # Compute normalized popularity: log(1 + count) / max(log(1 + count))
        for aid, count in counts.items():
            norm_pop = np.log(1 + count) / max_log_count
            self.article_popularity[aid] = norm_pop
            
            # If the article is in the candidate index, store it in the array
            row_idx = ir_store.mind_id_to_row.get(aid)
            if row_idx is not None:
                self.popularity_array[row_idx] = norm_pop

        self.is_loaded = True
        print(f"Loaded train popularity for {len(self.article_popularity)} articles. Max count: {max_count if counts else 0}")

    def get_popularity_array(self) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        return self.popularity_array
        
    def get_article_popularity(self, mind_id: str) -> float:
        if not self.is_loaded:
            self.load()
        return self.article_popularity.get(mind_id, 0.0)

popularity_store = PopularityStore.get_instance()
