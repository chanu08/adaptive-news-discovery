"""
Dataset Loader & Inspection — Phase 11
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

Handles:
- MIND schema inspection and split verification
- Detection of missing labels in MIND test split
- Deterministic, reproducible user sampling with exclusion reporting
- Strict leak-free temporal data reconstruction (Train historical -> Dev ground truth)
"""

import os
import random
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple, Optional

class MINDDatasetInspector:
    """
    Inspects the actual MIND dataset schema and splits in the database.
    Documents available fields, timestamps, and label availability.
    """

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)

    def inspect(self) -> Dict[str, Any]:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        # 1. Behavior logs by split
        cur.execute("""
            SELECT split, count(*), min(time), max(time), sum(n_clicks), sum(n_impressions)
            FROM behavior_logs
            GROUP BY split
        """)
        behavior_splits = {}
        impression_splits = {}
        for r in cur.fetchall():
            sp = r[0]
            b_cnt = r[1]
            min_t = r[2]
            max_t = r[3]
            clicks = r[4] or 0
            imps = r[5] or 0
            behavior_splits[sp] = {
                "count": b_cnt,
                "min_time": min_t,
                "max_time": max_t,
                "total_clicks": clicks,
                "total_impressions": imps
            }
            impression_splits[sp] = {
                "count": imps,
                "total_clicks": clicks
            }

        # Also get distinct users per split (train, dev, test)
        for sp in ["dev", "train", "test"]:
            u_cnt = cur.execute("SELECT count(distinct user_mind_id) FROM behavior_logs WHERE split=?", (sp,)).fetchone()[0]
            if sp in behavior_splits:
                behavior_splits[sp]["unique_users"] = u_cnt

        # 3. Article splits
        cur.execute("""
            SELECT split, count(*), count(distinct article_id)
            FROM article_splits
            GROUP BY split
        """)
        article_splits = {}
        for r in cur.fetchall():
            article_splits[r[0]] = {
                "count": r[1],
                "distinct_articles": r[2]
            }

        # 4. Total canonical articles
        total_articles = cur.execute("SELECT count(*) FROM news_articles").fetchone()[0]

        # 5. Total MIND users
        total_mind_users = cur.execute("SELECT count(*) FROM mind_users").fetchone()[0]

        # 6. Check test split label status
        test_clicks = behavior_splits.get("test", {}).get("total_clicks", 0)
        test_suitable = test_clicks > 0
        test_status_message = (
            "Test split is suitable for labeled evaluation."
            if test_suitable
            else "Test split is not suitable for direct labeled offline ranking metrics because click labels are unavailable."
        )

        con.close()

        inspection_report = {
            "database_path": self.db_path,
            "total_canonical_articles": total_articles,
            "total_mind_users": total_mind_users,
            "behavior_logs_by_split": behavior_splits,
            "impression_items_by_split": impression_splits,
            "article_splits": article_splits,
            "test_split_labeled": test_suitable,
            "test_split_statement": test_status_message,
            "timestamp_availability": {
                "behavior_logs": "Behavior impression timestamps available as 'MM/DD/YYYY HH:MM:SS AM/PM' string.",
                "history_field": "Historical clicks stored as space-delimited string of article IDs without individual click timestamps.",
                "leakage_safeguard": "Training clicks reconstructed strictly from train split; dev clicks strictly held-out as evaluation targets."
            }
        }
        return inspection_report


class EvaluationCohort:
    """
    Manages deterministic sampling of evaluation users and constructs
    leak-free training interaction sequences and held-out evaluation ground truth.
    """

    def __init__(self, db_path: str, max_users: int = 100, seed: int = 42, split: str = "dev"):
        self.db_path = os.path.abspath(db_path)
        self.max_users = max_users
        self.seed = seed
        self.split = split
        self.sampling_stats: Dict[str, Any] = {}
        self.users_data: List[Dict[str, Any]] = []

    def load_cohort(self) -> List[Dict[str, Any]]:
        """
        Loads and prepares the evaluation user cohort deterministically.
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        # Step 1: Population sizing
        total_dev_users = cur.execute("SELECT count(distinct user_mind_id) FROM behavior_logs WHERE split='dev'").fetchone()[0]
        
        # Step 2: Eligible users
        # Must have:
        # 1. Dev behavior logs with at least one click (n_clicks > 0)
        # 2. Train behavior logs (to construct past profile)
        print("Identifying eligible evaluation users (train behavior + dev clicks)...")
        cur.execute("""
            SELECT distinct b_dev.user_mind_id
            FROM behavior_logs b_dev
            WHERE b_dev.split = 'dev' AND b_dev.n_clicks > 0
              AND EXISTS (
                  SELECT 1 FROM behavior_logs b_tr
                  WHERE b_tr.user_mind_id = b_dev.user_mind_id
                    AND b_tr.split = 'train'
              )
        """)
        eligible_user_ids = [r[0] for r in cur.fetchall()]
        eligible_count = len(eligible_user_ids)
        excluded_count = total_dev_users - eligible_count

        # Deterministic sampling
        rng = random.Random(self.seed)
        eligible_user_ids.sort()  # Ensure identical base ordering
        sample_size = min(self.max_users, eligible_count) if self.max_users is not None else eligible_count
        sampled_user_ids = rng.sample(eligible_user_ids, sample_size)
        sampled_user_ids.sort()

        self.sampling_stats = {
            "total_dev_users": total_dev_users,
            "eligible_users": eligible_count,
            "sampled_users": sample_size,
            "excluded_users": excluded_count,
            "exclusion_reasons": {
                "no_train_history": total_dev_users - cur.execute("""
                    SELECT count(distinct b_dev.user_mind_id)
                    FROM behavior_logs b_dev
                    WHERE b_dev.split = 'dev'
                      AND EXISTS (SELECT 1 FROM behavior_logs b_tr WHERE b_tr.user_mind_id = b_dev.user_mind_id AND b_tr.split = 'train')
                """).fetchone()[0],
                "no_dev_clicks": total_dev_users - cur.execute("""
                    SELECT count(distinct user_mind_id) FROM behavior_logs WHERE split = 'dev' AND n_clicks > 0
                """).fetchone()[0]
            },
            "seed": self.seed,
            "split": self.split
        }

        print(f"Sampled {sample_size} users from {eligible_count} eligible users (seed={self.seed}).")

        # Step 3: Reconstruct leak-free user states
        print("Reconstructing leak-free user histories and held-out ground truth...")
        self.users_data = []

        for uid in sampled_user_ids:
            # A. Training sequence (STRICTLY TRAIN)
            # Earliest train log history_raw
            cur.execute("""
                SELECT history_raw 
                FROM behavior_logs 
                WHERE user_mind_id = ? AND split = 'train' 
                ORDER BY id ASC LIMIT 1
            """, (uid,))
            hist_row = cur.fetchone()
            hist_raw = hist_row[0] if hist_row and hist_row[0] else ""
            hist_clicks = [aid for aid in hist_raw.split() if aid]

            # Train impression clicks with timestamps
            cur.execute("""
                SELECT i.article_mind_id, b.time, i.id
                FROM impression_items i
                JOIN behavior_logs b ON i.behavior_log_id = b.id
                WHERE b.user_mind_id = ? AND b.split = 'train' AND i.clicked = 1
                ORDER BY i.id ASC
            """, (uid,))
            train_imp_clicks = []
            for aid, dt_str, iid in cur.fetchall():
                dt = None
                if dt_str:
                    try:
                        dt = datetime.strptime(dt_str, "%m/%d/%Y %I:%M:%S %p")
                    except Exception:
                        try:
                            dt = datetime.fromisoformat(dt_str)
                        except Exception:
                            dt = None
                train_imp_clicks.append((aid, dt, iid))

            train_imp_clicks.sort(key=lambda x: (x[1] if x[1] else datetime.min, x[2]))

            # Train sequence: (aid, timestamp)
            train_sequence = [(aid, None) for aid in hist_clicks] + [(aid, dt) for aid, dt, _ in train_imp_clicks]
            train_clicked_aids = [aid for aid, _ in train_sequence]

            # B. Dev Ground Truth Clicks
            cur.execute("""
                SELECT distinct i.article_mind_id
                FROM impression_items i
                JOIN behavior_logs b ON i.behavior_log_id = b.id
                WHERE b.user_mind_id = ? AND b.split = 'dev' AND i.clicked = 1
            """, (uid,))
            dev_ground_truth = set(r[0] for r in cur.fetchall())

            # C. Dev Impression Sessions (for impression-level protocol)
            cur.execute("""
                SELECT b.id, b.time, b.impressions_raw
                FROM behavior_logs b
                WHERE b.user_mind_id = ? AND b.split = 'dev'
                ORDER BY b.id ASC
            """, (uid,))
            dev_sessions = []
            for b_id, b_time, imp_raw in cur.fetchall():
                if not imp_raw:
                    continue
                impressed_aids = []
                clicked_aids = set()
                for token in imp_raw.split():
                    parts = token.split("-")
                    if len(parts) == 2:
                        aid, label = parts[0], parts[1]
                        impressed_aids.append(aid)
                        if label == "1":
                            clicked_aids.add(aid)
                if impressed_aids:
                    dev_sessions.append({
                        "behavior_log_id": b_id,
                        "time": b_time,
                        "impressed_articles": impressed_aids,
                        "clicked_articles": clicked_aids
                    })

            self.users_data.append({
                "user_id": uid,
                "train_sequence": train_sequence,
                "train_clicked_aids": train_clicked_aids,
                "dev_ground_truth": dev_ground_truth,
                "dev_sessions": dev_sessions
            })

        con.close()
        return self.users_data
