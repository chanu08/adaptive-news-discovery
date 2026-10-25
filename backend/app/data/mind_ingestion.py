"""
MIND Dataset Ingestion Pipeline — Phase 2

Entry point: `ingest_split(split_dir, split_name, db_session)`

Design principles
─────────────────
* Streaming — reads raw files line by line, never loads full dataset to RAM.
* Batch insertion — groups records into configurable batches before committing.
* Idempotent — duplicate MIND IDs are skipped using INSERT OR IGNORE logic
  (SQLite UNIQUE constraint + on_conflict_do_nothing).
* Auditable — writes an IngestionRun record to the DB.
* Safe — malformed rows are logged and counted; ingestion continues.
"""

from __future__ import annotations

import logging
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.mind_parser import (
    ParsedBehaviorRecord,
    ParsedNewsRecord,
    parse_behaviors_file,
    parse_news_file,
)
from app.data.validation import (
    ValidationSummary,
    validate_behavior_record,
    validate_news_record,
)
from app.models.behavior import BehaviorLog, ImpressionItem
from app.models.ingestion_run import IngestionRun
from app.models.mind_user import MindUser
from app.models.news_article import NewsArticle

logger = logging.getLogger(__name__)

# ── Batch sizes ───────────────────────────────────────────────────────────────
NEWS_BATCH_SIZE      = 500
BEHAVIOR_BATCH_SIZE  = 200
ITEM_BATCH_SIZE      = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_split_dir(base_dir: Path, split: str) -> Path:
    """
    Return the directory containing news.tsv for the given split.

    Supports layouts:
      base_dir/news.tsv                         (flat)
      base_dir/<split>/news.tsv                 (split sub-dir)
      base_dir/MINDlarge_<split>/news.tsv       (zip-extracted MIND layout)
    """
    # Flat layout
    if (base_dir / "news.tsv").exists():
        return base_dir

    # Sub-directory layout
    for candidate in [
        base_dir / split,
        base_dir / f"MIND_{split}",
        base_dir / f"MINDlarge_{split}",
        base_dir / f"MINDsmall_{split}",
        base_dir / f"MIND{split}",
    ]:
        if candidate.is_dir() and (candidate / "news.tsv").exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find news.tsv for split '{split}' under {base_dir}. "
        f"Searched: {base_dir} and common sub-directories."
    )


async def _get_existing_news_ids(session: AsyncSession, split: str) -> set[str]:
    """Return the set of MIND article IDs already in the DB for this split."""
    result = await session.execute(
        text("SELECT mind_id FROM news_articles WHERE split = :split"),
        {"split": split},
    )
    return {row[0] for row in result.fetchall()}


async def _get_existing_impression_ids(session: AsyncSession, split: str) -> set[int]:
    """Return set of impression_ids already stored for this split."""
    result = await session.execute(
        text("SELECT impression_id FROM behavior_logs WHERE split = :split"),
        {"split": split},
    )
    return {row[0] for row in result.fetchall()}


async def _get_or_create_user(
    session: AsyncSession,
    user_mind_id: str,
    user_cache: dict[str, int],
) -> int:
    """Return internal user DB id (creates if missing). Uses in-memory cache."""
    if user_mind_id in user_cache:
        return user_cache[user_mind_id]

    result = await session.execute(
        text("SELECT id FROM mind_users WHERE mind_user_id = :uid"),
        {"uid": user_mind_id},
    )
    row = result.fetchone()
    if row:
        user_cache[user_mind_id] = row[0]
        return row[0]

    # Create new user
    user = MindUser(mind_user_id=user_mind_id)
    session.add(user)
    await session.flush()
    user_cache[user_mind_id] = user.id
    return user.id


# ─────────────────────────────────────────────────────────────────────────────
# News ingestion
# ─────────────────────────────────────────────────────────────────────────────

async def _ingest_news(
    news_path: Path,
    split: str,
    session: AsyncSession,
    run: IngestionRun,
) -> set[str]:
    """
    Parse and insert all news articles from news.tsv.

    Returns set of ingested article MIND IDs (for orphan-check in behaviors).
    """
    logger.info("Ingesting news from %s (split=%s)…", news_path, split)

    existing_ids = await _get_existing_news_ids(session, split)
    logger.info("  %d articles already in DB for split=%s — will skip.", len(existing_ids), split)

    val_summary   = ValidationSummary()
    batch: list[NewsArticle] = []
    ingested_ids: set[str]   = set(existing_ids)
    total_read = skipped_dup = skipped_invalid = inserted = 0

    def _flush(force: bool = False) -> None:
        nonlocal inserted
        if not batch:
            return
        if force or len(batch) >= NEWS_BATCH_SIZE:
            session.add_all(batch)
            inserted += len(batch)
            batch.clear()

    for item in parse_news_file(news_path):
        total_read += 1

        if isinstance(item, Exception):
            skipped_invalid += 1
            run.news_skipped += 1
            logger.warning("  Skipped news line (parse error): %s", item)
            continue

        record: ParsedNewsRecord = item

        # Validate
        vr = validate_news_record(record)
        val_summary.record(vr)
        if not vr.valid:
            skipped_invalid += 1
            run.news_skipped += 1
            logger.warning(
                "  Skipped news %s (validation): %s", record.mind_id, "; ".join(vr.issues)
            )
            continue

        # Duplicate check
        if record.mind_id in existing_ids or record.mind_id in ingested_ids:
            skipped_dup += 1
            continue

        ingested_ids.add(record.mind_id)
        batch.append(NewsArticle(
            mind_id=record.mind_id,
            category=record.category,
            subcategory=record.subcategory,
            title=record.title,
            abstract=record.abstract,
            url=record.url,
            title_entities=record.title_entities,
            abstract_entities=record.abstract_entities,
            split=split,
        ))
        _flush()

        if total_read % 10_000 == 0:
            await session.flush()
            logger.info(
                "  News progress: read=%d inserted=%d skipped_dup=%d skipped_invalid=%d",
                total_read, inserted + len(batch), skipped_dup, skipped_invalid,
            )

    _flush(force=True)
    await session.flush()

    run.news_read     += total_read
    run.news_inserted += inserted
    run.news_skipped  += skipped_invalid

    logger.info(
        "  News done: read=%d inserted=%d skip_dup=%d skip_invalid=%d",
        total_read, inserted, skipped_dup, skipped_invalid,
    )
    logger.info("  Validation summary: %s", val_summary)
    return ingested_ids


# ─────────────────────────────────────────────────────────────────────────────
# Behavior ingestion
# ─────────────────────────────────────────────────────────────────────────────

async def _ingest_behaviors(
    behaviors_path: Path,
    split: str,
    session: AsyncSession,
    run: IngestionRun,
    known_article_ids: set[str],
) -> None:
    logger.info("Ingesting behaviors from %s (split=%s)…", behaviors_path, split)
    existing_imp_ids = await _get_existing_impression_ids(session, split)
    logger.info("  %d behavior logs already in DB for split=%s — will skip.", len(existing_imp_ids), split)

    val_summary = ValidationSummary()
    
    # Pre-fetch all user IDs to avoid 300,000 DB roundtrips!
    res = await session.execute(text("SELECT mind_user_id, id FROM mind_users"))
    user_cache = {row[0]: row[1] for row in res.fetchall()}
    new_users_to_add = {}

    log_batch_records = []
    item_batch_tuples = []
    
    total_read = skipped_dup = skipped_invalid = logs_inserted = items_inserted = 0
    
    from sqlalchemy import insert
    from app.models.behavior import BehaviorLog, ImpressionItem
    from app.models.mind_user import MindUser

    async def _flush():
        nonlocal logs_inserted, items_inserted
        if new_users_to_add:
            user_list = list(new_users_to_add.values())
            for i in range(0, len(user_list), 1000):
                await session.execute(insert(MindUser).values(user_list[i:i+1000]))
            await session.flush()
            
            uids = list(new_users_to_add.keys())
            for i in range(0, len(uids), 900):
                chunk = uids[i:i+900]
                in_clause = ",".join(f"'{u}'" for u in chunk)
                res = await session.execute(text(f"SELECT mind_user_id, id FROM mind_users WHERE mind_user_id IN ({in_clause})"))
                for r in res.fetchall():
                    user_cache[r[0]] = r[1]
            new_users_to_add.clear()
            
        if log_batch_records:
            orm_logs = []
            for r in log_batch_records:
                orm_logs.append(BehaviorLog(
                    impression_id=r.impression_id,
                    user_mind_id=r.user_mind_id,
                    time=r.time,
                    history_raw=(" ".join(r.history)) if r.history else None,
                    impressions_raw=(" ".join(f"{i.article_mind_id}-{i.clicked}" for i in r.impressions)) if r.impressions else None,
                    history_length=r.history_length,
                    n_impressions=r.n_impressions,
                    n_clicks=r.n_clicks,
                    split=split,
                ))
            
            for i in range(0, len(orm_logs), 2000):
                session.add_all(orm_logs[i:i+2000])
            await session.flush()
            
            item_batch_tuples = []
            for orm_log, rec in zip(orm_logs, log_batch_records):
                for imp in rec.impressions:
                    item_batch_tuples.append((
                        orm_log.id,
                        imp.article_mind_id,
                        max(imp.clicked, 0),
                        split
                    ))
                    
            logs_inserted += len(orm_logs)
            log_batch_records.clear()
            
        if item_batch_tuples:
            conn = await session.connection()
            raw_conn = await conn.get_raw_connection()
            
            # Using raw aiosqlite connection
            # Wait, raw_conn is an asyncio.Connection. We need to await its executemany.
            # actually conn.executemany from SQLAlchemy DBAPI wrapper is sync in sync engine, but in asyncio it's async?
            # We can just use SQLAlchemy's conn.execute with text() but it's slow.
            # A better way is: driver_connection = raw_conn.driver_connection
            driver_conn = raw_conn.driver_connection
            await driver_conn.executemany(
                "INSERT INTO impression_items (behavior_log_id, article_mind_id, clicked, split) VALUES (?, ?, ?, ?)",
                item_batch_tuples
            )
            items_inserted += len(item_batch_tuples)
            item_batch_tuples.clear()
            
    for item in parse_behaviors_file(behaviors_path):
        total_read += 1
        if isinstance(item, Exception):
            skipped_invalid += 1; run.behaviors_skipped += 1
            continue
            
        record: ParsedBehaviorRecord = item
        vr = validate_behavior_record(record, known_article_ids=known_article_ids)
        val_summary.record(vr)
        if not vr.valid:
            skipped_invalid += 1; run.behaviors_skipped += 1
            continue
            
        if record.impression_id in existing_imp_ids:
            skipped_dup += 1
            continue
            
        existing_imp_ids.add(record.impression_id)
        
        if record.user_mind_id not in user_cache and record.user_mind_id not in new_users_to_add:
            new_users_to_add[record.user_mind_id] = {"mind_user_id": record.user_mind_id}
            
        log_batch_records.append(record)
        
        if len(log_batch_records) >= 10000:
            await _flush()
            
        if total_read % 20_000 == 0:
            logger.info("  Behavior progress: read=%d logs=%d items=%d", total_read, logs_inserted, items_inserted)

    await _flush()
    run.behaviors_read += total_read
    run.behaviors_inserted += logs_inserted
    run.items_inserted += items_inserted
    run.users_inserted += len(user_cache)
    logger.info("  Behaviors done: read=%d logs=%d items=%d", total_read, logs_inserted, items_inserted)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_split(
    split_dir: Path,
    split: str,
    session: AsyncSession,
) -> IngestionRun:
    """
    Ingest one MIND split (train / dev / test) from a directory or zip.

    Parameters
    ----------
    split_dir : Path to directory containing news.tsv and behaviors.tsv,
                OR path to a .zip file that will be auto-extracted.
    split     : "train" | "dev" | "test"
    session   : Async SQLAlchemy session (caller manages commit)

    Returns
    -------
    IngestionRun record (already flushed into session).
    """
    split_dir = Path(split_dir)
    run_id    = str(uuid.uuid4())
    start_ts  = time.monotonic()

    # Auto-extract zip if needed
    if split_dir.is_file() and split_dir.suffix == ".zip":
        extract_dir = split_dir.parent / split_dir.stem
        logger.info("Extracting %s → %s …", split_dir, extract_dir)
        with zipfile.ZipFile(split_dir) as zf:
            zf.extractall(extract_dir)
        split_dir = extract_dir

    # Locate actual data directory
    data_dir   = _find_split_dir(split_dir, split)
    news_path  = data_dir / "news.tsv"
    beh_path   = data_dir / "behaviors.tsv"

    logger.info(
        "=== MIND ingestion START  split=%s  run_id=%s ===", split, run_id
    )
    logger.info("  news.tsv      : %s (%.1f MB)", news_path, news_path.stat().st_size / 1e6)
    if beh_path.exists():
        logger.info(
            "  behaviors.tsv : %s (%.1f MB)", beh_path, beh_path.stat().st_size / 1e6
        )

    # Create ingestion run record
    run = IngestionRun(run_id=run_id, split=split, status="running")
    session.add(run)
    await session.flush()

    try:
        # 1. Ingest news
        known_ids = await _ingest_news(news_path, split, session, run)

        # 2. Ingest behaviors (if present — test set may lack behaviors)
        if beh_path.exists():
            await _ingest_behaviors(beh_path, split, session, run, known_ids)
        else:
            logger.warning("  behaviors.tsv not found for split=%s — skipping.", split)

        elapsed = time.monotonic() - start_ts
        run.status          = "success"
        run.finished_at     = datetime.utcnow()
        run.duration_seconds = elapsed

        logger.info(
            "=== MIND ingestion DONE  split=%s  elapsed=%.1fs  "
            "news=%d  behaviors=%d  items=%d ===",
            split, elapsed, run.news_inserted, run.behaviors_inserted, run.items_inserted,
        )

    except Exception as exc:
        run.status    = "error"
        run.error_log = str(exc)[:4000]
        run.finished_at = datetime.utcnow()
        logger.exception("Ingestion failed for split=%s: %s", split, exc)
        raise

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    CLI usage (from backend/):
        python -m app.data.mind_ingestion <split_dir> <split>

    Example:
        python -m app.data.mind_ingestion ../data/raw/mind/large train
        python -m app.data.mind_ingestion ../data/raw/mind/large/dev dev
    """
    import asyncio
    import sys

    async def _main() -> None:
        if len(sys.argv) < 3:
            print("Usage: python -m app.data.mind_ingestion <split_dir> <split>")
            sys.exit(1)

        split_dir_arg = Path(sys.argv[1])
        split_arg     = sys.argv[2]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Bootstrap DB
        from app.database.connection import AsyncSessionLocal, init_db
        await init_db()

        async with AsyncSessionLocal() as session:
            run = await ingest_split(split_dir_arg, split_arg, session)
            await session.commit()

        print("\n" + "=" * 60)
        print(f"Ingestion complete — split={split_arg}")
        print(f"  Status     : {run.status}")
        print(f"  Duration   : {run.duration_seconds:.1f}s")
        print(f"  News       : read={run.news_read}  inserted={run.news_inserted}  skipped={run.news_skipped}")
        print(f"  Behaviors  : read={run.behaviors_read}  inserted={run.behaviors_inserted}  skipped={run.behaviors_skipped}")
        print(f"  Items      : {run.items_inserted}")
        print(f"  Users      : {run.users_inserted}")
        print("=" * 60)

    asyncio.run(_main())
