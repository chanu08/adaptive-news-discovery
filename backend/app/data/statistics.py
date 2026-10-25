"""
Dataset Statistics Generator — Phase 2

Computes all statistics directly from the database (never from in-memory data).
All counts are real — never hardcoded.

Usage:
    stats = await compute_dataset_stats(session)
    save_stats(stats, Path("docs/mind-statistics.json"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stats containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsStats:
    total_articles: int = 0
    unique_categories: int = 0
    articles_per_category: dict[str, int] = field(default_factory=dict)
    articles_per_split: dict[str, int] = field(default_factory=dict)
    articles_with_abstract: int = 0
    articles_without_abstract: int = 0
    articles_with_url: int = 0
    articles_without_url: int = 0
    date_ingested_from: Optional[str] = None
    date_ingested_to: Optional[str] = None


@dataclass
class UserStats:
    total_users: int = 0
    users_with_clicks: int = 0
    users_without_clicks: int = 0
    avg_clicks_per_user: float = 0.0
    min_clicks: int = 0
    max_clicks: int = 0
    avg_impressions_per_user: float = 0.0


@dataclass
class BehaviorStats:
    total_behavior_logs: int = 0
    total_impression_items: int = 0
    total_clicks: int = 0
    total_non_clicks: int = 0
    click_through_rate: float = 0.0
    behavior_logs_per_split: dict[str, int] = field(default_factory=dict)
    earliest_timestamp: Optional[str] = None
    latest_timestamp: Optional[str] = None
    avg_impressions_per_log: float = 0.0
    avg_clicks_per_log: float = 0.0
    avg_history_length: float = 0.0


@dataclass
class RelationshipStats:
    articles_with_impressions: int = 0
    articles_without_impressions: int = 0
    orphan_impression_article_ids: int = 0  # items whose article not in news_articles
    ingestion_runs: int = 0
    last_ingestion_status: Optional[str] = None
    last_ingestion_at: Optional[str] = None


@dataclass
class DatasetStats:
    generated_at: str = ""
    news: NewsStats = field(default_factory=NewsStats)
    users: UserStats = field(default_factory=UserStats)
    behaviors: BehaviorStats = field(default_factory=BehaviorStats)
    relationships: RelationshipStats = field(default_factory=RelationshipStats)


# ─────────────────────────────────────────────────────────────────────────────
# DB query helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _scalar(session: AsyncSession, sql: str, params: dict | None = None) -> int | float | None:
    result = await session.execute(text(sql), params or {})
    row = result.fetchone()
    return row[0] if row else None


async def _rows(session: AsyncSession, sql: str, params: dict | None = None) -> list:
    result = await session.execute(text(sql), params or {})
    return result.fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

async def compute_dataset_stats(session: AsyncSession) -> DatasetStats:
    """
    Compute all statistics from the SQLite database.
    All values are real — no hardcoding.
    """
    stats = DatasetStats(generated_at=datetime.utcnow().isoformat())

    # ── News ──────────────────────────────────────────────────────────────────
    news = stats.news

    news.total_articles = (await _scalar(
        session, "SELECT COUNT(*) FROM news_articles"
    )) or 0

    rows = await _rows(
        session,
        "SELECT category, COUNT(*) AS cnt FROM news_articles GROUP BY category ORDER BY cnt DESC"
    )
    news.articles_per_category = {r[0]: r[1] for r in rows}
    news.unique_categories     = len(news.articles_per_category)

    rows = await _rows(
        session,
        "SELECT split, COUNT(*) AS cnt FROM article_splits GROUP BY split"
    )
    news.articles_per_split = {r[0]: r[1] for r in rows}

    news.articles_with_abstract = (await _scalar(
        session,
        "SELECT COUNT(*) FROM news_articles WHERE abstract IS NOT NULL AND abstract != ''"
    )) or 0
    news.articles_without_abstract = news.total_articles - news.articles_with_abstract

    news.articles_with_url = (await _scalar(
        session,
        "SELECT COUNT(*) FROM news_articles WHERE url IS NOT NULL AND url != ''"
    )) or 0
    news.articles_without_url = news.total_articles - news.articles_with_url

    row = await _rows(
        session,
        "SELECT MIN(ingested_at), MAX(ingested_at) FROM news_articles"
    )
    if row and row[0][0]:
        news.date_ingested_from = str(row[0][0])
        news.date_ingested_to   = str(row[0][1])

    # ── Users ─────────────────────────────────────────────────────────────────
    users = stats.users

    users.total_users = (await _scalar(
        session, "SELECT COUNT(*) FROM mind_users"
    )) or 0

    users.users_with_clicks = (await _scalar(
        session, "SELECT COUNT(*) FROM mind_users WHERE total_clicks > 0"
    )) or 0
    users.users_without_clicks = users.total_users - users.users_with_clicks

    agg = await _rows(
        session,
        """
        SELECT
            AVG(total_clicks),
            MIN(total_clicks),
            MAX(total_clicks),
            AVG(total_impressions)
        FROM mind_users
        """
    )
    if agg and agg[0][0] is not None:
        users.avg_clicks_per_user      = round(float(agg[0][0]), 2)
        users.min_clicks               = int(agg[0][1])
        users.max_clicks               = int(agg[0][2])
        users.avg_impressions_per_user = round(float(agg[0][3]), 2)

    # ── Behaviors ─────────────────────────────────────────────────────────────
    beh = stats.behaviors

    beh.total_behavior_logs = (await _scalar(
        session, "SELECT COUNT(*) FROM behavior_logs"
    )) or 0

    beh.total_impression_items = (await _scalar(
        session, "SELECT COUNT(*) FROM impression_items"
    )) or 0

    beh.total_clicks = (await _scalar(
        session, "SELECT COUNT(*) FROM impression_items WHERE clicked = 1"
    )) or 0
    beh.total_non_clicks = (await _scalar(
        session, "SELECT COUNT(*) FROM impression_items WHERE clicked = 0"
    )) or 0

    if beh.total_impression_items > 0:
        beh.click_through_rate = round(
            beh.total_clicks / beh.total_impression_items, 4
        )

    rows = await _rows(
        session,
        "SELECT split, COUNT(*) FROM behavior_logs GROUP BY split"
    )
    beh.behavior_logs_per_split = {r[0]: r[1] for r in rows}

    ts_row = await _rows(
        session,
        "SELECT MIN(time), MAX(time) FROM behavior_logs WHERE time IS NOT NULL"
    )
    if ts_row and ts_row[0][0]:
        beh.earliest_timestamp = str(ts_row[0][0])
        beh.latest_timestamp   = str(ts_row[0][1])

    agg2 = await _rows(
        session,
        """
        SELECT
            AVG(n_impressions),
            AVG(n_clicks),
            AVG(history_length)
        FROM behavior_logs
        """
    )
    if agg2 and agg2[0][0] is not None:
        beh.avg_impressions_per_log = round(float(agg2[0][0]), 2)
        beh.avg_clicks_per_log      = round(float(agg2[0][1]), 2)
        beh.avg_history_length      = round(float(agg2[0][2]), 2)

    # ── Relationships ─────────────────────────────────────────────────────────
    rel = stats.relationships

    rel.articles_with_impressions = (await _scalar(
        session,
        """
        SELECT COUNT(DISTINCT article_mind_id)
        FROM impression_items
        WHERE article_mind_id IN (SELECT mind_id FROM news_articles)
        """
    )) or 0

    rel.articles_without_impressions = (
        news.total_articles - rel.articles_with_impressions
    )

    rel.orphan_impression_article_ids = (await _scalar(
        session,
        """
        SELECT COUNT(DISTINCT article_mind_id)
        FROM impression_items
        WHERE article_mind_id NOT IN (SELECT mind_id FROM news_articles)
        """
    )) or 0

    rel.ingestion_runs = (await _scalar(
        session, "SELECT COUNT(*) FROM ingestion_runs"
    )) or 0

    last_run = await _rows(
        session,
        "SELECT status, finished_at FROM ingestion_runs ORDER BY id DESC LIMIT 1"
    )
    if last_run:
        rel.last_ingestion_status = last_run[0][0]
        rel.last_ingestion_at     = str(last_run[0][1]) if last_run[0][1] else None

    logger.info("Dataset statistics computed: %d articles, %d users, %d behavior logs",
                news.total_articles, users.total_users, beh.total_behavior_logs)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_stats(stats: DatasetStats, json_path: Path) -> None:
    """Save DatasetStats to a JSON file."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, indent=2, default=str)
    logger.info("Statistics saved to %s", json_path)


def stats_to_markdown(stats: DatasetStats) -> str:
    """Render DatasetStats as a markdown report."""
    n = stats.news
    u = stats.users
    b = stats.behaviors
    r = stats.relationships

    lines: list[str] = [
        "# MIND Dataset Statistics",
        f"\n_Generated at: {stats.generated_at}_\n",

        "## News Articles",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total articles | **{n.total_articles:,}** |",
        f"| Unique categories | {n.unique_categories} |",
        f"| Articles with abstract | {n.articles_with_abstract:,} |",
        f"| Articles without abstract | {n.articles_without_abstract:,} |",
        f"| Articles with URL | {n.articles_with_url:,} |",
        f"| Date ingested (from) | {n.date_ingested_from} |",
        f"| Date ingested (to) | {n.date_ingested_to} |",

        "\n### Articles per Category",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for cat, cnt in sorted(n.articles_per_category.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {cnt:,} |")

    lines += [
        "\n### Articles per Split",
        "| Split | Count |",
        "|-------|-------|",
    ]
    for sp, cnt in n.articles_per_split.items():
        lines.append(f"| {sp} | {cnt:,} |")

    lines += [
        "\n## Users",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total users | **{u.total_users:,}** |",
        f"| Users with clicks | {u.users_with_clicks:,} |",
        f"| Users without clicks | {u.users_without_clicks:,} |",
        f"| Avg clicks per user | {u.avg_clicks_per_user} |",
        f"| Min clicks | {u.min_clicks} |",
        f"| Max clicks | {u.max_clicks} |",
        f"| Avg impressions per user | {u.avg_impressions_per_user} |",

        "\n## Behavior Logs",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total behavior logs | **{b.total_behavior_logs:,}** |",
        f"| Total impression items | {b.total_impression_items:,} |",
        f"| Total clicks | {b.total_clicks:,} |",
        f"| Total non-clicks | {b.total_non_clicks:,} |",
        f"| Click-through rate | {b.click_through_rate:.2%} |",
        f"| Earliest timestamp | {b.earliest_timestamp} |",
        f"| Latest timestamp | {b.latest_timestamp} |",
        f"| Avg impressions per log | {b.avg_impressions_per_log} |",
        f"| Avg clicks per log | {b.avg_clicks_per_log} |",
        f"| Avg history length | {b.avg_history_length} |",

        "\n### Behavior Logs per Split",
        "| Split | Logs |",
        "|-------|------|",
    ]
    for sp, cnt in b.behavior_logs_per_split.items():
        lines.append(f"| {sp} | {cnt:,} |")

    lines += [
        "\n## Relationships",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Articles with impressions | {r.articles_with_impressions:,} |",
        f"| Articles without impressions | {r.articles_without_impressions:,} |",
        f"| Orphan impression article IDs | {r.orphan_impression_article_ids:,} |",
        f"| Total ingestion runs | {r.ingestion_runs} |",
        f"| Last ingestion status | {r.last_ingestion_status} |",
        f"| Last ingestion at | {r.last_ingestion_at} |",
    ]
    return "\n".join(lines)
