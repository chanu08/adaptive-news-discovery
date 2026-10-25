"""
Dataset Admin Routes — Phase 2

GET  /api/dataset/stats    — return live dataset statistics from DB
POST /api/dataset/ingest   — trigger MIND ingestion for a split
GET  /api/dataset/runs     — list all ingestion runs
GET  /api/dataset/runs/{run_id} — single run details
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.mind_ingestion import ingest_split
from app.data.statistics import compute_dataset_stats, save_stats, stats_to_markdown
from app.database.connection import AsyncSessionLocal, get_db
from app.models.ingestion_run import IngestionRun
from app.schemas.dataset import (
    DatasetStatsResponse,
    IngestionRequest,
    IngestionRunResponse,
)

router   = APIRouter()
logger   = logging.getLogger(__name__)

# Path where statistics JSON is saved (relative to repo root)
STATS_JSON_PATH = Path("docs/mind-statistics.json")
STATS_MD_PATH   = Path("docs/mind-statistics.md")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(db: AsyncSession = Depends(get_db)):
    """
    Compute and return live dataset statistics directly from the database.
    All values are real — never hardcoded.
    """
    stats = await compute_dataset_stats(db)

    # Persist to docs/ for reference
    try:
        save_stats(stats, STATS_JSON_PATH)
        STATS_MD_PATH.write_text(stats_to_markdown(stats), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save statistics to disk: %s", e)

    return DatasetStatsResponse(
        generated_at=stats.generated_at,
        total_articles=stats.news.total_articles,
        total_users=stats.users.total_users,
        total_behavior_logs=stats.behaviors.total_behavior_logs,
        total_impression_items=stats.behaviors.total_impression_items,
        total_clicks=stats.behaviors.total_clicks,
        click_through_rate=stats.behaviors.click_through_rate,
        unique_categories=stats.news.unique_categories,
        articles_per_category=stats.news.articles_per_category,
        articles_per_split=stats.news.articles_per_split,
        behavior_logs_per_split=stats.behaviors.behavior_logs_per_split,
        articles_with_abstract=stats.news.articles_with_abstract,
        articles_without_abstract=stats.news.articles_without_abstract,
        earliest_behavior=stats.behaviors.earliest_timestamp,
        latest_behavior=stats.behaviors.latest_timestamp,
        avg_impressions_per_log=stats.behaviors.avg_impressions_per_log,
        avg_history_length=stats.behaviors.avg_history_length,
        orphan_impression_article_ids=stats.relationships.orphan_impression_article_ids,
        last_ingestion_status=stats.relationships.last_ingestion_status,
        last_ingestion_at=stats.relationships.last_ingestion_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion trigger
# ─────────────────────────────────────────────────────────────────────────────

async def _run_ingestion_task(split_dir: str, split: str) -> None:
    """Background task: runs ingestion in its own DB session."""
    logger.info("Background ingestion starting: split_dir=%s split=%s", split_dir, split)
    async with AsyncSessionLocal() as session:
        try:
            await ingest_split(Path(split_dir), split, session)
            await session.commit()
            logger.info("Background ingestion committed for split=%s", split)
        except Exception as exc:
            await session.rollback()
            logger.exception("Background ingestion failed for split=%s: %s", split, exc)


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger MIND data ingestion as a background task.

    Body:
      split_dir: path to directory containing news.tsv & behaviors.tsv
                 (or path to a .zip file)
      split:     "train" | "dev" | "test"

    The endpoint returns immediately with 202 Accepted.
    Monitor progress at GET /api/dataset/runs.
    """
    split_path = Path(request.split_dir)

    # Basic existence check before handing off
    if not split_path.exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"split_dir does not exist: {request.split_dir}",
        )

    allowed_splits = {"train", "dev", "test"}
    if request.split not in allowed_splits:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"split must be one of {allowed_splits}, got: {request.split!r}",
        )

    background_tasks.add_task(
        _run_ingestion_task, str(split_path.resolve()), request.split
    )

    return {
        "message": f"Ingestion started for split='{request.split}'.",
        "split_dir": str(split_path.resolve()),
        "note": "Monitor progress at GET /api/dataset/runs",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion run history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=list[IngestionRunResponse])
async def list_ingestion_runs(db: AsyncSession = Depends(get_db)):
    """List all past and running ingestion runs, newest first."""
    result = await db.execute(
        select(IngestionRun).order_by(IngestionRun.id.desc()).limit(50)
    )
    return result.scalars().all()


@router.get("/runs/{run_id}", response_model=IngestionRunResponse)
async def get_ingestion_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get details of a single ingestion run."""
    result = await db.execute(
        select(IngestionRun).where(IngestionRun.run_id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion run '{run_id}' not found.",
        )
    return run
