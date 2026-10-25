"""
Pydantic schemas for dataset stats and ingestion endpoints — Phase 2.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IngestionRequest(BaseModel):
    split_dir: str     # absolute or relative path to MIND data directory or zip
    split: str         # "train" | "dev" | "test"


class IngestionRunResponse(BaseModel):
    run_id: str
    split: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    news_read: int
    news_inserted: int
    news_skipped: int
    behaviors_read: int
    behaviors_inserted: int
    behaviors_skipped: int
    items_inserted: int
    users_inserted: int
    error_log: Optional[str] = None

    class Config:
        from_attributes = True


class DatasetStatsResponse(BaseModel):
    generated_at: str
    total_articles: int
    total_users: int
    total_behavior_logs: int
    total_impression_items: int
    total_clicks: int
    click_through_rate: float
    unique_categories: int
    articles_per_category: dict[str, int]
    articles_per_split: dict[str, int]
    behavior_logs_per_split: dict[str, int]
    articles_with_abstract: int
    articles_without_abstract: int
    earliest_behavior: Optional[str]
    latest_behavior: Optional[str]
    avg_impressions_per_log: float
    avg_history_length: float
    orphan_impression_article_ids: int
    last_ingestion_status: Optional[str]
    last_ingestion_at: Optional[str]
