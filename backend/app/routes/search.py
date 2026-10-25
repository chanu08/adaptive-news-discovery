"""
Search routes — Phase 1 skeleton.

Full TF-IDF / BM25 retrieval implemented in Phase 2.
"""

from fastapi import APIRouter, Query
from typing import List, Optional

router = APIRouter()


@router.get("/")
async def search_articles(
    q: str = Query(..., min_length=1, description="Search query string"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(10, ge=1, le=100),
):
    """
    Placeholder search endpoint.

    Phase 2 will replace this with:
    - TF-IDF indexing
    - Cosine similarity ranking
    - Category filtering
    """
    return {
        "query": q,
        "category": category,
        "results": [],
        "total": 0,
        "message": "Search engine not yet implemented — coming in Phase 2.",
    }
