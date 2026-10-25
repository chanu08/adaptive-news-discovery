"""
Feedback Routes — Phase 9

POST /api/users/mind/{user_id}/feedback  — submit a feedback event
GET  /api/users/mind/{user_id}/feedback  — retrieve feedback history

These routes do NOT modify any MIND historical data.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, field_validator
from typing import Optional

from app.database.connection import get_db
from app.services.feedback.feedback_service import (
    submit_feedback,
    get_feedback_history,
    FeedbackValidationError,
)
from app.services.feedback.constants import VALID_EVENT_TYPES

router = APIRouter()


class FeedbackRequest(BaseModel):
    mind_id: str
    event_type: str
    source: Optional[str] = "recommendation"

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        up = v.upper()
        if up not in VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of: {sorted(VALID_EVENT_TYPES)}"
            )
        return up

    @field_validator("mind_id")
    @classmethod
    def validate_mind_id(cls, v: str) -> str:
        v = v.strip()
        if not v or not v.startswith("N"):
            raise ValueError("mind_id must be a valid MIND article ID (e.g. 'N12345')")
        return v


@router.post(
    "/mind/{mind_user_id}/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
)
async def post_feedback(
    mind_user_id: str,
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Records a user feedback event for a recommended article and
    immediately updates the RECENT interest profile + drift.

    Event types: READ (+0.5), LIKE (+1.0), DISLIKE (-1.0)
    """
    try:
        result = await submit_feedback(
            session=db,
            user_mind_id=mind_user_id,
            article_mind_id=body.mind_id,
            event_type=body.event_type,
            source=body.source or "recommendation",
        )
        return result
    except FeedbackValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/mind/{mind_user_id}/feedback",
    tags=["Feedback"],
)
async def get_user_feedback_history(
    mind_user_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max events to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the most recent feedback events for a user (newest first).
    """
    events = await get_feedback_history(db, mind_user_id, limit)
    return {"user_id": mind_user_id, "count": len(events), "events": events}
