"""
Interest Drift Store — Phase 4
Saves and retrieves computed drift scores from the database.
"""

import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from app.models.interest_drift import UserInterestDrift

async def save_user_drift(session: AsyncSession, drift_data: dict) -> None:
    """
    Upserts the drift analysis for a user.
    """
    stmt = insert(UserInterestDrift).values(
        user_mind_id=drift_data["user_id"],
        jsd_score=drift_data["jsd_score"],
        evidence_level=drift_data["evidence_level"],
        category_drift_json=json.dumps(drift_data["category_changes"]),
        subcategory_drift_json=json.dumps(drift_data["subcategory_changes"]),
        emerging_json=json.dumps(drift_data["emerging"]),
        declining_json=json.dumps(drift_data["declining"]),
        stable_json=json.dumps(drift_data["stable"]),
        evidence_json=json.dumps(drift_data["evidence"])
    )
    
    # On conflict (user already exists), update the drift
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_mind_id"],
        set_={
            "jsd_score": stmt.excluded.jsd_score,
            "evidence_level": stmt.excluded.evidence_level,
            "category_drift_json": stmt.excluded.category_drift_json,
            "subcategory_drift_json": stmt.excluded.subcategory_drift_json,
            "emerging_json": stmt.excluded.emerging_json,
            "declining_json": stmt.excluded.declining_json,
            "stable_json": stmt.excluded.stable_json,
            "evidence_json": stmt.excluded.evidence_json,
            "computed_at": stmt.excluded.computed_at
        }
    )
    
    await session.execute(stmt)
    await session.commit()

async def get_user_drift(session: AsyncSession, user_id: str) -> dict | None:
    """
    Retrieves the drift analysis for a user.
    """
    result = await session.execute(
        select(UserInterestDrift).where(UserInterestDrift.user_mind_id == user_id)
    )
    drift = result.scalar_one_or_none()
    
    if not drift:
        return None
        
    return {
        "user_id": drift.user_mind_id,
        "jsd_score": drift.jsd_score,
        "evidence_level": drift.evidence_level,
        "category_changes": json.loads(drift.category_drift_json),
        "subcategory_changes": json.loads(drift.subcategory_drift_json),
        "emerging": json.loads(drift.emerging_json),
        "declining": json.loads(drift.declining_json),
        "stable": json.loads(drift.stable_json),
        "evidence": json.loads(drift.evidence_json),
        "computed_at": drift.computed_at.isoformat()
    }
