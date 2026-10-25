"""
Profile Storage Service — Phase 3
Handles database operations for UserInterestProfile.
"""

import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from app.models.interest_profile import UserInterestProfile

async def save_user_profile(
    session: AsyncSession, 
    user_id: str, 
    profile_type: str, 
    categories: dict, 
    subcategories: dict
) -> None:
    """
    Upserts the user interest profile into the database.
    """
    stmt = insert(UserInterestProfile).values(
        user_mind_id=user_id,
        profile_type=profile_type,
        categories_json=json.dumps(categories),
        subcategories_json=json.dumps(subcategories)
    )
    
    # On conflict (user_mind_id + profile_type unique constraint), update the json strings
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_mind_id", "profile_type"],
        set_={
            "categories_json": stmt.excluded.categories_json,
            "subcategories_json": stmt.excluded.subcategories_json,
            "updated_at": stmt.excluded.updated_at
        }
    )
    
    await session.execute(stmt)

async def get_user_profile(session: AsyncSession, user_id: str, profile_type: str) -> dict | None:
    """
    Retrieves and parses a user's interest profile.
    Returns None if the profile doesn't exist.
    """
    result = await session.execute(
        select(UserInterestProfile)
        .where(UserInterestProfile.user_mind_id == user_id)
        .where(UserInterestProfile.profile_type == profile_type)
    )
    profile = result.scalar_one_or_none()
    
    if not profile:
        return None
        
    return {
        "user_id": profile.user_mind_id,
        "profile_type": profile.profile_type,
        "categories": json.loads(profile.categories_json),
        "subcategories": json.loads(profile.subcategories_json),
        "updated_at": profile.updated_at.isoformat()
    }

async def get_all_user_profiles(session: AsyncSession, user_id: str) -> dict:
    """
    Retrieves both LONG_TERM and RECENT profiles.
    """
    long_term = await get_user_profile(session, user_id, "LONG_TERM")
    recent = await get_user_profile(session, user_id, "RECENT")
    
    return {
        "long_term": long_term,
        "recent": recent
    }
