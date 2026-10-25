"""
Interest Drift Service — Phase 4
Integrates profile retrieval, drift detection, and persistence.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from app.services.interest_modeling.profile_store import get_user_profile
from app.services.interest_modeling.profile_builder import _get_user_click_sequence
from app.services.interest_modeling.drift_detector import detect_drift
from app.services.interest_modeling.drift_store import save_user_drift, get_user_drift

async def calculate_drift_for_user(session: AsyncSession, user_id: str) -> dict | None:
    """
    Computes and saves interest drift for a user.
    """
    # 1. Fetch profiles
    lt_profile = await get_user_profile(session, user_id, "LONG_TERM")
    rt_profile = await get_user_profile(session, user_id, "RECENT")
    
    if not lt_profile or not rt_profile:
        return None
        
    # 2. Get evidence stats
    sequence = await _get_user_click_sequence(session, user_id)
    total_clicks = len(sequence)
    
    # sequence is list of (aid, timestamp). Non-temporal history clicks have timestamp=None.
    # Recent profile uses the last 20 clicks.
    recent_seq = sequence[-20:]
    timestamped_clicks = sum(1 for _, dt in sequence if dt is not None)
    recent_timestamped = sum(1 for _, dt in recent_seq if dt is not None)
    
    evidence_stats = {
        "historical_clicks": total_clicks,
        "timestamped_clicks": timestamped_clicks,
        "recent_timestamped_clicks": recent_timestamped
    }
    
    # 3. Detect drift
    drift_result = detect_drift(user_id, lt_profile, rt_profile, evidence_stats)
    
    # 4. Save to database
    await save_user_drift(session, drift_result)
    
    return drift_result

async def get_drift_report(session: AsyncSession, user_id: str) -> dict | None:
    """
    Retrieves the drift report for a user, computing it if missing.
    """
    drift = await get_user_drift(session, user_id)
    if not drift:
        drift = await calculate_drift_for_user(session, user_id)
    return drift
