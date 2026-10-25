"""
Profile Builder Service — Phase 3
Constructs user interest profiles from historical clicks and impressions.
"""

import logging
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_article import NewsArticle
from app.models.behavior import BehaviorLog, ImpressionItem
from app.services.interest_modeling.weighting import apply_temporal_decay
from app.services.interest_modeling.profile_store import save_user_profile

logger = logging.getLogger(__name__)

# Configurable parameters for decay and window
LONG_TERM_DECAY = 0.01
RECENT_DECAY = 0.1
RECENT_WINDOW_SIZE = 20

from datetime import datetime, timedelta

from sqlalchemy import cast, String

async def _get_user_click_sequence(session: AsyncSession, user_id: str) -> list[tuple[str, datetime]]:
    """
    Returns the complete chronological sequence of article_mind_id clicked by the user
    and their corresponding interaction timestamp.
    Oldest first, newest last.
    """
    sequence = []
    
    # 1. Get history_raw from the user's earliest log (or the one with max history_length)
    # 1. Get history_raw from the user's EARLIEST log
    # This prevents double-counting clicks that appear in both later history_raw and impression_items.
    # We sort by b.time ASC to get the earliest log. Since time is stored as string 'MM/DD/YYYY ...', 
    # we can't reliably sort by string. We sort by id ASC which correlates with ingestion/time.
    result = await session.execute(
        select(BehaviorLog.history_raw)
        .where(BehaviorLog.user_mind_id == user_id)
        .order_by(BehaviorLog.id.asc())
        .limit(1)
    )
    history_row = result.first()
    
    if history_row and history_row.history_raw:
        history_raw = history_row[0]
        history_clicks = [aid for aid in history_raw.split() if aid]
        
        # We don't have true timestamps for history items.
        # DO NOT simulate dates. Pass None.
        for aid in history_clicks:
            sequence.append((aid, None))
            
    # 2. Get clicks from impression_items ordered roughly by id
    # We will sort accurately in Python since SQLite string sorting fails on '11/9' vs '11/15'
    sql = """
        SELECT i.article_mind_id, b.time, i.id
        FROM impression_items i
        JOIN behavior_logs b ON i.behavior_log_id = b.id
        WHERE b.user_mind_id = :user_id AND i.clicked = 1
    """
    result = await session.execute(text(sql), {"user_id": user_id})
    
    impression_clicks = []
    for row in result.fetchall():
        aid = row[0]
        dt = row[1]
        iid = row[2]
        
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                dt = datetime.strptime(dt, "%m/%d/%Y %I:%M:%S %p")
                
        impression_clicks.append((aid, dt, iid))
        
    # Sort strictly by datetime, then by impression_item id
    impression_clicks.sort(key=lambda x: (x[1], x[2]))
    
    for aid, dt, iid in impression_clicks:
        sequence.append((aid, dt))
        
    return sequence

async def _get_article_features(session: AsyncSession, article_ids: list[str]) -> dict[str, dict]:
    """
    Returns a dictionary mapping article_mind_id to its features (category, subcategory).
    """
    if not article_ids:
        return {}
        
    unique_ids = list(set(article_ids))
    features = {}
    
    # Query in batches of 900 to avoid SQLite limits
    batch_size = 900
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        result = await session.execute(
            select(NewsArticle.mind_id, NewsArticle.category, NewsArticle.subcategory)
            .where(NewsArticle.mind_id.in_(batch))
        )
        for row in result.all():
            features[row[0]] = {
                "category": row[1],
                "subcategory": row[2]
            }
            
    return features

async def build_user_profiles(session: AsyncSession, user_id: str) -> bool:
    """
    Builds and saves both LONG_TERM and RECENT profiles for a user.
    Returns True if successful, False if the user has no history (cold user).
    """
    sequence = await _get_user_click_sequence(session, user_id)
    if not sequence:
        # Cold user, save empty profiles
        await save_user_profile(session, user_id, "LONG_TERM", {}, {})
        await save_user_profile(session, user_id, "RECENT", {}, {})
        return False
        
    # Extract chronological feature sequences
    cat_seq = []
    subcat_seq = []
    
    # We map unique IDs to fetch features. The `sequence` has (aid, dt)
    article_ids = [aid for aid, dt in sequence]
    article_features = await _get_article_features(session, article_ids)
    
    for aid, dt in sequence:
        feats = article_features.get(aid)
        if feats:
            cat_seq.append(([feats["category"]], dt))
            subcat_seq.append(([feats["subcategory"]], dt))
        else:
            cat_seq.append(([], dt))
            subcat_seq.append(([], dt))
            
    # Long Term Profile
    lt_categories = apply_temporal_decay(cat_seq, LONG_TERM_DECAY)
    lt_subcategories = apply_temporal_decay(subcat_seq, LONG_TERM_DECAY)
    await save_user_profile(session, user_id, "LONG_TERM", lt_categories, lt_subcategories)
    
    # Recent Profile
    recent_cat_seq = cat_seq[-RECENT_WINDOW_SIZE:]
    recent_subcat_seq = subcat_seq[-RECENT_WINDOW_SIZE:]
    rt_categories = apply_temporal_decay(recent_cat_seq, RECENT_DECAY)
    rt_subcategories = apply_temporal_decay(recent_subcat_seq, RECENT_DECAY)
    await save_user_profile(session, user_id, "RECENT", rt_categories, rt_subcategories)
    
    return True
