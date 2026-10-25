"""
User Vector Builder — Phase 6
Constructs the user's TF-IDF interest vector by weighting their clicked articles.
"""

import math
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.preprocessing import normalize

from app.services.interest_modeling.profile_builder import _get_user_click_sequence, RECENT_WINDOW_SIZE, LONG_TERM_DECAY, RECENT_DECAY
from app.services.interest_modeling.drift_service import get_drift_report
from app.services.retrieval.index_loader import ir_store

def _compute_article_weights(sequence: List[Tuple[str, Optional[datetime]]], decay_lambda: float) -> List[Tuple[str, float]]:
    if not sequence:
        return []
    
    # Reference time is the latest timestamp for this user
    reference_time = max((t for aid, t in sequence if t), default=datetime.utcnow())
    weights = []
    
    for aid, timestamp in sequence:
        if timestamp is None:
            w = 1.0
        else:
            age_days = max(0.0, (reference_time - timestamp).total_seconds() / 86400.0)
            w = math.exp(-decay_lambda * age_days)
        weights.append((aid, w))
        
    return weights

def _build_weighted_vector(weights: List[Tuple[str, float]]) -> sp.csr_matrix:
    if not weights:
        return None
        
    # Accumulate scaled vectors
    user_vec = None
    total_weight = 0.0
    
    for aid, w in weights:
        vec = ir_store.get_article_vector(aid)
        if vec is not None:
            if user_vec is None:
                user_vec = vec * w
            else:
                user_vec += vec * w
            total_weight += w
            
    if user_vec is None or total_weight == 0:
        return None
        
    user_vec = user_vec / total_weight
    user_vec = normalize(user_vec, norm='l2', axis=1)
    return user_vec

async def get_user_vector(session: AsyncSession, user_id: str, alpha: float = 0.8) -> Tuple[Optional[sp.csr_matrix], dict]:
    """
    Constructs the long-term and recent vectors separately, then combines them:
    user_vector = alpha * long_term_vector + (1 - alpha) * recent_vector
    
    Optionally adjusts `alpha` based on Interest Drift (Phase 4).
    """
    ir_store.load()
    
    sequence = await _get_user_click_sequence(session, user_id)
    if not sequence:
        return None, {"status": "INSUFFICIENT"}
        
    clicked_mind_ids = [aid for aid, dt in sequence]
        
    # Long Term
    lt_weights = _compute_article_weights(sequence, LONG_TERM_DECAY)
    lt_vec = _build_weighted_vector(lt_weights)
    
    # Recent
    rt_sequence = sequence[-RECENT_WINDOW_SIZE:]
    rt_weights = _compute_article_weights(rt_sequence, RECENT_DECAY)
    rt_vec = _build_weighted_vector(rt_weights)
    
    if lt_vec is None and rt_vec is None:
        return None, {"status": "INSUFFICIENT"}
        
    # Drift Integration
    drift_report = await get_drift_report(session, user_id)
    base_recent_weight = 1.0 - alpha
    effective_recent_weight = base_recent_weight
    
    if drift_report and drift_report.get("evidence_level") in ["ADEQUATE", "LIMITED"]:
        # If there's high drift, increase recent weight
        jsd = drift_report.get("jsd_score", 0.0)
        # Configurable mapping: shift up to 0.5 depending on JSD. 
        # JSD is [0, ~1], so we can just add JSD to the base recent weight, bounded at 1.0
        drift_adjustment = min(jsd, 0.5) 
        effective_recent_weight = min(1.0, base_recent_weight + drift_adjustment)
        
    effective_lt_weight = 1.0 - effective_recent_weight
    
    if lt_vec is not None and rt_vec is not None:
        combined = (lt_vec * effective_lt_weight) + (rt_vec * effective_recent_weight)
    elif lt_vec is not None:
        combined = lt_vec
    else:
        combined = rt_vec
        
    combined = normalize(combined, norm='l2', axis=1)
    
    stats = {
        "status": "OK",
        "long_term_weight": round(effective_lt_weight, 4),
        "recent_weight": round(effective_recent_weight, 4),
        "base_alpha": round(alpha, 4),
        "clicked_articles": clicked_mind_ids
    }
    
    return combined, stats
