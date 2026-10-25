"""
Temporal Weighting and Feature Aggregation — Phase 3
"""

import math
from typing import List, Dict
from collections import defaultdict

from datetime import datetime

def apply_temporal_decay(features_with_time: List[tuple[List[str], datetime]], decay_lambda: float) -> Dict[str, float]:
    """
    Applies exponential time decay to a sequence of feature lists using timestamps.
    
    features_with_time: Chronological list of (features, timestamp).
                        Oldest click is at index 0.
    decay_lambda: The exponential decay parameter (lambda).
    
    Returns a dictionary of normalized feature weights.
    """
    if not features_with_time:
        return {}
        
    weights = defaultdict(float)
    
    # Reference time is the latest timestamp for this user
    reference_time = max((t for f, t in features_with_time if t), default=datetime.utcnow())
    
    for features, timestamp in features_with_time:
        if timestamp is None:
            # Documented Strategy: Untimestamped historical clicks are retained 
            # as an unweighted historical interest signal (w = 1.0).
            w = 1.0
        else:
            # Calculate age in days
            age_days = (reference_time - timestamp).total_seconds() / 86400.0
            age_days = max(0.0, age_days)
            w = math.exp(-decay_lambda * age_days)
        
        for feature in features:
            if feature:
                weights[feature] += w
                
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        for k in weights:
            weights[k] = round(weights[k] / total_weight, 4)
            
    return dict(weights)
