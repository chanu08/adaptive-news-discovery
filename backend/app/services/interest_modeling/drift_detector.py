"""
Interest Drift Detector — Phase 4
Calculates Jensen-Shannon Divergence and categorizes feature drift.
"""

import math
from typing import Dict, List, Tuple

# Configurable thresholds
STABLE_THRESHOLD = 0.05
EVIDENCE_THRESHOLDS = {
    "ADEQUATE": 10,   # Needs >= 10 recent timestamped clicks to be adequate
    "LIMITED": 5      # Needs >= 5 recent timestamped clicks to be limited
}

def align_features(long_term: Dict[str, float], recent: Dict[str, float]) -> List[Tuple[str, float, float]]:
    """
    Creates a unified feature space.
    Returns list of (feature_name, long_term_weight, recent_weight).
    """
    all_features = set(long_term.keys()).union(set(recent.keys()))
    aligned = []
    
    for feat in all_features:
        lt_w = long_term.get(feat, 0.0)
        rt_w = recent.get(feat, 0.0)
        aligned.append((feat, lt_w, rt_w))
        
    return aligned

def compute_jsd(p: List[float], q: List[float], epsilon: float = 1e-9) -> float:
    """
    Computes Jensen-Shannon Divergence between two probability distributions.
    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q).
    Epsilon is used to avoid log(0) for safe probability handling.
    """
    if len(p) != len(q):
        raise ValueError("Distributions must be of equal length.")
        
    if not p:
        return 0.0
        
    def kl_divergence(dist_a, dist_b):
        kl = 0.0
        for a, b in zip(dist_a, dist_b):
            a_safe = max(a, epsilon)
            b_safe = max(b, epsilon)
            kl += a * math.log2(a_safe / b_safe)
        return kl
        
    m = [0.5 * (a + b) for a, b in zip(p, q)]
    
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    # Due to floating point inaccuracy, JSD can occasionally be slightly negative
    return max(0.0, jsd)

def classify_feature_change(feat: str, lt_w: float, rt_w: float, threshold: float = STABLE_THRESHOLD) -> Dict:
    """
    Classifies a feature as EMERGING, DECLINING, or STABLE.
    """
    delta = rt_w - lt_w
    
    if delta >= threshold:
        classification = "EMERGING"
    elif delta <= -threshold:
        classification = "DECLINING"
    else:
        classification = "STABLE"
        
    return {
        "feature": feat,
        "long_term_weight": round(lt_w, 4),
        "recent_weight": round(rt_w, 4),
        "delta": round(delta, 4),
        "classification": classification
    }

def evaluate_evidence(recent_timestamped_clicks: int) -> str:
    """
    Evaluates confidence/evidence level based on recent timestamped clicks.
    """
    if recent_timestamped_clicks >= EVIDENCE_THRESHOLDS["ADEQUATE"]:
        return "ADEQUATE"
    elif recent_timestamped_clicks >= EVIDENCE_THRESHOLDS["LIMITED"]:
        return "LIMITED"
    return "INSUFFICIENT"

def detect_drift(user_id: str, lt_profile: Dict, rt_profile: Dict, evidence_stats: Dict) -> Dict:
    """
    Computes overall drift and categorizes feature changes.
    lt_profile / rt_profile: {"categories": {...}, "subcategories": {...}}
    """
    # 1. Align features
    cat_aligned = align_features(lt_profile.get("categories", {}), rt_profile.get("categories", {}))
    subcat_aligned = align_features(lt_profile.get("subcategories", {}), rt_profile.get("subcategories", {}))
    
    # JSD requires comparing the complete distribution. We can normalize across both cat and subcat 
    # to form a joint probability distribution, OR compute them separately and average them.
    # We will compute JSD on the unified flattened distribution of categories + subcategories.
    # To make them sum to 1.0, we must divide each by 2.0 since they individually sum to 1.0 (if present).
    # However, JSD handles unnormalized or pseudo-normalized distributions fine if we normalize them.
    
    p = []
    q = []
    
    lt_total = sum(x[1] for x in cat_aligned) + sum(x[1] for x in subcat_aligned)
    rt_total = sum(x[2] for x in cat_aligned) + sum(x[2] for x in subcat_aligned)
    
    all_aligned = cat_aligned + subcat_aligned
    
    for feat, lt, rt in all_aligned:
        p.append(lt / lt_total if lt_total > 0 else 0.0)
        q.append(rt / rt_total if rt_total > 0 else 0.0)
    
    # 2. Compute JSD
    jsd_score = compute_jsd(p, q)
    
    # 3. Categorize changes
    category_changes = []
    for feat, lt, rt in cat_aligned:
        category_changes.append(classify_feature_change(feat, lt, rt))
        
    subcategory_changes = []
    for feat, lt, rt in subcat_aligned:
        subcategory_changes.append(classify_feature_change(feat, lt, rt))
        
    # Sort changes by absolute delta descending
    category_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    subcategory_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    
    all_changes = category_changes + subcategory_changes
    all_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    
    emerging = [c for c in all_changes if c["classification"] == "EMERGING"]
    declining = [c for c in all_changes if c["classification"] == "DECLINING"]
    stable = [c for c in all_changes if c["classification"] == "STABLE"]
    
    # 4. Evidence
    evidence_level = evaluate_evidence(evidence_stats.get("recent_timestamped_clicks", 0))
    
    return {
        "user_id": user_id,
        "jsd_score": round(jsd_score, 4),
        "evidence_level": evidence_level,
        "category_changes": category_changes,
        "subcategory_changes": subcategory_changes,
        "emerging": emerging,
        "declining": declining,
        "stable": stable,
        "evidence": evidence_stats
    }
