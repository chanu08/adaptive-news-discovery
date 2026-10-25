"""
Explanation Generator — Phase 8
Produces human-readable, deterministic explanations for article recommendations.
"""

from typing import Dict, Any, List

# Thresholds
STRONG_WEIGHT_THRESHOLD = 0.10
MODERATE_WEIGHT_THRESHOLD = 0.05
DIVERSITY_RANK_IMPROVEMENT_MIN = 5

def generate_explanation(
    candidate: Dict[str, Any],
    profiles: Dict[str, Any],
    drift_report: Dict[str, Any],
    mmr_applied: bool
) -> Dict[str, Any]:
    """
    Generates a deterministic explanation for a single candidate.
    
    candidate: The ranked candidate dict (contains category, subcategory, similarity, mmr_rank, orig_rank)
    profiles: Contains long_term and recent profiles
    drift_report: Phase 4 drift report
    mmr_applied: Boolean indicating if MMR was used
    """
    cat = candidate.get("category", "")
    subcat = candidate.get("subcategory", "")
    
    # 1. Gather Signals
    lt_profile = profiles.get("long_term", {}) or {}
    rt_profile = profiles.get("recent", {}) or {}
    
    lt_cat_weight = lt_profile.get("categories", {}).get(cat, 0.0)
    rt_cat_weight = rt_profile.get("categories", {}).get(cat, 0.0)
    lt_sub_weight = lt_profile.get("subcategories", {}).get(subcat, 0.0)
    rt_sub_weight = rt_profile.get("subcategories", {}).get(subcat, 0.0)
    
    # Check drift for this category/subcategory
    drift_status = "STABLE"
    drift_delta = 0.0
    if drift_report:
        for item in drift_report.get("emerging", []):
            # We match by string, which could be category or subcategory
            if item["feature"] == cat or item["feature"] == subcat:
                drift_status = "EMERGING"
                drift_delta = item["delta"]
                break
        if drift_status == "STABLE":
            for item in drift_report.get("declining", []):
                if item["feature"] == cat or item["feature"] == subcat:
                    drift_status = "DECLINING"
                    drift_delta = item["delta"]
                    break
                    
    evidence_level = drift_report.get("evidence_level", "INSUFFICIENT") if drift_report else "INSUFFICIENT"
    
    # Diversity signal
    rank_improvement = 0
    is_diverse = False
    if mmr_applied:
        orig_rank = candidate.get("orig_rank", 9999)
        mmr_rank = candidate.get("mmr_rank", 9999)
        rank_improvement = orig_rank - mmr_rank
        if rank_improvement >= DIVERSITY_RANK_IMPROVEMENT_MIN:
            is_diverse = True
            
    signals = {
        "long_term_match": max(lt_cat_weight, lt_sub_weight),
        "recent_match": max(rt_cat_weight, rt_sub_weight),
        "drift_status": drift_status,
        "diversity_contribution": rank_improvement if is_diverse else 0
    }
    
    # 2. Reason Selection Priority
    primary_reason = ""
    supporting_reasons = []
    
    has_strong_rt = rt_cat_weight >= STRONG_WEIGHT_THRESHOLD or rt_sub_weight >= STRONG_WEIGHT_THRESHOLD
    has_strong_lt = lt_cat_weight >= STRONG_WEIGHT_THRESHOLD or lt_sub_weight >= STRONG_WEIGHT_THRESHOLD
    
    # Priority 1: Strong recent interest + strong emerging drift
    if has_strong_rt and drift_status == "EMERGING":
        if evidence_level in ["ADEQUATE", "STRONG"]:
            primary_reason = f"Related to {cat.capitalize()}, an area that has recently become more prominent in your interests."
            supporting_reasons.append("Emerging Interest Match")
        elif evidence_level == "LIMITED":
            primary_reason = f"This article is related to a topic appearing in your recent activity."
            supporting_reasons.append("Recent Topic Match")
            
    # Priority 2: Strong recent + strong long term
    if not primary_reason and has_strong_rt and has_strong_lt:
        primary_reason = "This article matches both your recent and long-term interests."
        supporting_reasons.append("Consistent Interest")
        
    # Priority 3: Strong recent only
    if not primary_reason and has_strong_rt and not has_strong_lt:
        primary_reason = f"Matches your recent interest in {cat.capitalize()}."
        supporting_reasons.append("Recent Interest")
        
    # Priority 4: Strong long term only
    if not primary_reason and has_strong_lt and not has_strong_rt:
        primary_reason = f"Matches your long-term interest in {cat.capitalize()}."
        supporting_reasons.append("Long-Term Interest")
        
    # Priority 5: Diversity
    if not primary_reason and is_diverse:
        primary_reason = "Included to add a different perspective while remaining relevant to your interests."
        supporting_reasons.append("Diversity Recommendation")
        
    # Priority 6: Moderate Subcategory Match
    if not primary_reason and (rt_sub_weight >= MODERATE_WEIGHT_THRESHOLD or lt_sub_weight >= MODERATE_WEIGHT_THRESHOLD):
        primary_reason = "This article matches a specific topic you have engaged with."
        supporting_reasons.append("Subcategory Match")
        
    # Priority 7: Moderate Category Match
    if not primary_reason and (rt_cat_weight >= MODERATE_WEIGHT_THRESHOLD or lt_cat_weight >= MODERATE_WEIGHT_THRESHOLD):
        primary_reason = "This article matches categories you frequently read."
        supporting_reasons.append("Category Match")
        
    # Priority 8: Fallback
    if not primary_reason:
        primary_reason = "This article has strong textual similarity to the topics in your reading history."
        supporting_reasons.append("General Relevance")
        
    # Add secondary diversity support if not primary
    if is_diverse and primary_reason != "Included to add a different perspective while remaining relevant to your interests.":
        supporting_reasons.append("Adds Topical Diversity")
        
    return {
        "primary_reason": primary_reason,
        "supporting_reasons": supporting_reasons,
        "signals": signals
    }
