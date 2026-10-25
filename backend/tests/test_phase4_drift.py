import pytest
from app.services.interest_modeling.drift_detector import (
    align_features, compute_jsd, classify_feature_change,
    detect_drift, evaluate_evidence
)

def test_feature_alignment():
    lt = {"Sports": 0.4, "Tech": 0.3, "News": 0.2}
    rt = {"News": 0.6, "Tech": 0.25, "Music": 0.15}
    
    aligned = align_features(lt, rt)
    assert len(aligned) == 4
    
    # Order doesn't strictly matter, let's convert to a dict of tuples
    aligned_map = {feat: (l, r) for feat, l, r in aligned}
    
    assert aligned_map["Sports"] == (0.4, 0.0)
    assert aligned_map["Tech"] == (0.3, 0.25)
    assert aligned_map["News"] == (0.2, 0.6)
    assert aligned_map["Music"] == (0.0, 0.15)

def test_jsd_identical_profiles():
    p = [0.2, 0.5, 0.3]
    q = [0.2, 0.5, 0.3]
    jsd = compute_jsd(p, q)
    assert jsd == 0.0

def test_jsd_different_profiles():
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    jsd = compute_jsd(p, q)
    assert jsd > 0.0
    
def test_jsd_symmetry():
    p = [0.1, 0.9]
    q = [0.8, 0.2]
    jsd_pq = compute_jsd(p, q)
    jsd_qp = compute_jsd(q, p)
    # Floating point precision might cause tiny differences
    assert abs(jsd_pq - jsd_qp) < 1e-7

def test_classification():
    assert classify_feature_change("A", 0.1, 0.9, 0.05)["classification"] == "EMERGING"
    assert classify_feature_change("A", 0.9, 0.1, 0.05)["classification"] == "DECLINING"
    assert classify_feature_change("A", 0.5, 0.52, 0.05)["classification"] == "STABLE"

def test_evidence_evaluation():
    assert evaluate_evidence(0) == "INSUFFICIENT"
    assert evaluate_evidence(4) == "INSUFFICIENT"
    assert evaluate_evidence(5) == "LIMITED"
    assert evaluate_evidence(9) == "LIMITED"
    assert evaluate_evidence(10) == "ADEQUATE"
    assert evaluate_evidence(50) == "ADEQUATE"

def test_detect_drift_empty_profiles():
    res = detect_drift("U1", {}, {}, {"recent_timestamped_clicks": 0})
    assert res["jsd_score"] == 0.0
    assert res["evidence_level"] == "INSUFFICIENT"

def test_detect_drift_full_pipeline():
    lt = {"categories": {"news": 0.8, "sports": 0.2}}
    rt = {"categories": {"news": 0.1, "sports": 0.9}}
    
    res = detect_drift("U1", lt, rt, {"recent_timestamped_clicks": 15})
    
    assert res["evidence_level"] == "ADEQUATE"
    assert res["jsd_score"] > 0
    assert len(res["emerging"]) == 1
    assert res["emerging"][0]["feature"] == "sports"
    assert len(res["declining"]) == 1
    assert res["declining"][0]["feature"] == "news"
