import pytest
import math
from datetime import datetime, timedelta
from app.services.interest_modeling.weighting import apply_temporal_decay

def test_temporal_decay_empty():
    assert apply_temporal_decay([], 0.1) == {}

def test_temporal_decay_single():
    res = apply_temporal_decay([(["sports"], datetime(2026, 9, 25))], 0.1)
    assert res == {"sports": 1.0}

def test_temporal_decay_multiple():
    # Sequence: [oldest, newest]
    t1 = datetime(2026, 9, 25)
    t2 = datetime(2026, 9, 26) # 1 day newer
    
    # Age of oldest (sports) = 1 day, Age of newest (news) = 0 days
    # lambda = 0.1
    # Weight of news = exp(0) = 1.0
    # Weight of sports = exp(-0.1 * 1.0) = 0.9048
    # Total = 1.9048
    res = apply_temporal_decay([(["sports"], t1), (["news"], t2)], 0.1)
    
    assert "news" in res
    assert "sports" in res
    
    w_news = res["news"]
    w_sports = res["sports"]
    
    # news is newer, should have higher weight
    assert w_news > w_sports
    
    expected_news = round(1.0 / (1.0 + math.exp(-0.1)), 4)
    expected_sports = round(math.exp(-0.1) / (1.0 + math.exp(-0.1)), 4)
    
    assert res["news"] == expected_news
    assert res["sports"] == expected_sports

def test_temporal_decay_multiple_features_per_article():
    t1 = datetime(2026, 9, 25)
    t2 = datetime(2026, 9, 26)
    res = apply_temporal_decay([(["A", "B"], t1), (["B"], t2)], 0.1)
    assert res["B"] > res["A"]

def test_temporal_decay_missing_timestamp():
    t_recent = datetime(2026, 9, 26)
    # Old un-timestamped click (None), recent timestamped click
    res = apply_temporal_decay([(["history_feat"], None), (["recent_feat"], t_recent)], 0.1)
    
    # Missing timestamp should receive unweighted 1.0. 
    # Recent (age 0) receives exp(0) = 1.0.
    # So they should be EXACTLY equal.
    assert res["history_feat"] == res["recent_feat"]
    assert res["history_feat"] == 0.5

def test_temporal_decay_missing_timestamp_vs_older():
    t_old = datetime(2026, 9, 25)
    t_new = datetime(2026, 9, 26)
    
    # history_feat: w = 1.0
    # old_feat: w = exp(-0.1 * 1.0) = 0.9048
    # new_feat: w = exp(-0.1 * 0.0) = 1.0
    res = apply_temporal_decay([
        (["history_feat"], None),
        (["old_feat"], t_old),
        (["new_feat"], t_new)
    ], 0.1)
    
    assert res["history_feat"] == res["new_feat"]
    assert res["new_feat"] > res["old_feat"]
