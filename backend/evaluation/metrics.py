"""
Evaluation Metrics — Phase 11
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

Implements standard, reproducible IR and Recommender Systems metrics:
- Precision@K
- Recall@K
- NDCG@K (Binary relevance)
- Hit Rate@K
- Intra-List Diversity (ILD) using existing TF-IDF article representations
- Category Diversity@K
- Subcategory Diversity@K
- Catalog Coverage
- Statistical summarization with fixed-seed bootstrap confidence intervals
"""

import math
import numpy as np
from typing import List, Set, Dict, Any, Tuple, Optional

def precision_at_k(recommended_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Precision@K = (number of relevant items in top K) / K
    """
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for aid in top_k if aid in ground_truth)
    return float(hits / k)

def recall_at_k(recommended_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Recall@K = (number of relevant items in top K) / (total relevant items)
    If ground_truth is empty, returns 0.0.
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for aid in top_k if aid in ground_truth)
    return float(hits / len(ground_truth))

def ndcg_at_k(recommended_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |ground_truth|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth or k <= 0:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for i, aid in enumerate(top_k):
        if aid in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i=0 -> rank 1 -> log2(2)=1

    ideal_hits = min(k, len(ground_truth))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg)

def hit_rate_at_k(recommended_ids: List[str], ground_truth: Set[str], k: int) -> float:
    """
    Hit Rate@K = 1.0 if at least one item in top K is in ground_truth, else 0.0.
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    return 1.0 if any(aid in ground_truth for aid in top_k) else 0.0

def intra_list_diversity(recommended_ids: List[str], ir_store: Any, k: Optional[int] = None) -> float:
    """
    Intra-List Diversity (ILD) using existing TF-IDF article representations.
    For recommendation list L of size M:
    ILD = 1.0 - mean_{i < j} cosine_similarity(v_i, v_j)
    
    If M < 2, returns 0.0 (diversity undefined for single/empty list).
    TF-IDF vectors from ir_store are L2-normalized, so dot product = cosine similarity.
    """
    if k is not None:
        rec_ids = recommended_ids[:k]
    else:
        rec_ids = recommended_ids

    if len(rec_ids) < 2:
        return 0.0

    vectors = []
    for aid in rec_ids:
        vec = ir_store.get_article_vector(aid)
        if vec is not None:
            vectors.append(vec)

    if len(vectors) < 2:
        return 0.0

    # Stack sparse vectors and compute pairwise similarity matrix
    import scipy.sparse as sp
    mat = sp.vstack(vectors)
    # Cosine similarity matrix (M x M)
    sim_matrix = (mat @ mat.T).toarray()
    
    # Extract strictly upper triangle
    m = sim_matrix.shape[0]
    triu_indices = np.triu_indices(m, k=1)
    if len(triu_indices[0]) == 0:
        return 0.0

    pairwise_sims = sim_matrix[triu_indices]
    mean_sim = float(np.mean(pairwise_sims))
    # Bounded between 0 and 1
    mean_sim = max(0.0, min(1.0, mean_sim))
    return float(1.0 - mean_sim)

def category_diversity(recommended_items: List[Dict[str, Any]], k: Optional[int] = None) -> int:
    """
    Category Diversity@K = Number of unique categories in top K recommendations.
    """
    if k is not None:
        items = recommended_items[:k]
    else:
        items = recommended_items

    cats = set(item.get("category", "") for item in items if item.get("category"))
    return len(cats)

def subcategory_diversity(recommended_items: List[Dict[str, Any]], k: Optional[int] = None) -> int:
    """
    Subcategory Diversity@K = Number of unique subcategories in top K recommendations.
    """
    if k is not None:
        items = recommended_items[:k]
    else:
        items = recommended_items

    subcats = set(item.get("subcategory", "") for item in items if item.get("subcategory"))
    return len(subcats)

def catalog_coverage(all_recommended_ids: Set[str], eligible_catalog_size: int) -> float:
    """
    Catalog Coverage = |unique recommended articles across all users| / |eligible catalog articles|
    """
    if eligible_catalog_size <= 0:
        return 0.0
    coverage = len(all_recommended_ids) / eligible_catalog_size
    return float(max(0.0, min(1.0, coverage)))

def compute_bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes mean, median, standard deviation, and bootstrap percentile confidence interval.
    Deterministic with fixed random seed.
    """
    if not data:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n": 0
        }

    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n < 2:
        return {
            "mean": round(mean_val, 6),
            "median": round(median_val, 6),
            "std": round(std_val, 6),
            "ci_lower": round(mean_val, 6),
            "ci_upper": round(mean_val, 6),
            "n": n
        }

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return {
        "mean": round(mean_val, 6),
        "median": round(median_val, 6),
        "std": round(std_val, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "n": n
    }
