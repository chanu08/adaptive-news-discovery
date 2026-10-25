"""
MMR Ranker — Phase 7
Implements Maximal Marginal Relevance for diversity-aware re-ranking.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict

from app.services.retrieval.index_loader import ir_store

def mmr_rerank(candidate_mids: List[str], candidate_scores: List[float], final_k: int, mmr_lambda: float) -> List[int]:
    """
    Re-ranks a list of candidates using Maximal Marginal Relevance.
    
    candidate_mids: List of candidate mind_ids (already sorted by relevance)
    candidate_scores: Corresponding relevance scores (user-to-article cosine similarity)
    final_k: The number of documents to select
    mmr_lambda: Weight for relevance (lambda) vs diversity (1 - lambda)
    
    Returns:
        List of indices (from the original candidate list) representing the selected items in MMR order.
    """
    ir_store.load()
    
    n_candidates = len(candidate_mids)
    final_k = min(final_k, n_candidates)
    
    if final_k == 0:
        return []
        
    if n_candidates == 1:
        return [0]
        
    # Extract the TF-IDF vectors for the candidate pool
    # We build a small sparse matrix for just the candidates
    candidate_rows = [ir_store.mind_id_to_row[mid] for mid in candidate_mids if mid in ir_store.mind_id_to_row]
    
    if not candidate_rows:
        return []
        
    candidate_matrix = ir_store.tfidf_matrix[candidate_rows]
    
    # Compute the pairwise similarity matrix (candidate x candidate)
    # Since vectors are normalized, dot product = cosine similarity
    sim_matrix = candidate_matrix.dot(candidate_matrix.T).toarray()
    
    selected_indices = []
    unselected_indices = list(range(len(candidate_rows)))
    
    # The first document selected is always the one with highest relevance
    # because diversity penalty is 0 when nothing is selected.
    # We assume candidate_scores is already sorted, so index 0 is the highest relevance.
    first_idx = np.argmax(candidate_scores)
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)
    
    while len(selected_indices) < final_k and unselected_indices:
        best_score = -np.inf
        best_idx = -1
        
        for idx in unselected_indices:
            # Relevance score
            rel = candidate_scores[idx]
            
            # Diversity penalty: max similarity to already selected documents
            # sim_matrix[idx, selected_indices] gives similarity to all selected
            max_sim = np.max(sim_matrix[idx, selected_indices])
            
            # MMR Score
            mmr_score = (mmr_lambda * rel) - ((1.0 - mmr_lambda) * max_sim)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
                
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    return selected_indices
