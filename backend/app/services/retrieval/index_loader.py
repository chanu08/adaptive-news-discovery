import os
import json
import scipy.sparse as sp
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from app.config.settings import settings

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data", "processed", "tfidf"))
MATRIX_FILE = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
MAPPING_FILE = os.path.join(INDEX_DIR, "row_mapping.json")
NMF_FILE = os.path.join(INDEX_DIR, "nmf_model.joblib")

class IRIndexStore:
    _instance = None
    
    def __init__(self):
        self.tfidf_matrix = None
        self.topic_matrix = None
        self.nmf_model = None
        self.row_to_mind_id: Dict[int, str] = {}
        self.mind_id_to_row: Dict[str, int] = {}
        self.is_loaded = False
        
    @classmethod
    def get_instance(cls) -> "IRIndexStore":
        if cls._instance is None:
            cls._instance = IRIndexStore()
        return cls._instance
        
    def load(self):
        if self.is_loaded:
            return
            
        if not os.path.exists(MATRIX_FILE) or not os.path.exists(MAPPING_FILE):
            raise FileNotFoundError("TF-IDF index not found. Run phase 5 build script.")
            
        self.tfidf_matrix = sp.load_npz(MATRIX_FILE)
        
        with open(MAPPING_FILE, "r") as f:
            mapping_str = json.load(f)
            self.row_to_mind_id = {int(k): v for k, v in mapping_str.items()}
            self.mind_id_to_row = {v: int(k) for k, v in mapping_str.items()}
            
        if os.path.exists(NMF_FILE):
            self.nmf_model = joblib.load(NMF_FILE)
            # Pre-compute topic matrix for all articles for fast retrieval
            self.topic_matrix = self.nmf_model.transform(self.tfidf_matrix)
            
        self.is_loaded = True
        
    def get_article_vector(self, mind_id: str) -> Optional[sp.csr_matrix]:
        row_idx = self.mind_id_to_row.get(mind_id)
        if row_idx is not None:
            return self.tfidf_matrix[row_idx]
        return None
        
    def get_article_topic_vector(self, mind_id: str) -> Optional[np.ndarray]:
        row_idx = self.mind_id_to_row.get(mind_id)
        if row_idx is not None and self.topic_matrix is not None:
            return self.topic_matrix[row_idx]
        return None
        
    def compute_similarity(self, user_vector: sp.csr_matrix) -> np.ndarray:
        return cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
    def compute_topic_similarity(self, user_topic_vector: np.ndarray) -> np.ndarray:
        if self.topic_matrix is None:
            return np.zeros(len(self.row_to_mind_id))
        # Ensure 2D
        if user_topic_vector.ndim == 1:
            user_topic_vector = user_topic_vector.reshape(1, -1)
        return cosine_similarity(user_topic_vector, self.topic_matrix).flatten()

ir_store = IRIndexStore.get_instance()

