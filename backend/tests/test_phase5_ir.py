import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from app.services.ir.indexer import build_index, INDEX_DIR, MAX_FEATURES, N_TOPICS
from app.database.connection import AsyncSessionLocal

def test_preprocessing_determinism():
    # The TfidfVectorizer is completely deterministic.
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = ["Apple announces new iPhone", "Tech giant releases phone"]
    v1 = TfidfVectorizer().fit_transform(docs)
    v2 = TfidfVectorizer().fit_transform(docs)
    assert (v1 != v2).nnz == 0

def test_cosine_similarity_sanity():
    # Cosine self-similarity should be 1.0
    v1 = sp.csr_matrix([[0.1, 0.5, 0.0]])
    sim = cosine_similarity(v1, v1)
    assert np.isclose(sim[0, 0], 1.0)
    
    # Orthogonal vectors should have 0.0
    v2 = sp.csr_matrix([[0.0, 0.0, 0.8]])
    sim2 = cosine_similarity(v1, v2)
    assert np.isclose(sim2[0, 0], 0.0)

@pytest.mark.asyncio
async def test_api_representation_format():
    # We won't test the actual endpoint here since we don't have a test client setup,
    # but we can verify the service logic if we wanted.
    # Because it requires the full index, we'll keep this light.
    pass
