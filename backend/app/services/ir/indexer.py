"""
IR Indexer — Phase 5
Builds TF-IDF vectors and extracts topics using NMF.
"""

import os
import json
import joblib
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from app.models.news_article import NewsArticle
from app.models.article_feature import ArticleTextFeature
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data", "processed", "tfidf")
INDEX_DIR = os.path.abspath(INDEX_DIR)

# Configuration
N_TOPICS = 50
MAX_FEATURES = 50000
PREPROCESSING_VERSION = "v1-tfidf-nmf50"

async def build_index(session: AsyncSession):
    """
    1. Load canonical articles.
    2. Preprocess and compute TF-IDF.
    3. Extract Topics with NMF.
    4. Save to disk.
    5. Save metadata to DB.
    """
    print("Loading articles from database...")
    result = await session.execute(select(NewsArticle))
    articles = result.scalars().all()
    
    if not articles:
        print("No articles found in DB.")
        return None
        
    print(f"Loaded {len(articles)} canonical articles.")
    
    # We must maintain stable order
    articles.sort(key=lambda x: x.id)
    
    documents = []
    mind_ids = []
    
    stats = {
        "total": len(articles),
        "with_abstract": 0,
        "title_only": 0
    }
    
    for a in articles:
        mind_ids.append(a.mind_id)
        if a.abstract and a.abstract.strip():
            documents.append(f"{a.title} {a.abstract}")
            stats["with_abstract"] += 1
        else:
            documents.append(a.title or "")
            stats["title_only"] += 1
            
    print("Building TF-IDF representation...")
    # Standard TF-IDF: lowercasing, stop-words, min_df to remove very rare typos
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        min_df=2,
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab_size = len(vectorizer.vocabulary_)
    
    print(f"TF-IDF matrix built: {tfidf_matrix.shape} with {tfidf_matrix.nnz} non-zero entries.")
    
    print("Extracting topics with NMF...")
    nmf_model = NMF(n_components=N_TOPICS, random_state=42, init='nndsvd', max_iter=200)
    topic_weights = nmf_model.fit_transform(tfidf_matrix)
    
    print("Saving representations to disk...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Save sparse matrix
    sp.save_npz(os.path.join(INDEX_DIR, "tfidf_matrix.npz"), tfidf_matrix)
    # Save vectorizer and NMF
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    joblib.dump(nmf_model, os.path.join(INDEX_DIR, "nmf_model.joblib"))
    
    # Save mapping (Row Index -> mind_id)
    mapping = {i: mid for i, mid in enumerate(mind_ids)}
    with open(os.path.join(INDEX_DIR, "row_mapping.json"), "w") as f:
        json.dump(mapping, f)
        
    # Save topic terms mapping for easy inspection
    feature_names = vectorizer.get_feature_names_out()
    topics_metadata = {}
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_terms = [feature_names[i] for i in top_indices]
        topics_metadata[str(topic_idx)] = top_terms
        
    with open(os.path.join(INDEX_DIR, "topics_metadata.json"), "w") as f:
        json.dump(topics_metadata, f, indent=2)
        
    print("Updating database metadata...")
    # Clear existing
    await session.execute(delete(ArticleTextFeature))
    
    # Batch insert to prevent massive memory usage
    batch_size = 5000
    features_to_insert = []
    
    for i, mid in enumerate(mind_ids):
        row_weights = topic_weights[i]
        dominant_topic = int(np.argmax(row_weights))
        
        # Keep top 3 topic weights for compactness
        top_3_idx = np.argsort(row_weights)[-3:][::-1]
        w_dict = {int(idx): float(row_weights[idx]) for idx in top_3_idx if row_weights[idx] > 0}
        
        features_to_insert.append(
            ArticleTextFeature(
                mind_id=mid,
                preprocessing_version=PREPROCESSING_VERSION,
                dominant_topic=dominant_topic,
                topic_weights_json=json.dumps(w_dict)
            )
        )
        
        if len(features_to_insert) >= batch_size:
            session.add_all(features_to_insert)
            await session.commit()
            features_to_insert = []
            
    if features_to_insert:
        session.add_all(features_to_insert)
        await session.commit()
        
    print("Indexing complete!")
    
    return {
        "canonical_docs": stats["total"],
        "with_abstract": stats["with_abstract"],
        "title_only": stats["title_only"],
        "vocab_size": vocab_size,
        "matrix_shape": tfidf_matrix.shape,
        "nnz": tfidf_matrix.nnz,
        "topics": N_TOPICS
    }
