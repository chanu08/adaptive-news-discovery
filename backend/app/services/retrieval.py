"""
Retrieval Service — PLACEHOLDER

Phase 2 Implementation Plan:
─────────────────────────────
This module will implement the core IR retrieval engine:

  1. TF-IDF Index construction — build inverted index from preprocessed articles
  2. Query processing          — preprocess and vectorise user queries
  3. Cosine similarity scoring — rank candidate articles against query
  4. BM25 (optional upgrade)   — probabilistic retrieval model

Public API (planned):
  build_index(articles: List[Article]) -> None
  retrieve(query: str, top_k: int) -> List[Tuple[int, float]]
      Returns list of (article_id, score) sorted by descending score.
"""


def build_index(articles: list) -> None:
    """[PLACEHOLDER] Build TF-IDF inverted index from articles."""
    raise NotImplementedError("TF-IDF index will be built in Phase 2.")


def retrieve(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    """[PLACEHOLDER] Retrieve top-k articles for a query using TF-IDF + cosine similarity."""
    raise NotImplementedError("Retrieval engine will be implemented in Phase 2.")
