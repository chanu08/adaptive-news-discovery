"""
Topic Extraction Service — PLACEHOLDER

Phase 3 Implementation Plan:
─────────────────────────────
This module will extract latent topics from news articles:

  1. LDA (Latent Dirichlet Allocation) — unsupervised topic modelling
  2. Topic labelling                    — assign human-readable topic names
  3. Article-topic assignment           — map each article to its dominant topic
  4. Topic vocabulary                   — per-topic keyword lists

Dependencies (to be added in Phase 3):
  - gensim
  - scikit-learn (LDA)

Public API (planned):
  extract_topics(articles: List[str], n_topics: int) -> TopicModel
  get_article_topics(article_id: int) -> List[Tuple[int, float]]
      Returns list of (topic_id, weight).
"""


def extract_topics(articles: list[str], n_topics: int = 10) -> dict:
    """[PLACEHOLDER] Fit LDA topic model on article corpus."""
    raise NotImplementedError("Topic extraction will be implemented in Phase 3.")


def get_article_topics(article_text: str) -> list[tuple[int, float]]:
    """[PLACEHOLDER] Infer topic distribution for a single article."""
    raise NotImplementedError("Topic inference will be implemented in Phase 3.")
