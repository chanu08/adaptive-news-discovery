"""
Ranking Service — PLACEHOLDER

Phase 4 Implementation Plan:
─────────────────────────────
After retrieval, candidates are re-ranked combining multiple signals:

  Ranking Formula (planned):
    final_score(d, u) =
        λ₁ × relevance_score(d, q)        ← TF-IDF cosine similarity
      + λ₂ × profile_score(d, u)          ← similarity to user interest vector
      + λ₃ × recency_score(d)             ← prefer newer articles
      + λ₄ × popularity_score(d)          ← engagement signals

  Where λ₁ + λ₂ + λ₃ + λ₄ = 1 and are tuned per user / experiment.

Public API (planned):
  rank(candidates: List[Article], user_id: int, query: str) -> List[RankedArticle]
"""


def rank(candidates: list, user_id: int, query: str) -> list:
    """[PLACEHOLDER] Re-rank candidate articles for a user and query."""
    raise NotImplementedError("Ranking will be implemented in Phase 4.")
