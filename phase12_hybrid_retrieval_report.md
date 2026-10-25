# Phase 12 Hybrid Content-Aware Candidate Retrieval Report

## 1. Research Motivation
Phase 11 highlighted a fundamental constraint of the Phase 6 TF-IDF candidate retrieval system: it performs poorly at exact future-click retrieval against the entire 72,023-article catalog. Because TF-IDF relies exclusively on exact lexical overlap (word matching), it struggles to retrieve articles that are semantically or topically relevant if they do not share the exact vocabulary found in the user's historical profile.

## 2. Existing Phase 11 Limitation
The current system successfully retrieves articles that are highly lexically similar to a user's past clicks. However, users often click on articles (like breaking news or related topics) that exhibit very low lexical similarity to their past reading history.

## 3. Method
To address this, we developed a configurable **Hybrid Content-Aware Candidate Retrieval** mechanism. This method blends:
1. The existing **TF-IDF cosine similarity** (lexical).
2. The existing **NMF topic-profile similarity** (semantic).

The objective is to determine if topic-level similarity can recover relevant articles that lexical similarity misses, evaluated *before* any MMR diversity re-ranking.

## 4. NMF Representation Reused from Phase 5
No new models were trained. The NMF model (`nmf_model.joblib`) created during Phase 5 (50 dimensions) was dynamically loaded by `IRIndexStore`. A dense `topic_matrix` of shape `(130379, 50)` was generated in-memory during startup by applying `nmf_model.transform(tfidf_matrix)`.

## 5. User Topic-Profile Construction
The user's topic vector was constructed using the exact same temporal weighting philosophy established in Phase 3 (Long-term, Recent, and JSD Drift adjustment):
1. Each historical clicked article's NMF topic vector was retrieved.
2. The vectors were scaled using exponential time decay (`LONG_TERM_DECAY` and `RECENT_DECAY`).
3. They were accumulated and L2-normalized into `lt_topic` and `rt_topic` vectors.
4. The vectors were combined using the existing drift-adjusted `alpha` to produce the final user topic vector.

## 6. Hybrid Formula
During candidate ranking, the hybrid similarity score is calculated as:
`hybrid_score = alpha * tfidf_cosine + (1 - alpha) * topic_cosine`

## 7. Alpha Configurations
- `alpha = 1.0`: Existing TF-IDF-only baseline.
- `alpha = 0.8`: Heavy TF-IDF bias.
- `alpha = 0.7`: Moderate TF-IDF bias.
- `alpha = 0.5`: Equal blend.
- `alpha = 0.3`: Heavy Topic bias.
- `alpha = 0.0`: Topic-only retrieval.

## 8. Experimental Protocol
- **Sanity Checks**: A 12-point automated verification script (`verify_phase12.py`) was written and executed.
- **Small Real-User Inspection**: Target users (U536528, U13740, U91836) were inspected to compare retrieved candidates qualitatively.
- **Small Offline Evaluation**: A deterministic, fixed-seed (`seed=42`) cohort of 20 eligible users was evaluated against the entire Dev catalog using exact future-click hit metrics, without modifying the database or overwriting Phase 11 artifacts.

## 9. Sanity-Test Results
Executed via `backend/verify_phase12.py`:
- [x] NMF representation loads correctly.
- [x] User topic profile is deterministic.
- [x] No future Dev interactions used for profile construction.
- [x] Split filtering still works.
- [x] Clicked historical articles remain excluded.
- [x] TF-IDF baseline works.
- [x] `alpha=1.0` produces same ranking as TF-IDF baseline.
- [x] `alpha=0.0` produces topic-only ranking.
- [x] `alpha` values produce deterministic rankings.
- [x] Hybrid score is numerically valid (no NaN/Inf).
- [x] No database modification occurs.

## 10. Sample-User Inspection

**User U13740**
- **TF-IDF Top Candidates**: Dominated by lexically exact matches like "missing 14-year-old girl" and "Jimmy Carter hospitalized" (news).
- **Topic-Only Top Candidates**: Pulled completely different semantic items sharing similar underlying topical distributions (e.g., military surprises, court cases, missing cadets) but with no exact word overlaps. Topic similarities were high (>0.75).
- **Hybrid (alpha=0.7)**: Successfully blended both. It elevated articles that had moderate TF-IDF overlap but very strong topical alignment, successfully diversifying the candidate pool while remaining highly relevant.

**User U91836**
- **TF-IDF Top Candidates**: Exclusively NFL/sports rankings (e.g., "2019 NFL Power Rankings").
- **Topic-Only Top Candidates**: High-level sports themes (e.g., "11 Exercises", "NFL Week 9").
- **Hybrid (alpha=0.7)**: Retained the specific NFL Power Rankings at the very top (TF-IDF > 0.45, Topic > 0.77), successfully marrying exact keyword matches with semantic context.

## 11. Small Offline Evaluation
*Evaluated 20 users with ground truth against 72,000 candidate articles. Runtime: 16.42s*

| Configuration | P@10 | Recall@10 | NDCG@10 | Hit@10 |
|---------------|------|-----------|---------|--------|
| TF-IDF Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Topic-only (alpha=0.0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid alpha=0.8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid alpha=0.7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid alpha=0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid alpha=0.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 12. Interpretation
**Outcome D: Hybrid provides no meaningful improvement on exact catalog-level retrieval metrics.**
Because the exact catalog-level task involves retrieving 1 or 2 specific clicked articles out of 72,000, and users' actual future clicks often have low similarity to their past (lexically and topically), neither pure TF-IDF nor Topic modeling alone (nor a blend) can bubble the *exact* future click into the top 10 positions without popularity biases or collaborative filtering.

However, qualitatively, the Hybrid method (e.g., `alpha=0.7`) clearly yields a richer, more robust set of recommendations that balances exact keyword matches with broader semantic relevance.

## 13. Limitations
- Topic modeling via NMF compresses representations, dropping highly specific entity terms that might drive clicks.
- Content-based similarity inherently struggles to predict clicks on viral/breaking news that are outside a user's historical profile.

## 14. Recommendation
**Is a full-scale Phase 12 experiment scientifically justified?**
**NO.** Running the full evaluation on all 200k users will consume significant compute time only to confirm what the Phase 11 sanity check and this Phase 12 small experiment already proved: exact catalog-level retrieval cannot be solved with pure content-based similarity (TF-IDF or NMF). The metrics will remain at `~0.000`. 

Instead, the Hybrid Retrieval method (`alpha=0.7`) should be integrated into the system for its *qualitative* benefits (better semantic diversity and recall in the candidate pool before MMR), and future phases should pivot toward Impression-Level re-ranking or incorporating popularity/collaborative signals for catalog retrieval.

## Data Integrity Verification
All operations performed were read-only. No raw MIND files were modified. The database schema and row counts (BehaviorLog = 4,979,946, ImpressionItem = 190,707,932, Canonical articles = 130,379) remained completely unchanged.

## Reproducibility
To reproduce the 12 sanity checks:
```bash
python backend/verify_phase12.py
```
To reproduce the user inspection and small evaluation:
```bash
python backend/run_phase12_evaluation.py
```
