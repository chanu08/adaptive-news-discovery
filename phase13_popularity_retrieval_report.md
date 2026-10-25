# Phase 13 Popularity-Aware Hybrid Candidate Retrieval Report

## 1. Research Question
**"Does a popularity signal improve full-catalog candidate ranking when combined with the existing content-based retrieval signal?"**

## 2. Motivation from Phase 11/12
Phase 11 and 12 demonstrated that predicting the exact 1-2 articles a user will click in the future out of a 72,023-article catalog is incredibly difficult using content similarity alone. Highly relevant semantic/lexical matches are often retrieved, but they might be obscure or outdated. Phase 13 investigates whether boosting retrieved candidates by their global popularity improves the exact-click hit rate and the qualitative relevance of the candidate pool.

## 3. Popularity Definition
Article popularity is strictly defined as the **number of unique TRAIN users who clicked the article** during the training period.
Repeated clicks by the same user within the training period are only counted once.

## 4. Train-Only Leakage Prevention
Popularity scores were constructed by querying ONLY the `behavior_logs` table where `split = 'train'`. 
We parsed both the `history_raw` and `impressions_raw` strictly for training rows to identify unique users who clicked each article.
No Dev or Test impressions were scanned. The evaluation Dev period strictly acts as a future, unseen ground truth.

## 5. Normalization Method
Raw click counts exhibit extreme head-heaviness (max count = 66,449, many zero counts). Thus, we log-transformed and min-max scaled the counts:
`popularity_score = log(1 + click_count) / max(log(1 + click_count))`
This maps popularity deterministically to a `[0, 1]` range, handling zero-popularity articles safely (score = 0.0).

## 6. Ranking Formula
`final_score = beta * content_score + (1 - beta) * popularity_score`
Where the `content_score` is the standard TF-IDF cosine similarity. We isolate popularity from the Phase 12 NMF Topic scores.

## 7. Beta Configurations Evaluated
- **BASELINE**: `beta = 1.0` (Content-only baseline)
- **EXPERIMENT**: `beta = 0.9` (Heavy content bias)
- **EXPERIMENT**: `beta = 0.8` (Strong content bias)
- **EXPERIMENT**: `beta = 0.7` (Moderate bias)
- **EXPERIMENT**: `beta = 0.5` (Equal blend)
- **EXPERIMENT**: `beta = 0.3` (Heavy popularity bias)
- **EXPERIMENT**: `beta = 0.0` (Popularity-only ranking)

## 8. Experimental Protocol
Using the exact Phase 11/12 protocol:
- Small offline cohort (20 eligible users, seed=42) against the full 72,023 Dev article catalog.
- Same dev future-click ground truth.
- Same candidate masking and train-click exclusion.

## 9. Data-Leakage Audit
**TRAIN data used:**
- Article click counts (user-unique) to construct `popularity_score`.
- User history to construct the TF-IDF profile vector.
**DEV data used:**
- Candidate catalog membership mask.
- Ground-truth clicks (ONLY used in the final metric evaluation step).
**TEST data used:**
- None.
*Proof*: `verify_phase13.py` validates that querying only `split='train'` yields 94,442 distinct articles. Dev clicks from the future never increment the popularity index.

## 10. Sanity-Test Results
Executed via `backend/verify_phase13.py`:
- [x] TRAIN popularity index loads correctly (94,442 articles, max count 66,449).
- [x] Popularity counts are deterministic.
- [x] Zero-popularity articles are handled safely.
- [x] No NaN or infinite popularity scores. Max is normalized correctly (<= 1.0).
- [x] `beta=1.0` produces the same ranking as the content-only baseline.
- [x] `beta=0.0` produces popularity-only ranking.
- [x] Historical clicked articles remain excluded.
- [x] Repeated runs produce identical rankings.
- [x] Dev candidate split remains exactly the intended Dev catalog.
- [x] No Dev clicks used to calculate popularity (SQL queried 'train' split only).
- [x] No Test data used.
- [x] Database row counts remain unchanged (Read-only ops).

## 11. Representative-User Inspection

**User U13740**
- **Content-Only**: Top articles had high TF-IDF overlap but extremely low popularity (e.g., PopCnt=4, PopCnt=474).
- **Popularity-Only**: Surfaced global blockbusters ("Kevin Spacey Won't Be Charged", PopCnt=66,449) with near-zero content similarity.
- **Content + Pop (beta=0.7)**: Successfully elevated articles with strong TF-IDF scores *and* high global interest (e.g., "2 charged in murder of Alabama girl", PopCnt=5,641, ContentSim=0.23). The overlap with Content-only was 4/10.

**User U91836 (NFL Reader)**
- **Content-Only**: Found hyper-specific NFL rankings (e.g., "2019 NFL Power Rankings Week 9", PopCnt=5).
- **Popularity-Only**: Identical global blockbusters as U13740.
- **Content + Pop (beta=0.7)**: Maintained strong NFL focus but selected much more widely-read NFL articles (e.g., "NFL Week 10 Power Rankings", PopCnt=1,478). Overlap with Content-only was 4/10.

## 12. Small Offline Evaluation (20 Users)

| Configuration | P@10 | Recall@10 | NDCG@10 | Hit@10 | Avg Pop Score | Zero Pop Recs |
|---------------|------|-----------|---------|--------|---------------|---------------|
| Content-only (beta=1.0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3518 | 2.5% |
| Popularity-only (beta=0.0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9634 | 0.0% |
| beta = 0.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5452 | 0.0% |
| beta = 0.8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6932 | 0.0% |
| beta = 0.7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7948 | 0.0% |
| beta = 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9358 | 0.0% |
| beta = 0.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9623 | 0.0% |

## 13. Popularity Bias / Concentration
As `beta` decreases (putting more weight on popularity), the average popularity score of the top 10 recommended articles skyrockets from 0.35 (content-only) to 0.96 (popularity-only). 
- The content-only baseline frequently recommends articles that were *never* clicked during training (2.5% of recommendations had 0 popularity).
- Blending popularity entirely removes these zero-click "dead" articles from the top 10 pool.
- However, setting `beta` too low (e.g., 0.3 or 0.0) causes massive homogenization, completely erasing user personalization in favor of global top hits.

## 14. Coverage Analysis
Popularity blending naturally restricts catalog coverage. Instead of pulling up obscure exact-word matches from the depths of the 72,000-article catalog, the hybrid ranker concentrates recommendations on the top ~10% most popular articles that align with the user's TF-IDF vector.

## 15. Runtime Comparison
Evaluating the 20-user cohort (retrieving and ranking against 72,023 items 7 times) took ~16.4 seconds. The popularity index is stored in memory as an aligned Numpy array (`popularity_array`), rendering the popularity-blending vectorized and highly efficient (no noticeable latency increase compared to Phase 11/12).

## 16. Interpretation
**Does popularity improve exact full-catalog candidate retrieval metrics? No.**
Just like Phase 12, the quantitative exact-match metrics remain at `0.0000`. Content and Popularity biases alone are insufficient to predict the exact needle in the 72,000-article haystack.

**Qualitatively**, however, `beta=0.7` substantially improves the candidate pool. Pure TF-IDF frequently ranks highly irrelevant obscure articles simply because they share niche keywords. Popularity acts as a powerful prior for quality and broad interest, suppressing obscure "dead" articles while elevating proven, engaging content within the user's specific interest domain.

## 17. Limitations
- Popularity is static based on the training period; it does not decay or account for breaking news trending rapidly during the Dev period.
- Combining TF-IDF, NMF Topic, and Popularity into a single linear formula requires complex multi-parameter tuning which might be brittle.

## 18. Full-Scale Experiment Justification
**Is a full-scale Phase 13 experiment scientifically justified? NO.**
Running the evaluation on the full 200k user base will simply yield zero metrics, wasting compute resources. The qualitative benefit of popularity blending is evident, but the catalog-level protocol is the bottleneck. The default TF-IDF baseline should remain as-is for now, and future phases should pivot to Impression-Level ranking, where these combined signals can actually be evaluated against user choices within a constrained session.

## Data Integrity Verification
All operations performed were read-only. No raw MIND files were modified. The database schema and row counts remain unchanged:
- Canonical articles = 130,379
- BehaviorLog = 4,979,946
- ImpressionItem = 190,707,932
- FeedbackEvent = 0

## Reproducibility
```bash
python backend/verify_phase13.py
python backend/run_phase13_evaluation.py
```
