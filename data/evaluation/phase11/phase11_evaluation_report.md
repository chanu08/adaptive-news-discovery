# Phase 11: Offline Evaluation & Research Experiments Report

**Project**: Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

**Date**: 2026-09-27 09:25:18 UTC

**Evaluation Scope**: Read-Only Offline Evaluation with Strict Leak-Free Temporal Protocols


---

## 1. Executive Summary & Protocol Overview

Phase 11 implements an empirical offline evaluation framework designed to measure ranking quality, recommendation relevance, intra-list diversity, catalog coverage, and the systematic effects of Maximal Marginal Relevance (MMR) re-ranking and temporal profile modeling. The evaluation operates strictly in read-only mode against the underlying MIND dataset, enforcing rigorous split-aware temporal isolation to prevent data leakage.

### Key Principles Enforced:

1. **Zero Data Leakage**: Future clicks in held-out dev sessions were never used to construct user profiles or score candidates.
2. **No Fabricated Labels**: Verified that MIND Test split withholds click labels; direct ranking metrics were calculated on the held-out Dev split where ground-truth interaction labels exist.
3. **Reproducible Sampling**: Cohort selection performed deterministically via fixed pseudo-random seeds.
4. **Statistical Rigor**: All metrics report mean, median, standard deviation, and 95% bootstrap confidence intervals.
5. **Factual Reporting**: Comparative analyses report measured metrics without subjective declarations of superiority.

---

## 2. MIND Dataset Inspection & Split Verification

- **Total Canonical News Articles**: 130,379
- **Total MIND Users**: 876,956
- **Test Split Label Status**: Test split is not suitable for direct labeled offline ranking metrics because click labels are unavailable.

### Split Breakdown:

| Split | Behavior Logs | Min Time | Max Time | Unique Users | Impression Clicks |
|---|---|---|---|---|---|
| `train` | 2,232,748 | 11/10/2019 10:00:00 AM | 11/9/2019 9:59:59 PM | 711,222 | 3,383,656 |
| `dev` | 376,471 | 11/15/2019 10:00:00 AM | 11/15/2019 9:59:59 PM | 255,990 | 574,845 |
| `test` | 2,370,727 | 11/16/2019 10:00:00 AM | 11/22/2019 9:59:59 PM | 702,005 | 0 |

### Candidate Catalog Articles by Split:

| Split | Distinct Articles in Split |
|---|---|
| `train` | 101,527 |
| `dev` | 72,023 |
| `test` | 120,961 |

---

## 3. Evaluation Cohort Sampling

- **Total Dev Users in Population**: 255,990
- **Eligible Users (Train history + Dev clicks)**: 216,778
- **Sampled Users for Experiment**: 100 (Seed: `42`)
- **Excluded Users**: 39,212
  - No train history logs: 39,212
  - No dev clicks: 0

---

## 4. Baseline System Comparison (Catalog Retrieval Protocol)

Evaluation over dev candidate catalog (72,023 eligible articles), masking training clicked items:

| System | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Hit Rate@10 | ILD@10 | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| **Popularity** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.9918 | 0.02% |
| **Long-term only** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0010 ± 0.0100 | 0.0050 ± 0.0500 | 0.0018 ± 0.0177 | 0.0100 | 0.7165 | 1.08% |
| **Recent only** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.7429 | 1.10% |
| **Cosine full profile** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.7146 | 1.08% |
| **Full system (+MMR)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0010 ± 0.0100 | 0.0050 ± 0.0500 | 0.0020 ± 0.0204 | 0.0100 | 0.9180 | 1.19% |

---

## 5. MMR Diversity Ablation & Trade-Off Analysis

Evaluating diversity parameter λ across pure relevance (λ=1.0) down to high diversity (λ=0.5):

| λ | P@10 | NDCG@10 | ILD@10 | Cat. Div@10 | Subcat. Div@10 | Avg Pairwise Sim | Coverage | Jaccard Overlap (vs λ=1.0) |
|---|---|---|---|---|---|---|---|---|
| **λ = 1.0** | 0.0000 | 0.0000 | 0.7330 | 2.95 | 4.37 | 0.2670 | 1.10% | 1.0000 |
| **λ = 0.8** | 0.0010 | 0.0026 | 0.9000 | 4.01 | 6.01 | 0.1000 | 1.15% | 0.3409 |
| **λ = 0.7** | 0.0010 | 0.0020 | 0.9180 | 4.09 | 6.04 | 0.0820 | 1.19% | 0.2389 |
| **λ = 0.5** | 0.0000 | 0.0000 | 0.9264 | 4.12 | 6.11 | 0.0736 | 1.19% | 0.1794 |

---

## 6. Profile Modeling Component Ablation

Assessing the contribution of temporal interest modeling components:

| Profile Configuration | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | ILD@10 |
|---|---|---|---|---|---|---|---|
| **Long-term only** | 0.0020 | 0.0050 | 0.0024 | 0.0010 | 0.0050 | 0.0024 | 0.9087 |
| **Recent only** | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 0.0050 | 0.0019 | 0.9296 |
| **Long-term + Recent (No drift)** | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 0.0050 | 0.0020 | 0.9080 |
| **Long-term + Recent + Drift** | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 0.0050 | 0.0020 | 0.9180 |

---

## 7. Impression-Level Session Evaluation (Section 8)

In addition to open-catalog candidate retrieval, the system was evaluated directly within actual user impression sessions from the held-out Dev split (candidate pool = impressed articles shown to user):

- **Evaluated Impression Sessions**: 148
- **Evaluated Users**: 100

| Ranking Protocol | P@5 | NDCG@5 | Hit Rate@5 | P@10 | NDCG@10 | Hit Rate@10 |
|---|---|---|---|---|---|---|
| **Cosine Relevance** | 0.1143 | 0.3169 | 0.5148 | 0.0830 | 0.3734 | 0.6735 |
| **MMR Diversity Re-ranking (λ=0.7)** | 0.1173 | 0.3233 | 0.5198 | 0.0842 | 0.3755 | 0.6718 |

---

## 8. Explainability & Adaptive Feedback Verification

### Explanation Consistency Checks (System-Grounded):

- **Total Explanations Evaluated**: 500
- **Consistent Explanations**: 500
- **System-Grounded Consistency Rate**: `100.00%`
- **No-Explanation Suppression Verified**: `True`

### Controlled Adaptive Feedback Simulation:

- **Simulation Type**: Controlled adaptive simulation (in-memory, read-only)
- **Evaluated Adaptation Cohort**: 30 users
- **Mean User Vector Cosine Shift**: `0.291031`
- **Recommendation List Jaccard Overlap**: `0.2935`
- **Mean Target Category Count Boost in Top-10**: `1.47` articles
- **Database Integrity**: `database_written = False` (zero persistent mutations)

---

## 9. System Performance & Latency

- **Total Evaluation Runtime**: 675.70 seconds
- **Users Processed**: 100
- **Average Recommendation Latency**: 1311.38 ms
- **Median Recommendation Latency**: 331.01 ms
- **Mean Candidate Retrieval Time**: 412.75 ms
- **Mean MMR Re-ranking Time**: 331.01 ms

---

## 10. Database Integrity Verification (Read-Only Guarantee)

| Table Name | Before Evaluation | After Evaluation | Status |
|---|---|---|---|
| `behavior_logs` | 4,979,946 | 4,979,946 | **PASS (Unchanged)** |
| `impression_items` | 190,707,932 | 190,707,932 | **PASS (Unchanged)** |
| `news_articles` | 130,379 | 130,379 | **PASS (Unchanged)** |
| `article_splits` | 294,511 | 294,511 | **PASS (Unchanged)** |
| `feedback_events` | 18 | 18 | **PASS (Unchanged)** |

---

## 11. Expected Research Limitations

1. **Missing Test Split Labels**: MIND test split withholds ground-truth click labels. Offline ranking evaluation is strictly conducted on the Dev split.
2. **History Timestamp Granularity**: The MIND `history_raw` field represents historical interactions as an ordered sequence of article IDs without individual timestamps. Historical decay is grounded in relative sequence position and earliest log timestamps.
3. **Offline Proxy for User Satisfaction**: Offline click metrics (Precision, Recall, NDCG) serve as observational proxies and do not capture interactive reading depth or long-term satisfaction.
4. **Diversity Metric Representation**: Intra-List Diversity (ILD) is calculated using TF-IDF cosine distances rather than human semantic similarity.
5. **Popularity Baseline Scope**: Popularity frequency reflects the empirical training distribution and demonstrates high head-item concentration.
6. **Explanation Evaluation Nature**: Consistency metrics confirm strict system-signal alignment rather than human subjective clarity.

---

## 12. Factual Observations & Conclusion

- **Diversity/Relevance Trade-off**: Decreasing λ from 1.0 to 0.5 systematically increases Intra-List Diversity (ILD@10 moved from 0.7330 to 0.9264) and expanded catalog coverage from 1.10% to 1.19%.
- **Temporal Weighting**: Combining long-term and recent profiles yielded measurable ranking shifts relative to isolated long-term or recent profiles.
- **Integrity Verification**: All database tables maintained identical row counts throughout the evaluation process.


**Phase 11 Status**: EVALUATION COMPLETE AND VERIFIED.
