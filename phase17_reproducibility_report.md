# Phase 17 Final Reproducibility, Audit, and Research Package Validation Report

## 1. Executive Summary
This report formalizes the final readiness, security, and reproducibility state of the *Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval* project. Extensive read-only audits have confirmed the provenance of the MIND data, the correctness of the experimental pipelines, the security of the codebase, and the statistical validity of the claims made.

## 2. Project Inventory
- **Backend Source:** `backend/app/`
- **Frontend Source:** `frontend/`
- **Database:** `data/news_discovery.db`
- **Evaluation Checkpoints:** `data/evaluation/phase15/raw_session_results.jsonl`
- **Verification Scripts:** `backend/verify_phase16.py`, `backend/verify_phase17.py`, `backend/api_smoke_test.py`
- **Reports:** `phase11_*.md`, `phase12_*.md`, `phase13_*.md`, `phase14_*.md`, `phase15_*.md`, `phase16_*.md`

## 3. Architecture
- **DB/Model Layer:** SQLite for metadata/history; Scikit-Learn for TF-IDF sparse matrices and NMF components.
- **Unified Pipeline:** Profile Generation → Exponential Drift Decay → Similarity Ranking (Cosine) → Popularity Bias Reranking → MMR Diversification.

## 4. Data Provenance
- Dataset: Microsoft MIND (Dev/Train splits).
- Canonical Articles: 130,379.
- Behavior Logs: 4,979,946.
- The pipeline securely maintains dataset boundaries. No duplicate IDs exist. No orphan impressions exist.

## 5. Environment
- **Runtime:** Python 3.10+
- **Major Libraries:** FastAPI, Scikit-Learn, SciPy, NumPy, SQLite3.
- **Hardware Profile:** Fully CPU-bound. **GPU is NOT required for any part of this pipeline.**

## 6. Reproduction Commands
1. DB init/ingestion: Not run this phase, standard scripts.
2. Index construction: Not run this phase, pre-computed.
3. API execution: `python -m uvicorn app.main:app --port 8000`
4. Phase 15 Evaluation: `python backend/run_phase15_full_evaluation.py` (Approx runtime: ~9.5 hours for 93k users).
5. Phase 16 Analysis: `python backend/phase16_analysis.py` (Approx runtime: 30 seconds).

## 7. Core Pipeline Validation
All read-only recommendation path endpoints (health, representation, candidates, recommendations, feedback) successfully returned HTTP 200 during active validation. Uninitialized profiles gracefully fell back without crashing the server.

## 8. Leakage Audit
- **PASS**: Training interactions strictly precede Dev evaluation.
- **PASS**: Dev clicks were entirely excluded from user profile embeddings.
- **PASS**: Test clicks were never accessed.
- **PASS**: Future interactions are demonstrably filtered out.

## 9. Phase 11-16 Artifact Audit
- **Phase 11**: Verified 20-user catalog evaluation ceilings.
- **Phase 12**: Verified Topic Matrix existence and hybrid experiments.
- **Phase 13**: Verified train-only popularity bias evaluation.
- **Phase 14**: Verified 100-user impression protocols.
- **Phase 15**: Checked 93,612 users (43.18% coverage).
- **Phase 16**: Checked metric corrections, paired tests, Holm adjustments, and robustness distributions.

## 10. Statistical Claim Audit
All claims of "universal" or "strictly necessary" superiority have been rigorously bounded. Specifically, we replaced absolute claims with empiric descriptions: *Within the evaluated 93,612 user sample, TF-IDF achieved statistically significantly higher NDCG@10 than the Topic-only baseline.*

## 11. Research Findings
| Finding | Observed Result | Interpretation |
|---|---|---|
| TF-IDF Baseline | NDCG@10 = 0.3658 | Strong baseline due to exact lexical matching. |
| Hybrid alpha=0.5 | NDCG@10 = 0.3689 | Slight but statistically significant improvement over baseline. |
| MMR Application | NDCG@10 = 0.3602, ILD = 0.9959 | Increased diversity (ILD) explicitly taxes strict relevance (NDCG). |
| Popularity Bias | NDCG@10 = 0.3234, Pop = 0.4475 | High popularity radically degrades personalized semantic relevance. |

## 12. Limitations
- Phase 15 is a partial, not full-population evaluation.
- Test splits inherently lack click labels (preventing full test-set NDCG).
- Explanation quality metrics measure consistency, not necessarily human interpretability.

## 13. Security/Data Safety
- No API keys or tokens are hardcoded.
- The repository `.gitignore` explicitly prevents pushing MIND dataset TSVs or `.db` files.
- No raw MIND records are exposed in the markdown evaluation reports.

## 14. Repository Readiness
The project directory has been evaluated. Recommendations for standardizing the open-source directory structure have been documented in `phase17_repository_recommendation.md`.

## 15. Reproducibility Status
**PHASE 17 STATUS**
Source inventory: PASS
Data audit: PASS
Environment audit: PASS
Core pipeline: PASS
Leakage audit: PASS
Evaluation artifact consistency: PASS
Statistical claim audit: PASS
Security audit: PASS
Documentation: PASS
Repository readiness: PASS
Final verification: PASS

**OVERALL: COMPLETE**
