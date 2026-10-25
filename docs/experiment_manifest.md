# Experiment Manifest

| Phase | Purpose | Primary Artifact | Dataset / Split | Evaluation Population | Status | Important Limitation |
|---|---|---|---|---|---|---|
| **Phase 1-10** | Core pipeline construction | `backend/app/` source code | MIND Train / Dev | N/A | COMPLETE | Heuristic feedback weights are not learned end-to-end. |
| **Phase 11** | Evaluation framework sanity check | `phase11_catalog_sanity_report.md` | MIND Dev | 20 users (Catalog) | COMPLETE | Catalog-level exact future-click retrieval yielded extremely low metric ceilings. |
| **Phase 12** | Hybrid NMF/TF-IDF experiment | `phase12_hybrid_retrieval_report.md` | MIND Dev | 20 users (Catalog) | COMPLETE | Small sample size; exact future click matching remained difficult. |
| **Phase 13** | Popularity-aware experiment | `phase13_popularity_retrieval_report.md` | MIND Dev | 20 users (Catalog) | COMPLETE | Popularity increased hit rate artificially but degraded personalized profile relevance. |
| **Phase 14** | Impression-level evaluation protocol | `phase14_impression_level_evaluation_report.md` | MIND Dev | 100 users, 148 sessions | COMPLETE | 100 users is too small for tight confidence intervals on tiny NDCG differences. |
| **Phase 15** | Full-scale offline evaluation | `phase15_large_scale_partial_evaluation_report.md` | MIND Dev | 93,612 users (43.18%) | PARTIAL | 123,166 eligible users were not evaluated due to deterministic checkpoint termination. |
| **Phase 16** | Rigorous statistical analysis | `phase16_statistical_analysis_report.md` | Phase 15 Checkpoint | 93,612 users | COMPLETE | Effect sizes were incredibly small despite high statistical significance. |
