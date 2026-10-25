# Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

## 1. Project Title
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval

## 2. Research Motivation
Modern news recommendation systems often suffer from echo chambers and over-specialization. This project explores whether hybridizing explicit lexical features (TF-IDF) with latent topic distributions (NMF), coupled with Maximal Marginal Relevance (MMR) and popularity-aware reranking, can balance personalization, diversity, and trending awareness. Furthermore, we explicitly model temporal interest drift to separate long-term static preferences from short-term volatile news consumption.

## 3. Architecture
- **Backend:** Python FastAPI, SQLite (storage), Scikit-Learn (TF-IDF/NMF vectorization), SciPy (cosine similarity, statistics).
- **Frontend:** HTML/JS/Vanilla CSS Dashboard for visual debugging.
- **Evaluation Engine:** Custom deterministic offline impression-level reranking pipeline.

## 4. Core Pipeline
1. **User Profile:** Aggregates vectors of historically clicked articles.
2. **Interest Drift:** Applies exponential time-decay weighting to differentiate recent clicks from old clicks.
3. **Candidate Retrieval:** Filters eligible candidates based on the impression slate.
4. **Similarity Ranking:** Computes Cosine Similarity between candidate vectors and the drifted user profile.
5. **MMR Re-ranking:** Diversifies the top-K list by penalizing redundancy.
6. **Explanation:** Maps intersecting vector components to human-readable keywords/topics.
7. **Final Recommendation:** Presents the ranked, diverse slate.

## 5. Dataset & Licensing Restrictions
This project utilizes the **Microsoft MIND dataset**.
* MIND is research data distributed under Microsoft Research licensing terms.
* The dataset must not be redistributed as part of this project.
* Raw MIND data should remain external to any public source repository.
* Derived artifacts must be checked before publication.
* Research results may be published without redistributing a material/substantial portion of the dataset, subject to the applicable license terms.

## 6. Installation
```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

## 7. Configuration
No GPU is required for this pipeline. The entire system is optimized to run on CPU using Scikit-Learn sparse matrices and SQLite. 

## 8. Running the backend
```bash
cd backend
python -m uvicorn app.main:app --port 8000
```

## 9. Running the frontend
Open `frontend/index.html` in any web browser. 

## 10. Building the TF-IDF/NMF representation
```bash
python scripts/ingestion/build_tfidf_nmf.py
```

## 11. Generating user profiles
```bash
python scripts/ingestion/generate_profiles.py
```

## 12. Running recommendations
Recommendations are served via the `/api/users/mind/{user_id}/recommendations` endpoint.

## 13. Running evaluations
```bash
python backend/run_phase15_full_evaluation.py
```

## 14. Reproducing statistical analysis
```bash
python backend/phase16_analysis.py
```

## 15. Current Evaluation Coverage
Large-scale impression-level evaluation currently covers 93,612 of 216,778 eligible users (43.18%); 123,166 eligible users remain unevaluated.

## 16. Known Limitations
- Evaluation is partial, not full-population evaluation.
- MIND test split has no click labels.
- Historical MIND history does not provide individual timestamps for every historical click; temporal decay uses recoverable timestamps only.
- Impression-level evaluation measures ranking strictly within exposed impressions.
- ILD is based on TF-IDF representation alone.
- Explanation consistency is not the same as human-perceived explanation quality.
- Feedback evaluation is a controlled simulation, not real longitudinal deployment.
- Heuristic feedback weights are not learned.
- No claim of universal optimal hyperparameters is made.
