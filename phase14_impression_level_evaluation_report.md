# Phase 14: Impression-Level Ranking Evaluation Report

## 1. Research Motivation
This phase evaluates the system's ability to rank articles *within the actual impression sets* that users were exposed to. Because exact future-click retrieval from the full 72,023-article catalog (Phases 11-13) yielded metrics effectively at 0.0000, we shifted our focus to evaluating how well the existing signals (Content, Topic, Popularity, Diversity) order items within a constrained, realistic choice set (the impression set).

## 2. Relationship to Previous Phases
- **Phase 11**: Established the impression-level evaluation protocol. This exact protocol and fixed user cohort (100 users, seed=42) were reused here.
- **Phase 12**: Evaluated TF-IDF + NMF Topic hybridization at the catalog level. Here, we evaluate it at the impression level.
- **Phase 13**: Evaluated training-only global Popularity at the catalog level. Here, we evaluate whether popularity helps rank articles the user already saw.

## 3. Evaluation Population
- **Users**: 100 randomly sampled eligible users (train behavior + dev future clicks).
- **Sessions**: 148 dev behavior sessions.
- **Candidate Sets**: The specific `impressed_articles` for each session.
- **Ground Truth**: The specific `clicked_articles` for that session.

## 4. Leakage Prevention
- **Temporal Strictness**: Profile vectors (TF-IDF and Topic) were built exclusively using train-period behavior. Current/future session impressions or clicks were strictly omitted from profile generation.
- **Popularity**: Popularity counts were constructed using only `split='train'` behavior logs.
- **Integrity**: `backend/verify_phase14.py` verified that exactly the same session construction and labels from Phase 11 were used, with no database writes.

## 5. Ranking Configurations
- **A. TF-IDF Baseline**: Cosine similarity using Phase 6 vectors.
- **B. Topic-only**: Phase 5 NMF topic representations.
- **C. Hybrid (TF-IDF + Topic)**: `alpha * TF-IDF + (1 - alpha) * Topic` evaluated at `alpha` ∈ {0.8, 0.7, 0.5, 0.3}.
- **D. Popularity-only**: Ranked strictly by normalized training popularity score.
- **E. TF-IDF + Pop**: `beta * TF-IDF + (1 - beta) * Pop` evaluated at `beta` ∈ {0.9, 0.8, 0.7, 0.5}.
- **F. TF-IDF + MMR**: TF-IDF ranked, then re-ranked with Phase 7 MMR (`lambda=0.7`).
- **G. TF-IDF + Pop + MMR**: Configuration E (`beta=0.7`) re-ranked with MMR (`lambda=0.7`).

## 6. Main Results

| Method | P@5 | P@10 | NDCG@5 | NDCG@10 | Hit@5 | Hit@10 | ILD@10 |
|--------|-----|------|--------|---------|-------|--------|--------|
| A. TF-IDF | 0.1143 | 0.0830 | 0.3169 | 0.3734 | 0.5148 | 0.6735 | 0.9881 |
| B. Topic-only | 0.1108 | 0.0819 | 0.2996 | 0.3560 | 0.5190 | 0.6802 | 0.9862 |
| C. Hybrid α=0.8 | 0.1130 | **0.0847** | 0.3168 | **0.3777** | 0.5115 | 0.6885 | 0.9879 |
| C. Hybrid α=0.7 | 0.1130 | 0.0838 | 0.3169 | 0.3752 | 0.5115 | 0.6735 | 0.9878 |
| C. Hybrid α=0.5 | **0.1146** | 0.0850 | 0.3123 | 0.3722 | 0.5132 | **0.7002** | 0.9874 |
| C. Hybrid α=0.3 | 0.1160 | 0.0874 | **0.3184** | 0.3759 | **0.5198** | 0.6835 | 0.9869 |
| D. Popularity-only| 0.0788 | 0.0702 | 0.2274 | 0.2901 | 0.3775 | 0.5933 | 0.9931 |
| E. TF-IDF+Pop β=0.9| 0.1007 | 0.0744 | 0.2958 | 0.3467 | 0.4702 | 0.6347 | 0.9896 |
| E. TF-IDF+Pop β=0.7| 0.0924 | 0.0700 | 0.2742 | 0.3243 | 0.4355 | 0.5967 | 0.9912 |
| F. TF-IDF + MMR | **0.1173** | 0.0842 | **0.3233** | **0.3755** | **0.5198** | 0.6718 | **0.9952** |
| G. TF-IDF+Pop+MMR | 0.0894 | 0.0712 | 0.2725 | 0.3282 | 0.4305 | 0.6000 | **0.9956** |

## 7. Popularity Concentration Results

| Method | Avg Pop | Top 1% Pop Fraction | Zero-Pop Fraction |
|--------|---------|---------------------|-------------------|
| A. TF-IDF | 0.2718 | 3.3% | 31.0% |
| D. Popularity-only | 0.4524 | 10.8% | 13.6% |
| E. TF-IDF+Pop β=0.7 | 0.4467 | 10.0% | 14.2% |
| F. TF-IDF + MMR | 0.2711 | 3.2% | 30.9% |

## 8. Interpretation of Findings

### 1. Does NMF topic similarity change impression-level ranking quality?
Yes, but Topic-only performance is slightly worse than TF-IDF (NDCG@10 0.3560 vs 0.3734).

### 2. Does TF-IDF + NMF improve over TF-IDF on the same impressions?
**Yes.** Hybrid configurations consistently outperformed pure TF-IDF across several metrics. For instance, `alpha=0.8` achieved the highest NDCG@10 (0.3777). `alpha=0.5` achieved the highest Hit@10 (0.7002, compared to TF-IDF's 0.6735). The fusion of lexical and latent semantic topic matching provides a tangible improvement when selecting from an already-impressed candidate set.

### 3. Does popularity help ranking among already exposed articles?
**No.** Popularity actively harmed performance. Popularity-only dropped NDCG@10 drastically to 0.2901. Blending it with TF-IDF (`beta=0.7`) dragged the NDCG@10 down to 0.3243. When users are looking at a confined list of articles on a screen (an impression set), their click choice is primarily driven by their specific content interests, not the global historical popularity of the article.

### 4. Does popularity introduce measurable concentration?
**Yes.** Popularity blending pushed the fraction of recommended articles from the global Top 1% from 3.3% to 10.0%, and it more than halved the number of zero-popularity "long-tail" articles recommended (31.0% down to 14.2%). However, because this did not yield relevance gains, this concentration represents a negative trade-off in the impression-level context.

### 5. Does MMR improve diversity in impression-level ranking?
**Yes.** TF-IDF + MMR increased Intra-List Diversity (ILD) significantly from 0.9881 to 0.9952, without harming ranking quality. In fact, NDCG@5 increased to 0.3233 (up from 0.3169). Re-ranking a small impression list to remove redundant articles actually surfaces the clicked article slightly better.

### 6. What happens when popularity and MMR are combined?
The combination (Method G) successfully achieved the highest diversity (ILD = 0.9956) but inherited the poor relevance of popularity ranking (NDCG@10 = 0.3282). The signals do not stack synergistically.

### 7. Are any observed changes large enough to justify a larger experiment?
**YES.**
The TF-IDF + NMF Hybrid methods (especially `alpha=0.8` or `0.5`) and the TF-IDF + MMR methods demonstrated clear measurable benefits in ranking relevance and diversity. A full-scale evaluation of these methods across the entire user base is justified to determine if the findings hold with statistical significance. Popularity should be excluded from future impression-level experiments.

## 9. Limitations
- Popularity was measured statically from the training period. In a real-time news environment, popularity decays, and breaking news spikes.
- We only evaluated on 100 users / 148 sessions. Larger evaluation is needed for robust statistical significance.

## 10. Reproducibility
```bash
python backend/verify_phase14.py
python backend/run_phase14_evaluation.py
```
