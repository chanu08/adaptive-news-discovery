"""
Offline Evaluation & Research Experiments Package — Phase 11
Adaptive News Discovery Using Interest Drift and Diversity-Aware Information Retrieval
"""

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    category_diversity,
    subcategory_diversity,
    catalog_coverage,
    compute_bootstrap_ci
)
from evaluation.datasets import MINDDatasetInspector, EvaluationCohort
from evaluation.protocols import (
    CatalogCandidateRetrievalProtocol,
    ImpressionLevelRerankingProtocol
)
from evaluation.baselines import (
    TrainingPopularityRanker,
    build_user_vector_from_sequence
)
from evaluation.experiments import EvaluationExperimentRunner
from evaluation.reporting import EvaluationReportGenerator

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "intra_list_diversity",
    "category_diversity",
    "subcategory_diversity",
    "catalog_coverage",
    "compute_bootstrap_ci",
    "MINDDatasetInspector",
    "EvaluationCohort",
    "CatalogCandidateRetrievalProtocol",
    "ImpressionLevelRerankingProtocol",
    "TrainingPopularityRanker",
    "build_user_vector_from_sequence",
    "EvaluationExperimentRunner",
    "EvaluationReportGenerator"
]
