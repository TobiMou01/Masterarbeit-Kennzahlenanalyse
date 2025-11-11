"""
Validation Module - Algorithm Comparison and External Validation

Provides tools for:
- Comparing clustering results across different algorithms (ARI, Confusion Matrix)
- Validating clusters against external labels (Cramér's V, Chi²-Test)
- Identifying consensus clusters and disagreement cases
"""

from .algorithm_comparison import AlgorithmComparison
from .external_validation import ExternalValidation
from .validation_runner import perform_validation, add_external_labels, run_pca_validation

__all__ = [
    'AlgorithmComparison',
    'ExternalValidation',
    'perform_validation',
    'add_external_labels',
    'run_pca_validation'
]
