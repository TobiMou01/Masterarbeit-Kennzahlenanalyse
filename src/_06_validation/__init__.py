"""
Validation Module - Algorithm Comparison and External Validation

Provides tools for:
- Comparing clustering results across different algorithms (ARI, Confusion Matrix)
- Validating clusters against external labels (Cramér's V, Chi²-Test)
- Identifying consensus clusters and disagreement cases
"""

from .algorithm_comparison import AlgorithmComparison
from .external_validation import ExternalValidation

__all__ = ['AlgorithmComparison', 'ExternalValidation']
