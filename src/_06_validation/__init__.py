"""
Validation Phase - External Validation and Algorithm Comparison
"""

# Legacy imports (backward compatible)
from .algorithm_comparison import AlgorithmComparison  # Legacy
from .external_validation import ExternalValidation  # Legacy

# NEW: Modular validation
from .base_validation import BaseValidation
from .gics_validation import GICSValidation
from .size_validation import SizeValidation
from .comparison_metrics import ComparisonMetrics
from .comparison_analyzer import ComparisonAnalyzer

# Pipeline runners
from .validation_runner import perform_validation, add_external_labels, run_pca_validation

__all__ = [
    # Legacy (backward compatible)
    'AlgorithmComparison',
    'ExternalValidation',
    # New modular validation
    'BaseValidation',
    'GICSValidation',
    'SizeValidation',
    'ComparisonMetrics',
    'ComparisonAnalyzer',
    # Pipeline helpers
    'perform_validation',
    'add_external_labels',
    'run_pca_validation',
]
