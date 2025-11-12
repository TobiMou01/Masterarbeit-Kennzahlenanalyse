"""
Visualization Phase - Comprehensive Plotting Engines
"""

from . import plot_engine
from . import plot_engine_scores  # Legacy - use specific modules below
from . import plot_engine_validation  # Legacy
from . import plot_engine_pca  # Legacy
from . import plot_engine_insights  # Legacy

# NEW: Modular plot engines
from . import plot_engine_score_distributions
from . import plot_engine_score_evolution
from . import plot_engine_score_analysis
from . import plot_engine_validation_metrics
from . import plot_engine_validation_matrices
from . import plot_engine_pca_variance
from . import plot_engine_pca_loadings
from . import plot_engine_pca_clusters
from . import plot_engine_company_insights
from . import plot_engine_algorithm_congruence

__all__ = [
    'plot_engine',
    # Legacy (kept for backward compatibility)
    'plot_engine_scores',
    'plot_engine_validation',
    'plot_engine_pca',
    'plot_engine_insights',
    # New modular engines
    'plot_engine_score_distributions',
    'plot_engine_score_evolution',
    'plot_engine_score_analysis',
    'plot_engine_validation_metrics',
    'plot_engine_validation_matrices',
    'plot_engine_pca_variance',
    'plot_engine_pca_loadings',
    'plot_engine_pca_clusters',
    'plot_engine_company_insights',
    'plot_engine_algorithm_congruence',
]
