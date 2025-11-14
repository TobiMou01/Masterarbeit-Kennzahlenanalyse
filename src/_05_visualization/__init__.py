"""
Visualization Phase - Comprehensive Plotting Engines
"""

from . import plot_engine
from . import plot_engine_insights

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
    'plot_engine_insights',
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
