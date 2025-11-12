"""
Plot Engine Validation Wrapper
Combines matrices and metrics engines for backward compatibility
"""

from src._05_visualization.plot_engine_validation_matrices import PlotEngineValidationMatrices
from src._05_visualization.plot_engine_validation_metrics import PlotEngineValidationMetrics


class PlotEngineValidation:
    """
    Unified wrapper for validation plotting engines

    Combines:
    - PlotEngineValidationMatrices (confusion matrices, contingency tables)
    - PlotEngineValidationMetrics (ARI heatmaps, Cramér's V, Chi²)

    Provides backward compatibility with legacy plot_engine_validation module
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize unified validation plotting engine

        Args:
            style: Seaborn style
            dpi: Resolution for saved plots
        """
        self.matrices = PlotEngineValidationMatrices(style=style, dpi=dpi)
        self.metrics = PlotEngineValidationMetrics(style=style, dpi=dpi)
        self.dpi = dpi

    # =========================================================================
    # MATRICES METHODS (from PlotEngineValidationMatrices)
    # =========================================================================

    def plot_confusion_matrix(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMatrices"""
        return self.matrices.plot_confusion_matrix(*args, **kwargs)

    def plot_multiple_confusion_matrices(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMatrices"""
        return self.matrices.plot_multiple_confusion_matrices(*args, **kwargs)

    def plot_contingency_heatmap(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMatrices"""
        return self.matrices.plot_contingency_heatmap(*args, **kwargs)

    # =========================================================================
    # METRICS METHODS (from PlotEngineValidationMetrics)
    # =========================================================================

    def plot_ari_heatmap(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMetrics"""
        return self.metrics.plot_ari_heatmap(*args, **kwargs)

    def plot_cramers_v_comparison(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMetrics"""
        return self.metrics.plot_cramers_v_comparison(*args, **kwargs)

    def plot_chi_square_significance(self, *args, **kwargs):
        """Delegate to PlotEngineValidationMetrics"""
        return self.metrics.plot_chi_square_significance(*args, **kwargs)
