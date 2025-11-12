"""
Plot Engine PCA Wrapper
Combines variance, loadings, and clusters engines for backward compatibility
"""

from src._05_visualization.plot_engine_pca_variance import PlotEnginePCAVariance
from src._05_visualization.plot_engine_pca_loadings import PlotEnginePCALoadings
from src._05_visualization.plot_engine_pca_clusters import PlotEnginePCAClusters


class PlotEnginePCA:
    """
    Unified wrapper for PCA plotting engines

    Combines:
    - PlotEnginePCAVariance (scree plots, cumulative variance)
    - PlotEnginePCALoadings (loadings heatmap, biplot, feature contributions)
    - PlotEnginePCAClusters (cluster separation, 2D/3D scatter)

    Provides backward compatibility with legacy plot_engine_pca module
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize unified PCA plotting engine

        Args:
            style: Seaborn style
            dpi: Resolution for saved plots
        """
        self.variance = PlotEnginePCAVariance(style=style, dpi=dpi)
        self.loadings = PlotEnginePCALoadings(style=style, dpi=dpi)
        self.clusters = PlotEnginePCAClusters(style=style, dpi=dpi)
        self.dpi = dpi

    # =========================================================================
    # VARIANCE METHODS (from PlotEnginePCAVariance)
    # =========================================================================

    def plot_scree_plot(self, *args, **kwargs):
        """Delegate to PlotEnginePCAVariance"""
        return self.variance.plot_scree_plot(*args, **kwargs)

    def plot_cumulative_variance_bar(self, *args, **kwargs):
        """Delegate to PlotEnginePCAVariance"""
        return self.variance.plot_cumulative_variance_bar(*args, **kwargs)

    # =========================================================================
    # LOADINGS METHODS (from PlotEnginePCALoadings)
    # =========================================================================

    def plot_component_loadings_heatmap(self, *args, **kwargs):
        """Delegate to PlotEnginePCALoadings"""
        return self.loadings.plot_component_loadings_heatmap(*args, **kwargs)

    def plot_biplot(self, *args, **kwargs):
        """Delegate to PlotEnginePCALoadings"""
        return self.loadings.plot_biplot(*args, **kwargs)

    def plot_feature_contribution_polar(self, *args, **kwargs):
        """Delegate to PlotEnginePCALoadings"""
        return self.loadings.plot_feature_contribution_polar(*args, **kwargs)

    def plot_pca_vs_original_comparison(self, *args, **kwargs):
        """Delegate to PlotEnginePCALoadings"""
        return self.loadings.plot_pca_vs_original_comparison(*args, **kwargs)

    def plot_reconstruction_error_heatmap(self, *args, **kwargs):
        """Delegate to PlotEnginePCALoadings"""
        return self.loadings.plot_reconstruction_error_heatmap(*args, **kwargs)

    # =========================================================================
    # CLUSTERS METHODS (from PlotEnginePCAClusters)
    # =========================================================================

    def plot_cluster_separation_in_pca_space(self, *args, **kwargs):
        """Delegate to PlotEnginePCAClusters"""
        return self.clusters.plot_cluster_separation_in_pca_space(*args, **kwargs)

    def plot_pca_2d_scatter(self, *args, **kwargs):
        """Delegate to PlotEnginePCAClusters"""
        return self.clusters.plot_pca_2d_scatter(*args, **kwargs)

    def plot_pca_3d_scatter(self, *args, **kwargs):
        """Delegate to PlotEnginePCAClusters"""
        return self.clusters.plot_pca_3d_scatter(*args, **kwargs)
