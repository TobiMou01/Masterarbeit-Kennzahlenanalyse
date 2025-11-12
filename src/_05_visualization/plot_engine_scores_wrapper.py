"""
Plot Engine Scores Wrapper
Combines distribution and analysis engines for backward compatibility
"""

from src._05_visualization.plot_engine_score_distributions import PlotEngineScoreDistributions
from src._05_visualization.plot_engine_score_analysis import PlotEngineScoreAnalysis


class PlotEngineScores:
    """
    Unified wrapper for score plotting engines

    Combines:
    - PlotEngineScoreDistributions (distributions, rankings, homogeneity)
    - PlotEngineScoreAnalysis (correlations, radar charts, analysis)

    Provides backward compatibility with legacy plot_engine_scores module
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize unified score plotting engine

        Args:
            style: Seaborn style
            dpi: Resolution for saved plots
        """
        self.distributions = PlotEngineScoreDistributions(style=style, dpi=dpi)
        self.analysis = PlotEngineScoreAnalysis(style=style, dpi=dpi)
        self.dpi = dpi

    # =========================================================================
    # DISTRIBUTION METHODS (from PlotEngineScoreDistributions)
    # =========================================================================

    def plot_score_distribution(self, *args, **kwargs):
        """Delegate to PlotEngineScoreDistributions"""
        return self.distributions.plot_score_distribution(*args, **kwargs)

    def plot_score_ranking(self, *args, **kwargs):
        """Delegate to PlotEngineScoreDistributions"""
        return self.distributions.plot_score_ranking(*args, **kwargs)

    def plot_homogeneity_comparison(self, *args, **kwargs):
        """Delegate to PlotEngineScoreDistributions"""
        return self.distributions.plot_homogeneity_comparison(*args, **kwargs)

    # =========================================================================
    # ANALYSIS METHODS (from PlotEngineScoreAnalysis)
    # =========================================================================

    def plot_radar_chart(self, *args, **kwargs):
        """Delegate to PlotEngineScoreAnalysis"""
        return self.analysis.plot_radar_chart(*args, **kwargs)

    def plot_multiple_radar_charts(self, *args, **kwargs):
        """Delegate to PlotEngineScoreAnalysis"""
        return self.analysis.plot_multiple_radar_charts(*args, **kwargs)

    def plot_score_correlation_matrix(self, *args, **kwargs):
        """Delegate to PlotEngineScoreAnalysis"""
        return self.analysis.plot_score_correlation_matrix(*args, **kwargs)

    def create_all_score_plots(self, *args, **kwargs):
        """Delegate to PlotEngineScoreAnalysis"""
        return self.analysis.create_all_score_plots(*args, **kwargs)
