"""
Plot Engine for Advanced Score Analysis Visualizations

Provides advanced analysis visualization tools for:
- Multi-dimensional profiles (radar charts)
- Score correlations (heatmaps)
- Comprehensive plot orchestration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEngineScoreAnalysis:
    """
    Plotting engine for advanced score analysis visualizations

    Supports:
    - Multi-dimensional profiles (radar charts)
    - Correlation analysis (heatmaps)
    - Comprehensive plot orchestration
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Advanced Analysis

        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
            dpi: Resolution for saved plots
        """
        self.dpi = dpi

        # Set global style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # Color palettes
        self.cluster_colors = sns.color_palette("husl", n_colors=10)
        self.colorblind_palette = sns.color_palette("colorblind", n_colors=10)

        logger.info(f"✓ PlotEngineScoreAnalysis initialized (DPI={dpi})")

    # =========================================================================
    # RADAR CHARTS
    # =========================================================================

    def plot_radar_chart(
        self,
        scores: Dict[str, float],
        company_name: str,
        output_path: Path,
        benchmark_scores: Dict[str, float] = None,
        title: str = None
    ) -> Path:
        """
        Create radar chart for dimensional scores

        Args:
            scores: Dictionary {dimension: score}
            company_name: Company name for title
            output_path: Path to save plot
            benchmark_scores: Optional benchmark scores (e.g., cluster average)
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating radar chart: {output_path.name}")

        # Prepare data
        categories = list(scores.keys())
        values = list(scores.values())

        # Number of variables
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot company scores
        ax.plot(angles, values, 'o-', linewidth=2, label=company_name, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')

        # Plot benchmark if provided
        if benchmark_scores:
            benchmark_values = [benchmark_scores.get(cat, 0) for cat in categories]
            benchmark_values += benchmark_values[:1]
            ax.plot(angles, benchmark_values, 'o--', linewidth=2,
                   label='Benchmark', color='gray', alpha=0.7)
            ax.fill(angles, benchmark_values, alpha=0.1, color='gray')

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, fontweight='bold')

        # Set y-axis limits and labels
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], size=9, alpha=0.7)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Title
        if title:
            ax.set_title(title, size=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Dimensional Profile: {company_name}',
                        size=14, fontweight='bold', pad=20)

        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    def plot_multiple_radar_charts(
        self,
        df: pd.DataFrame,
        dimensional_score_columns: List[str],
        company_ids: List[int],
        output_dir: Path,
        cluster_column: str = 'cluster'
    ) -> List[Path]:
        """
        Create multiple radar charts for selected companies

        Args:
            df: DataFrame with dimensional scores
            dimensional_score_columns: List of score column names
            company_ids: List of company IDs (gvkey)
            output_dir: Directory to save plots
            cluster_column: Cluster column name

        Returns:
            List of paths to saved plots
        """
        logger.info(f"  Creating {len(company_ids)} radar charts")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []

        for company_id in company_ids:
            company_row = df[df['gvkey'] == company_id]

            if len(company_row) == 0:
                logger.warning(f"    ⚠️  Company {company_id} not found")
                continue

            company_row = company_row.iloc[0]

            # Get company name
            company_name = company_row.get('company_name', f'Company {company_id}')

            # Shorten if too long
            if len(company_name) > 30:
                company_name = company_name[:27] + '...'

            # Get scores
            scores = {}
            for col in dimensional_score_columns:
                # Clean up column name for display
                dimension = col.replace('_score', '').replace('_', ' ').title()
                scores[dimension] = company_row[col]

            # Get cluster average as benchmark
            cluster = company_row[cluster_column]
            cluster_data = df[df[cluster_column] == cluster]

            benchmark_scores = {}
            for col in dimensional_score_columns:
                dimension = col.replace('_score', '').replace('_', ' ').title()
                benchmark_scores[dimension] = cluster_data[col].mean()

            # Create radar chart
            output_path = output_dir / f"radar_{company_id}_{company_name.replace(' ', '_')}.png"
            path = self.plot_radar_chart(
                scores,
                company_name,
                output_path,
                benchmark_scores=benchmark_scores
            )
            paths.append(path)

        logger.info(f"    ✓ Created {len(paths)} radar charts")
        return paths

    # =========================================================================
    # CORRELATION PLOTS
    # =========================================================================

    def plot_score_correlation_matrix(
        self,
        df: pd.DataFrame,
        score_columns: List[str],
        output_path: Path,
        title: str = None
    ) -> Path:
        """
        Create correlation heatmap between different scores

        Args:
            df: DataFrame with score columns
            score_columns: List of score column names
            output_path: Path to save plot
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating score correlation matrix: {output_path.name}")

        # Calculate correlation matrix
        corr_matrix = df[score_columns].corr()

        # Clean up labels
        labels = [col.replace('_score', '').replace('_', ' ').title() for col in score_columns]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Score Correlation Matrix',
                        fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CONVENIENCE METHOD
    # =========================================================================

    def create_all_score_plots(
        self,
        df: pd.DataFrame,
        score_columns: List[str],
        dimensional_scores_df: pd.DataFrame,
        evolution_df: pd.DataFrame,
        output_dir: Path,
        cluster_column: str = 'cluster',
        homogeneity_df: pd.DataFrame = None
    ) -> Dict[str, Path]:
        """
        Create all score plots at once

        This orchestrates the creation of all visualization types by using
        the separate specialized plot engines for distributions, evolution,
        and analysis visualizations.

        Args:
            df: DataFrame with scores and clusters
            score_columns: List of score columns to plot
            dimensional_scores_df: DataFrame with dimensional scores
            evolution_df: DataFrame with evolution data
            output_dir: Directory to save plots
            cluster_column: Cluster column name
            homogeneity_df: Optional homogeneity DataFrame

        Returns:
            Dictionary mapping plot names to paths
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 Creating All Score Plots")
        logger.info("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Import the specialized plot engines
        from .plot_engine_score_distributions import PlotEngineScoreDistributions
        from .plot_engine_score_evolution import PlotEngineScoreEvolution

        # Initialize plot engines
        distributions_plotter = PlotEngineScoreDistributions(dpi=self.dpi)
        evolution_plotter = PlotEngineScoreEvolution(dpi=self.dpi)

        # 1. Score distributions (for each score column)
        for score_col in score_columns:
            if score_col in df.columns:
                plot_name = f"distribution_{score_col}"
                path = distributions_plotter.plot_score_distribution(
                    df, score_col, cluster_column,
                    output_dir / f"{plot_name}.png"
                )
                plots[plot_name] = path

        # 2. Score evolution scatter
        if evolution_df is not None and not evolution_df.empty:
            path = evolution_plotter.plot_score_evolution_scatter(
                evolution_df,
                output_path=output_dir / "score_evolution.png"
            )
            plots['score_evolution'] = path

        # 3. Score ranking (for main score)
        if score_columns:
            main_score = score_columns[0]
            path = distributions_plotter.plot_score_ranking(
                df, main_score, cluster_column,
                output_path=output_dir / "score_ranking.png"
            )
            plots['score_ranking'] = path

        # 4. Homogeneity comparison
        if homogeneity_df is not None and not homogeneity_df.empty:
            path = distributions_plotter.plot_homogeneity_comparison(
                homogeneity_df,
                output_path=output_dir / "homogeneity_comparison.png"
            )
            plots['homogeneity_comparison'] = path

        # 5. Dimensional heatmap
        if dimensional_scores_df is not None and not dimensional_scores_df.empty:
            dim_cols = [col for col in dimensional_scores_df.columns if col.endswith('_score')]
            if dim_cols:
                path = evolution_plotter.plot_dimensional_heatmap(
                    dimensional_scores_df, dim_cols, cluster_column,
                    output_path=output_dir / "dimensional_heatmap.png"
                )
                plots['dimensional_heatmap'] = path

        # 6. Score correlation matrix
        if len(score_columns) > 1:
            available_scores = [col for col in score_columns if col in df.columns]
            if len(available_scores) > 1:
                path = self.plot_score_correlation_matrix(
                    df, available_scores,
                    output_path=output_dir / "score_correlation.png"
                )
                plots['score_correlation'] = path

        logger.info("=" * 80)
        logger.info(f"✓ Created {len(plots)} score plots")
        logger.info("=" * 80 + "\n")

        return plots


if __name__ == "__main__":
    # Test PlotEngineScoreAnalysis
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE SCORE ANALYSIS TEST")
    print("=" * 80)

    # Mock data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'gvkey': range(n),
        'company_name': [f'Company_{i}' for i in range(n)],
        'cluster': np.random.randint(0, 4, n),
        'proximity_score': np.random.uniform(20, 90, n),
        'overall_score': np.random.uniform(30, 85, n),
        'profitability_score': np.random.uniform(25, 95, n),
        'leverage_score': np.random.uniform(30, 80, n),
        'efficiency_score': np.random.uniform(35, 90, n),
        'liquidity_score': np.random.uniform(40, 85, n)
    })

    evolution_df = pd.DataFrame({
        'gvkey': range(n),
        'company_name': [f'Company_{i}' for i in range(n)],
        'static_score': np.random.uniform(30, 80, n),
        'dynamic_score': np.random.uniform(35, 85, n),
        'pattern': np.random.choice(['Consistent Excellence', 'Improving Trend', 'Eroding Position'], n)
    })

    # Create homogeneity data
    homogeneity_df = pd.DataFrame({
        'cluster': [0, 1, 2, 3],
        'mean': [75, 60, 45, 50],
        'std': [10, 15, 20, 12],
        'coefficient_of_variation': [13.3, 25.0, 44.4, 24.0],
        'cluster_name': ['High Performers', 'Upper-Mid', 'Lower-Mid', 'Low Performers']
    }).set_index('cluster')

    output_dir = Path('output/test_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEngineScoreAnalysis:")
    print("-" * 80)

    try:
        plotter = PlotEngineScoreAnalysis()

        # Test 1: Radar Chart
        print("\n1. Testing radar chart:")
        scores = {
            'Profitability': 85,
            'Leverage': 65,
            'Efficiency': 72,
            'Liquidity': 80
        }
        benchmark = {
            'Profitability': 70,
            'Leverage': 60,
            'Efficiency': 65,
            'Liquidity': 75
        }
        path1 = plotter.plot_radar_chart(
            scores, 'Company_1',
            output_path=output_dir / 'radar_company1.png',
            benchmark_scores=benchmark
        )
        print(f"   Created: {path1.name}")

        # Test 2: Score Correlation
        print("\n2. Testing score correlation matrix:")
        score_cols = ['proximity_score', 'overall_score', 'profitability_score', 'leverage_score']
        path2 = plotter.plot_score_correlation_matrix(
            df, score_cols,
            output_path=output_dir / 'score_correlation.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Create All Plots
        print("\n3. Testing create_all_score_plots:")
        all_plots = plotter.create_all_score_plots(
            df=df,
            score_columns=['proximity_score', 'overall_score'],
            dimensional_scores_df=df,
            evolution_df=evolution_df,
            output_dir=output_dir / 'all_plots',
            homogeneity_df=homogeneity_df
        )
        print(f"   Created {len(all_plots)} plots:")
        for name, path in all_plots.items():
            print(f"     - {name}: {path.name}")

        print("\n✓ PlotEngineScoreAnalysis test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineScoreAnalysis test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
