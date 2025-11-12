"""
Plot Engine for Score Distribution Visualizations

Provides distribution visualization tools for:
- Score distributions (box plots)
- Company rankings (bar charts)
- Cluster homogeneity (comparison charts)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEngineScoreDistributions:
    """
    Plotting engine for score distribution visualizations

    Supports:
    - Distribution analysis (box plots)
    - Rankings (bar charts)
    - Homogeneity analysis (comparison charts)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Distribution Visualizations

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

        logger.info(f"✓ PlotEngineScoreDistributions initialized (DPI={dpi})")

    # =========================================================================
    # SCORE DISTRIBUTION PLOTS
    # =========================================================================

    def plot_score_distribution(
        self,
        df: pd.DataFrame,
        score_column: str,
        cluster_column: str,
        output_path: Path,
        title: str = None,
        cluster_names: Dict[int, str] = None
    ) -> Path:
        """
        Create box plot showing score distribution per cluster

        Args:
            df: DataFrame with scores and clusters
            score_column: Name of score column
            cluster_column: Name of cluster column
            output_path: Path to save plot
            title: Optional custom title
            cluster_names: Optional mapping {cluster_id: name}

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating score distribution plot: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Get clusters sorted
        clusters = sorted(df[cluster_column].unique())

        # Prepare data for plotting
        plot_data = []
        labels = []

        for cluster in clusters:
            cluster_data = df[df[cluster_column] == cluster][score_column]
            plot_data.append(cluster_data)

            # Create label
            if cluster_names and cluster in cluster_names:
                label = f"{cluster_names[cluster]}\n(n={len(cluster_data)})"
            else:
                label = f"Cluster {cluster}\n(n={len(cluster_data)})"
            labels.append(label)

        # Create box plot
        bp = ax.boxplot(
            plot_data,
            labels=labels,
            patch_artist=True,
            notch=False,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], self.cluster_colors[:len(clusters)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            score_name = score_column.replace('_', ' ').title()
            ax.set_title(f'{score_name} Distribution by Cluster',
                        fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)

        # Add horizontal lines for quartiles
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
        ax.axhline(y=75, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # RANKING CHARTS
    # =========================================================================

    def plot_score_ranking(
        self,
        df: pd.DataFrame,
        score_column: str,
        cluster_column: str,
        output_path: Path,
        top_n: int = 10,
        bottom_n: int = 10,
        title: str = None
    ) -> Path:
        """
        Create horizontal bar chart showing top and bottom companies

        Args:
            df: DataFrame with scores
            score_column: Score column name
            cluster_column: Cluster column name
            top_n: Number of top companies to show
            bottom_n: Number of bottom companies to show
            output_path: Path to save plot
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating score ranking chart: {output_path.name}")

        # Sort by score
        df_sorted = df.sort_values(score_column, ascending=False)

        # Get top and bottom
        top_companies = df_sorted.head(top_n)
        bottom_companies = df_sorted.tail(bottom_n)

        # Combine
        combined = pd.concat([top_companies, bottom_companies])

        # Calculate figure height dynamically
        n_companies = len(combined)
        fig_height = max(8, n_companies * 0.4)

        fig, ax = plt.subplots(figsize=(10, fig_height))

        # Create colors (green for top, red for bottom)
        colors = ['#2ecc71'] * top_n + ['#e74c3c'] * bottom_n

        # Get company names
        if 'company_name' in combined.columns:
            labels = combined['company_name'].tolist()
        elif 'conm' in combined.columns:
            labels = combined['conm'].tolist()
        else:
            labels = [f"Company {gvkey}" for gvkey in combined['gvkey'].tolist()]

        # Shorten long names
        labels = [name[:30] + '...' if len(name) > 30 else name for name in labels]

        # Reverse for plotting (matplotlib plots bottom-up)
        labels = labels[::-1]
        scores = combined[score_column].tolist()[::-1]
        colors = colors[::-1]

        # Create horizontal bar chart
        bars = ax.barh(range(len(labels)), scores, color=colors, alpha=0.7, edgecolor='black')

        # Add score values at end of bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9, fontweight='bold')

        # Style
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 105)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            score_name = score_column.replace('_', ' ').title()
            ax.set_title(f'Top {top_n} and Bottom {bottom_n} Companies by {score_name}',
                        fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.7, label=f'Top {top_n}'),
            Patch(facecolor='#e74c3c', alpha=0.7, label=f'Bottom {bottom_n}')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # HOMOGENEITY & CORRELATION PLOTS
    # =========================================================================

    def plot_homogeneity_comparison(
        self,
        homogeneity_df: pd.DataFrame,
        output_path: Path,
        metric_column: str = 'coefficient_of_variation',
        title: str = None,
        threshold: float = 30.0
    ) -> Path:
        """
        Create bar chart comparing cluster homogeneity

        Args:
            homogeneity_df: DataFrame from ScoreAnalyzer.analyze_cluster_homogeneity()
            metric_column: Metric to plot (CV, std, etc.)
            output_path: Path to save plot
            title: Optional custom title
            threshold: Threshold line (default: 30% for CV)

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating homogeneity comparison: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort by metric
        homogeneity_df = homogeneity_df.sort_values(metric_column, ascending=True)

        # Create gradient colors (green=low CV, red=high CV)
        values = homogeneity_df[metric_column].values
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
        colors = plt.cm.RdYlGn_r(norm(values))

        # Get cluster labels
        if 'cluster_name' in homogeneity_df.columns:
            labels = homogeneity_df['cluster_name'].tolist()
        else:
            labels = [f"Cluster {c}" for c in homogeneity_df.index]

        # Create bar chart
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black')

        # Add threshold line
        if threshold:
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold ({threshold}%)', alpha=0.7)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

        # Style
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Cluster Homogeneity Comparison (Lower = More Homogeneous)',
                        fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test PlotEngineScoreDistributions
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE SCORE DISTRIBUTIONS TEST")
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

    print("\nTesting PlotEngineScoreDistributions:")
    print("-" * 80)

    try:
        plotter = PlotEngineScoreDistributions()

        # Test 1: Score Distribution
        print("\n1. Testing score distribution:")
        path1 = plotter.plot_score_distribution(
            df, 'proximity_score', 'cluster',
            output_path=output_dir / 'score_distribution.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Score Ranking
        print("\n2. Testing score ranking:")
        path2 = plotter.plot_score_ranking(
            df, 'overall_score', 'cluster',
            output_path=output_dir / 'score_ranking.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Homogeneity Comparison
        print("\n3. Testing homogeneity comparison:")
        path3 = plotter.plot_homogeneity_comparison(
            homogeneity_df,
            output_path=output_dir / 'homogeneity_comparison.png'
        )
        print(f"   Created: {path3.name}")

        print("\n✓ PlotEngineScoreDistributions test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineScoreDistributions test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
