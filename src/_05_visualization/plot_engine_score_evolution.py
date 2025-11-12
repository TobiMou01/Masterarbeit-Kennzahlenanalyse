"""
Plot Engine for Score Evolution Visualizations

Provides evolution tracking visualization tools for:
- Score evolution patterns (scatter plots)
- Dimensional profiles (heatmaps)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEngineScoreEvolution:
    """
    Plotting engine for score evolution visualizations

    Supports:
    - Evolution tracking (scatter plots)
    - Multi-dimensional profiles (heatmaps)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Evolution Visualizations

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

        # Evolution pattern colors
        self.evolution_colors = {
            'Consistent Excellence': '#2ecc71',  # Green
            'Eroding Position': '#e74c3c',       # Red
            'Improving Trend': '#3498db',        # Blue
            'Challenged': '#95a5a6',             # Gray
            'Mixed': '#f39c12',                  # Orange
            'Stable Mid-Range': '#9b59b6',       # Purple
            'Volatile': '#e67e22'                # Dark Orange
        }

        # Pattern markers (for scatter plots)
        self.pattern_markers = {
            'Consistent Excellence': 'o',
            'Improving Trend': '^',
            'Eroding Position': 'v',
            'Challenged': 's',
            'Mixed': 'D',
            'Stable Mid-Range': 'p',
            'Volatile': '*'
        }

        logger.info(f"✓ PlotEngineScoreEvolution initialized (DPI={dpi})")

    # =========================================================================
    # SCORE EVOLUTION PLOTS
    # =========================================================================

    def plot_score_evolution_scatter(
        self,
        evolution_df: pd.DataFrame,
        output_path: Path,
        x_column: str = 'static_score',
        y_column: str = 'dynamic_score',
        pattern_column: str = 'pattern',
        title: str = None
    ) -> Path:
        """
        Create scatter plot showing score evolution patterns

        Args:
            evolution_df: DataFrame with evolution data
            output_path: Path to save plot
            x_column: Column for X-axis (static score)
            y_column: Column for Y-axis (dynamic score)
            pattern_column: Column with evolution patterns
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating score evolution scatter: {output_path.name}")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Get unique patterns
        patterns = evolution_df[pattern_column].unique()

        # Plot each pattern separately
        for pattern in patterns:
            pattern_data = evolution_df[evolution_df[pattern_column] == pattern]

            color = self.evolution_colors.get(pattern, '#95a5a6')
            marker = self.pattern_markers.get(pattern, 'o')

            ax.scatter(
                pattern_data[x_column],
                pattern_data[y_column],
                c=color,
                marker=marker,
                s=100,
                alpha=0.7,
                label=pattern,
                edgecolors='black',
                linewidth=0.5
            )

        # Add diagonal reference line (x=y)
        lim_min = 0
        lim_max = 100
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
               'k--', alpha=0.5, linewidth=2, label='No Change (x=y)')

        # Add quadrant labels
        ax.text(25, 75, 'Improving\nTrend', ha='center', va='center',
               fontsize=10, alpha=0.5, style='italic')
        ax.text(75, 75, 'Consistent\nExcellence', ha='center', va='center',
               fontsize=10, alpha=0.5, style='italic')
        ax.text(25, 25, 'Challenged', ha='center', va='center',
               fontsize=10, alpha=0.5, style='italic')
        ax.text(75, 25, 'Eroding\nPosition', ha='center', va='center',
               fontsize=10, alpha=0.5, style='italic')

        # Style
        ax.set_xlabel('Static Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dynamic Score', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Score Evolution: Static vs Dynamic',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)

        # Equal aspect ratio for square plot
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # DIMENSIONAL HEATMAP
    # =========================================================================

    def plot_dimensional_heatmap(
        self,
        df: pd.DataFrame,
        dimensional_score_columns: List[str],
        cluster_column: str,
        output_path: Path,
        top_n_per_cluster: int = 5,
        title: str = None
    ) -> Path:
        """
        Create heatmap showing dimensional scores for top companies per cluster

        Args:
            df: DataFrame with dimensional scores
            dimensional_score_columns: List of dimensional score columns
            cluster_column: Cluster column name
            output_path: Path to save plot
            top_n_per_cluster: Number of companies to show per cluster
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating dimensional heatmap: {output_path.name}")

        # Select top N companies per cluster
        selected_companies = []
        clusters = sorted(df[cluster_column].unique())

        for cluster in clusters:
            cluster_data = df[df[cluster_column] == cluster]

            # Sort by overall score or proximity score (if available)
            if 'overall_score' in cluster_data.columns:
                sort_col = 'overall_score'
            elif 'proximity_score' in cluster_data.columns:
                sort_col = 'proximity_score'
            else:
                sort_col = dimensional_score_columns[0]

            top_companies = cluster_data.nlargest(top_n_per_cluster, sort_col)
            selected_companies.append(top_companies)

        # Combine
        heatmap_df = pd.concat(selected_companies)

        # Extract scores
        scores_matrix = heatmap_df[dimensional_score_columns].values

        # Create labels
        if 'company_name' in heatmap_df.columns:
            y_labels = heatmap_df['company_name'].tolist()
        elif 'conm' in heatmap_df.columns:
            y_labels = heatmap_df['conm'].tolist()
        else:
            y_labels = [f"Company {gvkey}" for gvkey in heatmap_df['gvkey'].tolist()]

        # Shorten names
        y_labels = [name[:25] + '...' if len(name) > 25 else name for name in y_labels]

        # Clean up column names
        x_labels = [col.replace('_score', '').replace('_', ' ').title()
                   for col in dimensional_score_columns]

        # Calculate figure height dynamically
        fig_height = max(8, len(y_labels) * 0.4)

        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Create heatmap
        sns.heatmap(
            scores_matrix,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            center=50,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Score"},
            xticklabels=x_labels,
            yticklabels=y_labels,
            ax=ax
        )

        # Add cluster separators
        clusters_list = heatmap_df[cluster_column].tolist()
        for i in range(1, len(clusters_list)):
            if clusters_list[i] != clusters_list[i-1]:
                ax.axhline(y=i, color='black', linewidth=2)

        # Style
        ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
        ax.set_ylabel('Company', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Dimensional Scores: Top {top_n_per_cluster} per Cluster',
                        fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test PlotEngineScoreEvolution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE SCORE EVOLUTION TEST")
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

    output_dir = Path('output/test_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEngineScoreEvolution:")
    print("-" * 80)

    try:
        plotter = PlotEngineScoreEvolution()

        # Test 1: Score Evolution
        print("\n1. Testing score evolution scatter:")
        path1 = plotter.plot_score_evolution_scatter(
            evolution_df,
            output_path=output_dir / 'score_evolution.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Dimensional Heatmap
        print("\n2. Testing dimensional heatmap:")
        dim_cols = ['profitability_score', 'leverage_score', 'efficiency_score', 'liquidity_score']
        path2 = plotter.plot_dimensional_heatmap(
            df, dim_cols, 'cluster',
            output_path=output_dir / 'dimensional_heatmap.png',
            top_n_per_cluster=3
        )
        print(f"   Created: {path2.name}")

        print("\n✓ PlotEngineScoreEvolution test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineScoreEvolution test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
