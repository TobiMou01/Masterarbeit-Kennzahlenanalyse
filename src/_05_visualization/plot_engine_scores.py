"""
Plot Engine for Score Visualizations

Provides comprehensive visualization tools for:
- Score distributions (box plots)
- Score evolution patterns (scatter plots)
- Dimensional profiles (radar charts)
- Company rankings (bar charts)
- Cluster homogeneity (comparison charts)
- Score correlations (heatmaps)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEngineScores:
    """
    Comprehensive plotting engine for score visualizations

    Supports:
    - Distribution analysis (box plots)
    - Evolution tracking (scatter plots)
    - Multi-dimensional profiles (radar charts)
    - Rankings (bar charts)
    - Homogeneity analysis (comparison charts)
    - Correlation analysis (heatmaps)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine

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

        logger.info(f"✓ PlotEngineScores initialized (DPI={dpi})")

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

        # 1. Score distributions (for each score column)
        for score_col in score_columns:
            if score_col in df.columns:
                plot_name = f"distribution_{score_col}"
                path = self.plot_score_distribution(
                    df, score_col, cluster_column,
                    output_dir / f"{plot_name}.png"
                )
                plots[plot_name] = path

        # 2. Score evolution scatter
        if evolution_df is not None and not evolution_df.empty:
            path = self.plot_score_evolution_scatter(
                evolution_df,
                output_path=output_dir / "score_evolution.png"
            )
            plots['score_evolution'] = path

        # 3. Score ranking (for main score)
        if score_columns:
            main_score = score_columns[0]
            path = self.plot_score_ranking(
                df, main_score, cluster_column,
                output_path=output_dir / "score_ranking.png"
            )
            plots['score_ranking'] = path

        # 4. Homogeneity comparison
        if homogeneity_df is not None and not homogeneity_df.empty:
            path = self.plot_homogeneity_comparison(
                homogeneity_df,
                output_path=output_dir / "homogeneity_comparison.png"
            )
            plots['homogeneity_comparison'] = path

        # 5. Dimensional heatmap
        if dimensional_scores_df is not None and not dimensional_scores_df.empty:
            dim_cols = [col for col in dimensional_scores_df.columns if col.endswith('_score')]
            if dim_cols:
                path = self.plot_dimensional_heatmap(
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
    # Test PlotEngineScores
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE SCORES TEST")
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

    print("\nTesting PlotEngineScores:")
    print("-" * 80)

    try:
        plotter = PlotEngineScores()

        # Test 1: Score Distribution
        print("\n1. Testing score distribution:")
        path1 = plotter.plot_score_distribution(
            df, 'proximity_score', 'cluster',
            output_path=output_dir / 'score_distribution.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Score Evolution
        print("\n2. Testing score evolution scatter:")
        path2 = plotter.plot_score_evolution_scatter(
            evolution_df,
            output_path=output_dir / 'score_evolution.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Radar Chart
        print("\n3. Testing radar chart:")
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
        path3 = plotter.plot_radar_chart(
            scores, 'Company_1',
            output_path=output_dir / 'radar_company1.png',
            benchmark_scores=benchmark
        )
        print(f"   Created: {path3.name}")

        # Test 4: Score Ranking
        print("\n4. Testing score ranking:")
        path4 = plotter.plot_score_ranking(
            df, 'overall_score', 'cluster',
            output_path=output_dir / 'score_ranking.png'
        )
        print(f"   Created: {path4.name}")

        # Test 5: Homogeneity Comparison
        print("\n5. Testing homogeneity comparison:")
        path5 = plotter.plot_homogeneity_comparison(
            homogeneity_df,
            output_path=output_dir / 'homogeneity_comparison.png'
        )
        print(f"   Created: {path5.name}")

        # Test 6: Dimensional Heatmap
        print("\n6. Testing dimensional heatmap:")
        dim_cols = ['profitability_score', 'leverage_score', 'efficiency_score', 'liquidity_score']
        path6 = plotter.plot_dimensional_heatmap(
            df, dim_cols, 'cluster',
            output_path=output_dir / 'dimensional_heatmap.png',
            top_n_per_cluster=3
        )
        print(f"   Created: {path6.name}")

        # Test 7: Score Correlation
        print("\n7. Testing score correlation matrix:")
        score_cols = ['proximity_score', 'overall_score', 'profitability_score', 'leverage_score']
        path7 = plotter.plot_score_correlation_matrix(
            df, score_cols,
            output_path=output_dir / 'score_correlation.png'
        )
        print(f"   Created: {path7.name}")

        # Test 8: Create All Plots
        print("\n8. Testing create_all_score_plots:")
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

        print("\n✓ PlotEngineScores test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineScores test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
