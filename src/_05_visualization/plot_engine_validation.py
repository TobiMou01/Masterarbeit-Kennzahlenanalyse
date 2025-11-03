"""
Plot Engine for Validation Visualizations

Provides comprehensive visualization tools for:
- Algorithm comparison (ARI heatmaps, confusion matrices)
- External validation (Cramér's V, contingency tables, Chi²-tests)
- Threshold analysis and distribution patterns
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


class PlotEngineValidation:
    """
    Comprehensive plotting engine for validation visualizations

    Supports:
    - ARI heatmaps (algorithm congruence)
    - Confusion matrices (cluster overlap)
    - Cramér's V comparisons (external validation strength)
    - Contingency tables (cluster-external overlap patterns)
    - Chi²-test significance (statistical validation)
    - Distribution analysis (cluster composition)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Validation

        Args:
            style: Seaborn style
            dpi: Resolution for saved plots
        """
        self.dpi = dpi

        # Set global style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # Set font scale for better readability in heatmaps
        sns.set(font_scale=1.1)

        logger.info(f"✓ PlotEngineValidation initialized (DPI={dpi})")

    # =========================================================================
    # ARI HEATMAP
    # =========================================================================

    def plot_ari_heatmap(
        self,
        ari_matrix: pd.DataFrame,
        output_path: Path,
        title: str = None,
        annotate: bool = True
    ) -> Path:
        """
        Create ARI heatmap showing algorithm congruence

        Args:
            ari_matrix: Symmetric ARI matrix (DataFrame)
            output_path: Path to save plot
            title: Optional custom title
            annotate: Whether to annotate cells with values

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating ARI heatmap: {output_path.name}")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            ari_matrix,
            annot=annotate,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Adjusted Rand Index"},
            ax=ax
        )

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Algorithm Congruence: ARI Matrix',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=12, fontweight='bold')

        # Add interpretation box
        interpretation_text = (
            'ARI Interpretation:\n'
            '  > 0.75: Robust\n'
            '  0.50-0.75: Moderate\n'
            '  < 0.50: Low'
        )
        ax.text(
            1.15, 0.5, interpretation_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    def plot_confusion_matrix(
        self,
        confusion_matrix: pd.DataFrame,
        algo1_name: str,
        algo2_name: str,
        output_path: Path,
        normalize: bool = False,
        title: str = None
    ) -> Path:
        """
        Create confusion matrix heatmap

        Args:
            confusion_matrix: Confusion matrix (DataFrame)
            algo1_name: Name of first algorithm (rows)
            algo2_name: Name of second algorithm (columns)
            output_path: Path to save plot
            normalize: Whether to show percentages instead of counts
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating confusion matrix: {output_path.name}")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize if requested
        if normalize:
            # Normalize by rows (algo1)
            matrix_plot = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100
            fmt = '.1f'
            cbar_label = 'Percentage (%)'
        else:
            matrix_plot = confusion_matrix
            fmt = 'd'
            cbar_label = 'Count'

        # Create heatmap
        sns.heatmap(
            matrix_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": cbar_label},
            ax=ax
        )

        # Highlight diagonal
        for i in range(min(len(confusion_matrix), len(confusion_matrix.columns))):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Confusion Matrix: {algo1_name} vs {algo2_name}',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel(f'{algo2_name} Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{algo1_name} Cluster', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    def plot_multiple_confusion_matrices(
        self,
        confusion_matrices: Dict[Tuple[str, str], pd.DataFrame],
        output_path: Path,
        normalize: bool = False,
        title: str = None
    ) -> Path:
        """
        Create grid of confusion matrices

        Args:
            confusion_matrices: Dict mapping (algo1, algo2) -> matrix
            output_path: Path to save plot
            normalize: Whether to show percentages
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating multiple confusion matrices: {output_path.name}")

        n_matrices = len(confusion_matrices)
        n_cols = 2
        n_rows = (n_matrices + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))

        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_matrices > 1 else [axes]

        # Plot each matrix
        for idx, ((algo1, algo2), matrix) in enumerate(confusion_matrices.items()):
            ax = axes[idx]

            # Normalize if requested
            if normalize:
                matrix_plot = matrix.div(matrix.sum(axis=1), axis=0) * 100
                fmt = '.1f'
            else:
                matrix_plot = matrix
                fmt = 'd'

            # Create heatmap
            sns.heatmap(
                matrix_plot,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                square=True,
                linewidths=0.5,
                cbar=True,
                ax=ax
            )

            # Highlight diagonal
            for i in range(min(len(matrix), len(matrix.columns))):
                ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

            ax.set_title(f'{algo1} vs {algo2}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{algo2}', fontsize=10)
            ax.set_ylabel(f'{algo1}', fontsize=10)

        # Hide unused subplots
        for idx in range(n_matrices, len(axes)):
            axes[idx].axis('off')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        else:
            fig.suptitle('Confusion Matrices: Algorithm Comparisons',
                        fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CRAMÉR'S V COMPARISON
    # =========================================================================

    def plot_cramers_v_comparison(
        self,
        cramers_v_df: pd.DataFrame,
        output_path: Path,
        title: str = None
    ) -> Path:
        """
        Create bar chart comparing Cramér's V across external labels

        Args:
            cramers_v_df: DataFrame with external_label, cramers_v, interpretation
            output_path: Path to save plot
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating Cramér's V comparison: {output_path.name}")

        # Sort by Cramér's V
        cramers_v_df = cramers_v_df.sort_values('cramers_v', ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(cramers_v_df) * 0.5)))

        # Get colors based on Cramér's V strength
        colors = [self._get_cramers_v_color(v) for v in cramers_v_df['cramers_v']]

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(cramers_v_df)),
            cramers_v_df['cramers_v'],
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )

        # Add threshold lines
        thresholds = [0.2, 0.4, 0.6, 0.8]
        threshold_labels = ['Very Low', 'Low', 'Moderate', 'High']

        for thresh, label in zip(thresholds, threshold_labels):
            ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Add value annotations
        for i, (idx, row) in enumerate(cramers_v_df.iterrows()):
            v = row['cramers_v']
            interpretation = row['interpretation']
            ax.text(v + 0.02, i, f"{v:.3f}\n{interpretation}",
                   va='center', fontsize=9)

        # Style
        ax.set_yticks(range(len(cramers_v_df)))
        ax.set_yticklabels(cramers_v_df['external_label'], fontsize=11)
        ax.set_xlabel("Cramér's V", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title("External Validation: Cramér's V Comparison",
                        fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, axis='x')

        # Add interpretation legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.8, label='0.0-0.2: Very Low'),
            Patch(facecolor='#e67e22', alpha=0.8, label='0.2-0.4: Low'),
            Patch(facecolor='#f1c40f', alpha=0.8, label='0.4-0.6: Moderate'),
            Patch(facecolor='#2ecc71', alpha=0.8, label='0.6-0.8: High'),
            Patch(facecolor='#27ae60', alpha=0.8, label='0.8-1.0: Very High')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    def _get_cramers_v_color(self, v: float) -> str:
        """Get color based on Cramér's V strength"""
        if v < 0.2:
            return '#e74c3c'  # Red
        elif v < 0.4:
            return '#e67e22'  # Orange
        elif v < 0.6:
            return '#f1c40f'  # Yellow
        elif v < 0.8:
            return '#2ecc71'  # Light Green
        else:
            return '#27ae60'  # Dark Green

    # =========================================================================
    # CONTINGENCY HEATMAP
    # =========================================================================

    def plot_contingency_heatmap(
        self,
        contingency_table: pd.DataFrame,
        cluster_name: str,
        external_name: str,
        output_path: Path,
        normalize: str = None,
        title: str = None
    ) -> Path:
        """
        Create contingency table heatmap

        Args:
            contingency_table: Contingency table (DataFrame)
            cluster_name: Name for cluster dimension
            external_name: Name for external dimension
            output_path: Path to save plot
            normalize: 'index', 'columns', 'all', or None
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating contingency heatmap: {output_path.name}")

        # Remove 'Total' row/column if present
        if 'Total' in contingency_table.index:
            contingency_table = contingency_table.drop('Total')
        if 'Total' in contingency_table.columns:
            contingency_table = contingency_table.drop('Total', axis=1)

        fig, ax = plt.subplots(figsize=(max(10, len(contingency_table.columns) * 1.2),
                                        max(8, len(contingency_table) * 0.8)))

        # Determine format and colorbar label
        if normalize:
            fmt = '.1f'
            cbar_label = 'Percentage (%)'
        else:
            fmt = 'd'
            cbar_label = 'Count'

        # Create heatmap
        sns.heatmap(
            contingency_table,
            annot=True,
            fmt=fmt,
            cmap='Reds',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": cbar_label},
            ax=ax
        )

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Contingency Table: {cluster_name} vs {external_name}',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel(external_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(cluster_name, fontsize=12, fontweight='bold')

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CHI-SQUARE SIGNIFICANCE
    # =========================================================================

    def plot_chi_square_significance(
        self,
        chi_square_results: Dict[str, Dict],
        output_path: Path,
        alpha: float = 0.05,
        title: str = None
    ) -> Path:
        """
        Create bar chart showing Chi²-test significance

        Args:
            chi_square_results: Dict mapping external_label -> {p_value, chi2, ...}
            output_path: Path to save plot
            alpha: Significance level (default: 0.05)
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating Chi² significance plot: {output_path.name}")

        # Prepare data
        labels = []
        neg_log_p = []
        p_values = []
        significant = []

        for ext_label, results in chi_square_results.items():
            p_val = results['p_value']

            labels.append(ext_label)
            p_values.append(p_val)

            # Handle p=0 edge case
            if p_val == 0:
                neg_log_p.append(10)  # Very large value
            else:
                neg_log_p.append(-np.log10(p_val))

            significant.append(results.get('significant', p_val < alpha))

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'label': labels,
            'neg_log_p': neg_log_p,
            'p_value': p_values,
            'significant': significant
        }).sort_values('neg_log_p', ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.5)))

        # Colors based on significance
        colors = ['#2ecc71' if sig else '#e74c3c' for sig in plot_df['significant']]

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(plot_df)),
            plot_df['neg_log_p'],
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )

        # Add threshold line
        threshold = -np.log10(alpha)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'α = {alpha} (threshold)', alpha=0.7)

        # Add p-value annotations
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            p_val = row['p_value']
            if p_val < 0.001:
                p_text = 'p < 0.001'
            elif p_val < 0.01:
                p_text = f'p = {p_val:.3f}'
            else:
                p_text = f'p = {p_val:.2f}'

            ax.text(row['neg_log_p'] + 0.1, i, p_text, va='center', fontsize=9)

        # Style
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['label'], fontsize=11)
        ax.set_xlabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Chi²-Test Significance',
                        fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, axis='x')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.8, label='Significant'),
            Patch(facecolor='#e74c3c', alpha=0.8, label='Not Significant'),
            ax.axvline(x=threshold, color='red', linestyle='--', label=f'α = {alpha}')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # EXTERNAL DISTRIBUTION
    # =========================================================================

    def plot_external_distribution(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str,
        output_path: Path,
        stacked: bool = True,
        title: str = None
    ) -> Path:
        """
        Create stacked bar chart showing cluster distribution within external labels

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Cluster column name
            external_column: External label column name
            output_path: Path to save plot
            stacked: Whether to create stacked bars (True) or grouped (False)
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating external distribution plot: {output_path.name}")

        # Create contingency table (counts)
        contingency = pd.crosstab(df[external_column], df[cluster_column])

        # Normalize to percentages
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create stacked bar chart
        contingency_pct.plot(
            kind='bar',
            stacked=stacked,
            ax=ax,
            colormap='tab10',
            edgecolor='black',
            linewidth=0.5
        )

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Cluster Distribution within {external_column}',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel(external_column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)

        # Rotate x-labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Legend
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # ARI THRESHOLD ANALYSIS
    # =========================================================================

    def plot_ari_threshold_analysis(
        self,
        ari_values: Dict[str, float],
        output_path: Path,
        thresholds: List[float] = [0.5, 0.75, 0.9],
        title: str = None
    ) -> Path:
        """
        Create bar chart with ARI threshold zones

        Args:
            ari_values: Dict mapping algorithm_pair -> ARI
            output_path: Path to save plot
            thresholds: Threshold values for zones
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating ARI threshold analysis: {output_path.name}")

        # Prepare data
        pairs = list(ari_values.keys())
        values = list(ari_values.values())

        # Sort by ARI
        sorted_indices = np.argsort(values)
        pairs = [pairs[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(pairs) * 0.5)))

        # Add background zones
        zone_colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#27ae60']
        zone_labels = ['Low\n(< 0.5)', 'Moderate\n(0.5-0.75)', 'Good\n(0.75-0.9)', 'Excellent\n(≥ 0.9)']
        zone_boundaries = [0] + thresholds + [1.0]

        for i in range(len(zone_boundaries) - 1):
            ax.axvspan(
                zone_boundaries[i], zone_boundaries[i + 1],
                alpha=0.2,
                color=zone_colors[i],
                label=zone_labels[i]
            )

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(pairs)),
            values,
            color='steelblue',
            alpha=0.8,
            edgecolor='black'
        )

        # Add value annotations
        for i, val in enumerate(values):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

        # Add threshold lines
        for thresh in thresholds:
            ax.axvline(x=thresh, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Style
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=10)
        ax.set_xlabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('ARI Threshold Analysis',
                        fontsize=14, fontweight='bold', pad=20)

        # Legend for zones
        ax.legend(loc='lower right', fontsize=9, title='Agreement Level')

        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CONVENIENCE METHOD
    # =========================================================================

    def create_all_validation_plots(
        self,
        algo_comparison_results: Dict,
        external_validation_results: Dict,
        output_dir: Path,
        df: pd.DataFrame = None
    ) -> Dict[str, Path]:
        """
        Create all validation plots at once

        Args:
            algo_comparison_results: Results from AlgorithmComparison
            external_validation_results: Results from ExternalValidation
            output_dir: Directory to save plots
            df: Optional DataFrame for distribution plots

        Returns:
            Dictionary mapping plot names to paths
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 Creating All Validation Plots")
        logger.info("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # Algorithm Comparison Plots
        if algo_comparison_results:
            # 1. ARI Heatmap
            if 'ari_matrix' in algo_comparison_results:
                path = self.plot_ari_heatmap(
                    algo_comparison_results['ari_matrix'],
                    output_dir / 'ari_heatmap.png'
                )
                plots['ari_heatmap'] = path

            # 2. ARI Threshold Analysis
            if 'summary' in algo_comparison_results:
                pairwise = algo_comparison_results['summary'].get('pairwise_comparisons')
                if pairwise is not None and not pairwise.empty:
                    ari_values = {
                        f"{row['algo1']} vs {row['algo2']}": row['ari']
                        for _, row in pairwise.iterrows()
                    }
                    path = self.plot_ari_threshold_analysis(
                        ari_values,
                        output_dir / 'ari_threshold_analysis.png'
                    )
                    plots['ari_threshold_analysis'] = path

            # 3. Confusion Matrices
            if 'confusion_matrices' in algo_comparison_results:
                conf_matrices = algo_comparison_results['confusion_matrices']
                if conf_matrices:
                    path = self.plot_multiple_confusion_matrices(
                        conf_matrices,
                        output_dir / 'confusion_matrices.png'
                    )
                    plots['confusion_matrices'] = path

        # External Validation Plots
        if external_validation_results:
            # 4. Cramér's V Comparison
            if 'cramers_v' in external_validation_results:
                cramers_data = []
                for ext_label, data in external_validation_results['cramers_v'].items():
                    cramers_data.append({
                        'external_label': ext_label,
                        'cramers_v': data['value'],
                        'interpretation': data['interpretation']
                    })

                if cramers_data:
                    cramers_df = pd.DataFrame(cramers_data)
                    path = self.plot_cramers_v_comparison(
                        cramers_df,
                        output_dir / 'cramers_v_comparison.png'
                    )
                    plots['cramers_v_comparison'] = path

            # 5. Chi²-Test Significance
            if 'chi_square' in external_validation_results:
                path = self.plot_chi_square_significance(
                    external_validation_results['chi_square'],
                    output_dir / 'chi_square_significance.png'
                )
                plots['chi_square_significance'] = path

            # 6. Contingency Heatmaps
            if 'contingency_tables' in external_validation_results:
                for ext_label, cont_table in external_validation_results['contingency_tables'].items():
                    if not cont_table.empty:
                        path = self.plot_contingency_heatmap(
                            cont_table,
                            'Cluster',
                            ext_label,
                            output_dir / f'contingency_{ext_label}.png'
                        )
                        plots[f'contingency_{ext_label}'] = path

        logger.info("=" * 80)
        logger.info(f"✓ Created {len(plots)} validation plots")
        logger.info("=" * 80 + "\n")

        return plots


if __name__ == "__main__":
    # Test PlotEngineValidation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE VALIDATION TEST")
    print("=" * 80)

    # Mock data
    np.random.seed(42)

    # 1. ARI Matrix
    ari_matrix = pd.DataFrame(
        [[1.00, 0.82, 0.45],
         [0.82, 1.00, 0.48],
         [0.45, 0.48, 1.00]],
        index=['K-Means', 'Hierarchical', 'DBSCAN'],
        columns=['K-Means', 'Hierarchical', 'DBSCAN']
    )

    # 2. Confusion Matrix
    confusion = pd.DataFrame(
        [[85, 12, 3],
         [10, 78, 12],
         [5, 10, 85]],
        index=['C0', 'C1', 'C2'],
        columns=['C0', 'C1', 'C2']
    )

    # 3. Cramér's V
    cramers_df = pd.DataFrame({
        'external_label': ['GICS', 'Size', 'Country'],
        'cramers_v': [0.68, 0.15, 0.14],
        'interpretation': ['Very Strong - High Association', 'Low - Weak Association', 'Low - Weak Association']
    })

    # 4. Contingency Table
    contingency = pd.DataFrame(
        [[45, 12, 8, 5],
         [18, 55, 15, 7],
         [7, 13, 62, 10]],
        index=[0, 1, 2],
        columns=['GICS_A', 'GICS_B', 'GICS_C', 'GICS_D']
    )

    # 5. Chi² Results
    chi2_results = {
        'GICS': {'chi2': 307.5, 'p_value': 0.0001, 'significant': True},
        'Size': {'chi2': 89.2, 'p_value': 0.174, 'significant': False},
        'Country': {'chi2': 12.5, 'p_value': 0.451, 'significant': False}
    }

    # 6. ARI Values
    ari_values = {
        'KMeans vs Hierarchical': 0.82,
        'KMeans vs DBSCAN': 0.45,
        'Hierarchical vs DBSCAN': 0.48
    }

    output_dir = Path('output/test_validation_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEngineValidation:")
    print("-" * 80)

    try:
        plotter = PlotEngineValidation()

        # Test 1: ARI Heatmap
        print("\n1. Testing ARI heatmap:")
        path1 = plotter.plot_ari_heatmap(
            ari_matrix,
            output_path=output_dir / 'ari_heatmap.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Confusion Matrix
        print("\n2. Testing confusion matrix:")
        path2 = plotter.plot_confusion_matrix(
            confusion, 'K-Means', 'Hierarchical',
            output_path=output_dir / 'confusion_matrix.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Multiple Confusion Matrices
        print("\n3. Testing multiple confusion matrices:")
        conf_matrices = {
            ('K-Means', 'Hierarchical'): confusion,
            ('K-Means', 'DBSCAN'): confusion,
            ('Hierarchical', 'DBSCAN'): confusion
        }
        path3 = plotter.plot_multiple_confusion_matrices(
            conf_matrices,
            output_path=output_dir / 'confusion_matrices_grid.png'
        )
        print(f"   Created: {path3.name}")

        # Test 4: Cramér's V
        print("\n4. Testing Cramér's V comparison:")
        path4 = plotter.plot_cramers_v_comparison(
            cramers_df,
            output_path=output_dir / 'cramers_v.png'
        )
        print(f"   Created: {path4.name}")

        # Test 5: Contingency Heatmap
        print("\n5. Testing contingency heatmap:")
        path5 = plotter.plot_contingency_heatmap(
            contingency, 'Cluster', 'GICS',
            output_path=output_dir / 'contingency.png'
        )
        print(f"   Created: {path5.name}")

        # Test 6: Chi² Significance
        print("\n6. Testing Chi² significance:")
        path6 = plotter.plot_chi_square_significance(
            chi2_results,
            output_path=output_dir / 'chi_square.png'
        )
        print(f"   Created: {path6.name}")

        # Test 7: ARI Threshold Analysis
        print("\n7. Testing ARI threshold analysis:")
        path7 = plotter.plot_ari_threshold_analysis(
            ari_values,
            output_path=output_dir / 'ari_thresholds.png'
        )
        print(f"   Created: {path7.name}")

        print("\n✓ PlotEngineValidation test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineValidation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
