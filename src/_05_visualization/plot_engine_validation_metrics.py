"""
Plot Engine for Validation Metrics Visualizations

Provides comprehensive visualization tools for:
- ARI heatmaps (algorithm congruence)
- Cramér's V comparisons (external validation strength)
- Chi²-test significance (statistical validation)
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


class PlotEngineValidationMetrics:
    """
    Plotting engine for validation metrics visualizations

    Supports:
    - ARI heatmaps (algorithm congruence)
    - Cramér's V comparisons (external validation strength)
    - Chi²-test significance (statistical validation)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Validation Metrics

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

        logger.info(f"✓ PlotEngineValidationMetrics initialized (DPI={dpi})")

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


if __name__ == "__main__":
    # Test PlotEngineValidationMetrics
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE VALIDATION METRICS TEST")
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

    # 2. Cramér's V
    cramers_df = pd.DataFrame({
        'external_label': ['GICS', 'Size', 'Country'],
        'cramers_v': [0.68, 0.15, 0.14],
        'interpretation': ['Very Strong - High Association', 'Low - Weak Association', 'Low - Weak Association']
    })

    # 3. Chi² Results
    chi2_results = {
        'GICS': {'chi2': 307.5, 'p_value': 0.0001, 'significant': True},
        'Size': {'chi2': 89.2, 'p_value': 0.174, 'significant': False},
        'Country': {'chi2': 12.5, 'p_value': 0.451, 'significant': False}
    }

    output_dir = Path('output/test_validation_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEngineValidationMetrics:")
    print("-" * 80)

    try:
        plotter = PlotEngineValidationMetrics()

        # Test 1: ARI Heatmap
        print("\n1. Testing ARI heatmap:")
        path1 = plotter.plot_ari_heatmap(
            ari_matrix,
            output_path=output_dir / 'ari_heatmap.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Cramér's V
        print("\n2. Testing Cramér's V comparison:")
        path2 = plotter.plot_cramers_v_comparison(
            cramers_df,
            output_path=output_dir / 'cramers_v.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Chi² Significance
        print("\n3. Testing Chi² significance:")
        path3 = plotter.plot_chi_square_significance(
            chi2_results,
            output_path=output_dir / 'chi_square.png'
        )
        print(f"   Created: {path3.name}")

        print("\n✓ PlotEngineValidationMetrics test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineValidationMetrics test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
