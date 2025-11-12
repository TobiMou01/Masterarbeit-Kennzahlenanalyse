"""
Plot Engine for PCA Variance Analysis

Provides visualization tools for:
- Scree plots (variance explained per component)
- Cumulative variance bar charts
- Variance threshold analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEnginePCAVariance:
    """
    Plotting engine for PCA variance analysis

    Supports:
    - Scree plots showing individual and cumulative variance
    - Cumulative variance bar charts
    - Threshold-based component selection visualization
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for PCA Variance Analysis

        Args:
            style: Seaborn style
            dpi: Resolution for saved plots
        """
        self.dpi = dpi

        # Set global style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        logger.info(f"✓ PlotEnginePCAVariance initialized (DPI={dpi})")

    # =========================================================================
    # SCREE PLOT
    # =========================================================================

    def plot_scree_plot(
        self,
        explained_variance_ratio: np.ndarray,
        output_path: Path,
        cumulative: bool = True,
        threshold: float = 0.85,
        title: str = None
    ) -> Path:
        """
        Create scree plot showing variance explained per component

        Args:
            explained_variance_ratio: Variance explained by each component
            output_path: Path to save plot
            cumulative: Whether to show cumulative variance
            threshold: Variance threshold line (e.g., 0.85 for 85%)
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating scree plot: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 8))

        n_components = len(explained_variance_ratio)
        components = np.arange(1, n_components + 1)

        # Individual variance
        ax.plot(components, explained_variance_ratio * 100, 'bo-', linewidth=2,
               markersize=8, label='Individual Variance', alpha=0.8)

        # Cumulative variance
        if cumulative:
            cumulative_variance = np.cumsum(explained_variance_ratio)
            ax.plot(components, cumulative_variance * 100, 'ro-', linewidth=2,
                   markersize=8, label='Cumulative Variance', alpha=0.8)

            # Find component that reaches threshold
            threshold_idx = np.argmax(cumulative_variance >= threshold)
            if cumulative_variance[threshold_idx] >= threshold:
                ax.axhline(y=threshold * 100, color='green', linestyle='--',
                          linewidth=2, label=f'Threshold ({threshold:.0%})', alpha=0.7)
                ax.axvline(x=threshold_idx + 1, color='gray', linestyle=':',
                          linewidth=1.5, alpha=0.5)

                # Annotation
                ax.plot(threshold_idx + 1, cumulative_variance[threshold_idx] * 100,
                       'g*', markersize=20, label=f'Threshold at PC{threshold_idx + 1}')

                ax.text(threshold_idx + 1, threshold * 100 + 2,
                       f'PC{threshold_idx + 1}\n({cumulative_variance[threshold_idx]:.1%})',
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Style
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Scree Plot: Variance Explained by Components',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(components)
        ax.set_xticklabels([f'PC{i}' for i in components], rotation=45)
        ax.set_ylim(0, 105)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # CUMULATIVE VARIANCE BAR
    # =========================================================================

    def plot_cumulative_variance_bar(
        self,
        explained_variance_ratio: np.ndarray,
        output_path: Path,
        threshold: float = 0.85,
        title: str = None
    ) -> Path:
        """
        Create stacked bar chart showing cumulative variance

        Args:
            explained_variance_ratio: Variance explained by each component
            output_path: Path to save plot
            threshold: Variance threshold (e.g., 0.85)
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating cumulative variance bar chart: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 8))

        n_components = len(explained_variance_ratio)
        components = np.arange(1, n_components + 1)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Create gradient colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_components))

        # Create stacked bars (one bar per component with bottom offset)
        bottom = 0
        for i, (comp, var) in enumerate(zip(components, explained_variance_ratio)):
            ax.bar(comp, var * 100, bottom=bottom, color=colors[i],
                  edgecolor='black', linewidth=0.5, label=f'PC{comp}' if i < 5 else '')
            bottom += var * 100

        # Threshold line
        ax.axhline(y=threshold * 100, color='red', linestyle='--',
                  linewidth=2, label=f'Threshold ({threshold:.0%})', alpha=0.7)

        # Find number of components needed
        threshold_idx = np.argmax(cumulative_variance >= threshold)
        if cumulative_variance[threshold_idx] >= threshold:
            ax.text(threshold_idx + 1, threshold * 100 + 2,
                   f'{threshold_idx + 1} components\nneeded',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Style
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Cumulative Variance Explained by Components',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(components)
        ax.set_xticklabels([f'PC{i}' for i in components], rotation=45)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test PlotEnginePCAVariance
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE PCA VARIANCE TEST")
    print("=" * 80)

    # Mock PCA data
    np.random.seed(42)
    explained_variance_ratio = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.07])

    output_dir = Path('output/test_pca_variance_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEnginePCAVariance:")
    print("-" * 80)

    try:
        plotter = PlotEnginePCAVariance()

        # Test 1: Scree Plot
        print("\n1. Testing scree plot:")
        path1 = plotter.plot_scree_plot(
            explained_variance_ratio,
            output_path=output_dir / 'scree_plot.png',
            threshold=0.85
        )
        print(f"   Created: {path1.name}")

        # Test 2: Cumulative Variance Bar
        print("\n2. Testing cumulative variance bar:")
        path2 = plotter.plot_cumulative_variance_bar(
            explained_variance_ratio,
            output_path=output_dir / 'cumulative_variance.png',
            threshold=0.85
        )
        print(f"   Created: {path2.name}")

        print("\n✓ PlotEnginePCAVariance test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEnginePCAVariance test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
