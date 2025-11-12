"""
Plot Engine for Validation Matrices Visualizations

Provides comprehensive visualization tools for:
- Confusion matrices (cluster overlap)
- Contingency tables (cluster-external overlap patterns)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEngineValidationMatrices:
    """
    Plotting engine for validation matrices visualizations

    Supports:
    - Confusion matrices (cluster overlap)
    - Multiple confusion matrices (grid layout)
    - Contingency tables (cluster-external overlap patterns)
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for Validation Matrices

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

        logger.info(f"✓ PlotEngineValidationMatrices initialized (DPI={dpi})")

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


if __name__ == "__main__":
    # Test PlotEngineValidationMatrices
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE VALIDATION MATRICES TEST")
    print("=" * 80)

    # Mock data
    np.random.seed(42)

    # 1. Confusion Matrix
    confusion = pd.DataFrame(
        [[85, 12, 3],
         [10, 78, 12],
         [5, 10, 85]],
        index=['C0', 'C1', 'C2'],
        columns=['C0', 'C1', 'C2']
    )

    # 2. Contingency Table
    contingency = pd.DataFrame(
        [[45, 12, 8, 5],
         [18, 55, 15, 7],
         [7, 13, 62, 10]],
        index=[0, 1, 2],
        columns=['GICS_A', 'GICS_B', 'GICS_C', 'GICS_D']
    )

    output_dir = Path('output/test_validation_matrices')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEngineValidationMatrices:")
    print("-" * 80)

    try:
        plotter = PlotEngineValidationMatrices()

        # Test 1: Confusion Matrix
        print("\n1. Testing confusion matrix:")
        path1 = plotter.plot_confusion_matrix(
            confusion, 'K-Means', 'Hierarchical',
            output_path=output_dir / 'confusion_matrix.png'
        )
        print(f"   Created: {path1.name}")

        # Test 2: Multiple Confusion Matrices
        print("\n2. Testing multiple confusion matrices:")
        conf_matrices = {
            ('K-Means', 'Hierarchical'): confusion,
            ('K-Means', 'DBSCAN'): confusion,
            ('Hierarchical', 'DBSCAN'): confusion
        }
        path2 = plotter.plot_multiple_confusion_matrices(
            conf_matrices,
            output_path=output_dir / 'confusion_matrices_grid.png'
        )
        print(f"   Created: {path2.name}")

        # Test 3: Contingency Heatmap
        print("\n3. Testing contingency heatmap:")
        path3 = plotter.plot_contingency_heatmap(
            contingency, 'Cluster', 'GICS',
            output_path=output_dir / 'contingency.png'
        )
        print(f"   Created: {path3.name}")

        print("\n✓ PlotEngineValidationMatrices test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEngineValidationMatrices test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
