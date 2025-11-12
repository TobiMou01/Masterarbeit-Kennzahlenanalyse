"""
Plot Engine for PCA Cluster Separation Analysis

Provides visualization tools for:
- Cluster separation in PCA space
- 2D scatter plots (any PC combination)
- 3D scatter plots
- Multi-panel cluster comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEnginePCAClusters:
    """
    Plotting engine for cluster separation analysis in PCA space

    Supports:
    - Multi-panel cluster separation plots
    - 2D scatter plots for any PC combination
    - 3D scatter plots with cluster visualization
    - Cluster center identification
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for PCA Cluster Analysis

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

        # Color palettes
        self.cluster_colors = sns.color_palette("husl", n_colors=10)

        logger.info(f"✓ PlotEnginePCAClusters initialized (DPI={dpi})")

    # =========================================================================
    # CLUSTER SEPARATION IN PCA SPACE
    # =========================================================================

    def plot_cluster_separation_in_pca_space(
        self,
        X_pca: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_names: List[str],
        output_path: Path,
        components: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)],
        title: str = None
    ) -> Path:
        """
        Create subplots showing cluster separation in different PC combinations

        Args:
            X_pca: PCA-transformed data
            cluster_labels: Cluster assignments
            cluster_names: List of cluster names
            output_path: Path to save plot
            components: List of (pc_x, pc_y) tuples
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating cluster separation plot: {output_path.name}")

        n_plots = len(components)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

        # Flatten axes for easier indexing
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()

        unique_clusters = np.unique(cluster_labels)

        for idx, (pc_x, pc_y) in enumerate(components):
            ax = axes[idx]

            # Plot each cluster
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_name = cluster_names[cluster] if cluster < len(cluster_names) else f'Cluster {cluster}'

                ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y],
                          c=[self.cluster_colors[i]], label=cluster_name,
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

                # Plot cluster center
                center_x = X_pca[mask, pc_x].mean()
                center_y = X_pca[mask, pc_y].mean()
                ax.scatter(center_x, center_y, c=[self.cluster_colors[i]],
                          marker='X', s=300, edgecolors='black', linewidth=2)

            ax.set_xlabel(f'PC{pc_x + 1}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'PC{pc_y + 1}', fontsize=11, fontweight='bold')
            ax.set_title(f'PC{pc_x + 1} vs PC{pc_y + 1}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        else:
            fig.suptitle('Cluster Separation in PCA Space',
                        fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # PCA 2D SCATTER
    # =========================================================================

    def plot_pca_2d_scatter(
        self,
        X_pca: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_names: List[str],
        output_path: Path,
        pc_x: int = 0,
        pc_y: int = 1,
        show_centers: bool = True,
        show_legend: bool = True,
        title: str = None
    ) -> Path:
        """
        Create 2D scatter plot for specified PC components

        Args:
            X_pca: PCA-transformed data
            cluster_labels: Cluster assignments
            cluster_names: List of cluster names
            output_path: Path to save plot
            pc_x: Component index for X-axis (default: 0 = PC1)
            pc_y: Component index for Y-axis (default: 1 = PC2)
            show_centers: Whether to show cluster centers
            show_legend: Whether to show legend
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating 2D PCA scatter plot: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 10))

        unique_clusters = np.unique(cluster_labels)

        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            cluster_name = cluster_names[cluster] if cluster < len(cluster_names) else f'Cluster {cluster}'

            ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y],
                      c=[self.cluster_colors[i]], label=cluster_name,
                      alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

            # Plot cluster center
            if show_centers:
                center_x = X_pca[mask, pc_x].mean()
                center_y = X_pca[mask, pc_y].mean()
                ax.scatter(center_x, center_y, c=[self.cluster_colors[i]],
                          marker='X', s=400, edgecolors='black', linewidth=2.5,
                          zorder=10)

        # Style
        ax.set_xlabel(f'PC{pc_x + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y + 1}', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Cluster Separation: PC{pc_x + 1} vs PC{pc_y + 1}',
                        fontsize=14, fontweight='bold', pad=20)

        if show_legend:
            ax.legend(loc='best', fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # PCA 3D SCATTER
    # =========================================================================

    def plot_pca_3d_scatter(
        self,
        X_pca: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_names: List[str],
        output_path: Path,
        pc_x: int = 0,
        pc_y: int = 1,
        pc_z: int = 2,
        show_centers: bool = True,
        show_legend: bool = True,
        title: str = None,
        elev: float = 20,
        azim: float = 45
    ) -> Path:
        """
        Create 3D scatter plot for specified PC components

        Args:
            X_pca: PCA-transformed data
            cluster_labels: Cluster assignments
            cluster_names: List of cluster names
            output_path: Path to save plot
            pc_x: Component index for X-axis (default: 0 = PC1)
            pc_y: Component index for Y-axis (default: 1 = PC2)
            pc_z: Component index for Z-axis (default: 2 = PC3)
            show_centers: Whether to show cluster centers
            show_legend: Whether to show legend
            title: Optional custom title
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating 3D PCA scatter plot: {output_path.name}")

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        unique_clusters = np.unique(cluster_labels)

        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            cluster_name = cluster_names[cluster] if cluster < len(cluster_names) else f'Cluster {cluster}'

            ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y], X_pca[mask, pc_z],
                      c=[self.cluster_colors[i]], label=cluster_name,
                      alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

            # Plot cluster center
            if show_centers:
                center_x = X_pca[mask, pc_x].mean()
                center_y = X_pca[mask, pc_y].mean()
                center_z = X_pca[mask, pc_z].mean()
                ax.scatter(center_x, center_y, center_z, c=[self.cluster_colors[i]],
                          marker='X', s=500, edgecolors='black', linewidth=2.5,
                          zorder=10)

        # Style
        ax.set_xlabel(f'PC{pc_x + 1}', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel(f'PC{pc_y + 1}', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel(f'PC{pc_z + 1}', fontsize=12, fontweight='bold', labelpad=10)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Cluster Separation: PC{pc_x + 1} vs PC{pc_y + 1} vs PC{pc_z + 1}',
                        fontsize=14, fontweight='bold', pad=20)

        if show_legend:
            ax.legend(loc='best', fontsize=10)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test PlotEnginePCAClusters
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE PCA CLUSTERS TEST")
    print("=" * 80)

    # Mock PCA data
    np.random.seed(42)
    n_samples = 100
    n_components = 6

    # Mock PCA-transformed data
    X_pca = np.random.randn(n_samples, n_components)

    # Mock cluster labels
    cluster_labels = np.random.randint(0, 3, n_samples)
    cluster_names = ['High Performers', 'Mid-Range', 'Low Performers']

    output_dir = Path('output/test_pca_cluster_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEnginePCAClusters:")
    print("-" * 80)

    try:
        plotter = PlotEnginePCAClusters()

        # Test 1: Cluster Separation (Multi-panel)
        print("\n1. Testing cluster separation (multi-panel):")
        path1 = plotter.plot_cluster_separation_in_pca_space(
            X_pca,
            cluster_labels,
            cluster_names,
            output_path=output_dir / 'cluster_separation.png',
            components=[(0, 1), (0, 2), (1, 2)]
        )
        print(f"   Created: {path1.name}")

        # Test 2: 2D Scatter
        print("\n2. Testing 2D scatter plot:")
        path2 = plotter.plot_pca_2d_scatter(
            X_pca,
            cluster_labels,
            cluster_names,
            output_path=output_dir / 'pca_2d_scatter.png',
            pc_x=0,
            pc_y=1,
            show_centers=True
        )
        print(f"   Created: {path2.name}")

        # Test 3: 3D Scatter
        print("\n3. Testing 3D scatter plot:")
        path3 = plotter.plot_pca_3d_scatter(
            X_pca,
            cluster_labels,
            cluster_names,
            output_path=output_dir / '3d_scatter.png',
            pc_x=0,
            pc_y=1,
            pc_z=2,
            show_centers=True
        )
        print(f"   Created: {path3.name}")

        print("\n✓ PlotEnginePCAClusters test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEnginePCAClusters test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
