"""
Plot Engine for PCA Loadings and Feature Contributions

Provides visualization tools for:
- Component loadings heatmaps
- Biplots (observations + feature vectors)
- Feature contribution polar plots
- PCA vs original space comparisons
- Reconstruction error analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PlotEnginePCALoadings:
    """
    Plotting engine for PCA component loadings and feature contributions

    Supports:
    - Loadings heatmaps (feature contributions to components)
    - Biplots (observations with feature vectors)
    - Polar plots (feature contributions per component)
    - Space comparisons (original vs PCA)
    - Reconstruction error analysis
    """

    def __init__(self, style: str = 'whitegrid', dpi: int = 300):
        """
        Initialize Plot Engine for PCA Loadings

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

        logger.info(f"✓ PlotEnginePCALoadings initialized (DPI={dpi})")

    # =========================================================================
    # COMPONENT LOADINGS HEATMAP
    # =========================================================================

    def plot_component_loadings_heatmap(
        self,
        loadings_df: pd.DataFrame,
        output_path: Path,
        top_n_features: int = None,
        title: str = None
    ) -> Path:
        """
        Create heatmap of component loadings

        Args:
            loadings_df: DataFrame with features (rows) x components (columns)
            output_path: Path to save plot
            top_n_features: Show only top N features by total loading
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating component loadings heatmap: {output_path.name}")

        # Select top features if requested
        if top_n_features and top_n_features < len(loadings_df):
            # Calculate total absolute loading per feature
            total_loading = loadings_df.abs().sum(axis=1)
            top_features = total_loading.nlargest(top_n_features).index
            loadings_df = loadings_df.loc[top_features]

        # Sort features by loading on PC1
        loadings_df = loadings_df.sort_values(loadings_df.columns[0], ascending=False)

        fig, ax = plt.subplots(figsize=(max(10, len(loadings_df.columns) * 1.5),
                                        max(8, len(loadings_df) * 0.4)))

        # Create heatmap
        sns.heatmap(
            loadings_df,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Loading"},
            ax=ax
        )

        # Highlight important loadings (|loading| > 0.4)
        for i in range(len(loadings_df)):
            for j in range(len(loadings_df.columns)):
                value = loadings_df.iloc[i, j]
                if abs(value) > 0.4:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                              edgecolor='black', lw=2))

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Component Loadings: Feature Contributions',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # BIPLOT
    # =========================================================================

    def plot_biplot(
        self,
        X_pca: np.ndarray,
        loadings: np.ndarray,
        feature_names: List[str],
        cluster_labels: np.ndarray,
        output_path: Path,
        pc_x: int = 0,
        pc_y: int = 1,
        n_features_show: int = 10,
        title: str = None
    ) -> Path:
        """
        Create biplot (observations + feature vectors)

        Args:
            X_pca: PCA-transformed data (n_samples x n_components)
            loadings: Component loadings (n_features x n_components)
            feature_names: List of feature names
            cluster_labels: Cluster assignments for observations
            output_path: Path to save plot
            pc_x: Component index for X-axis (default: 0 = PC1)
            pc_y: Component index for Y-axis (default: 1 = PC2)
            n_features_show: Number of top features to show
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating biplot: {output_path.name}")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot observations (scatter)
        unique_clusters = np.unique(cluster_labels)
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            ax.scatter(X_pca[mask, pc_x], X_pca[mask, pc_y],
                      c=[self.cluster_colors[i]], label=f'Cluster {cluster}',
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Scale loadings for visibility
        scale_factor = np.max(np.abs(X_pca[:, [pc_x, pc_y]])) / \
                      np.max(np.abs(loadings[:, [pc_x, pc_y]])) * 0.8

        # Select top features by total loading on these components
        total_loading = np.abs(loadings[:, pc_x]) + np.abs(loadings[:, pc_y])
        top_indices = np.argsort(total_loading)[-n_features_show:]

        # Plot feature vectors (arrows)
        for idx in top_indices:
            dx = loadings[idx, pc_x] * scale_factor
            dy = loadings[idx, pc_y] * scale_factor

            ax.arrow(0, 0, dx, dy,
                    head_width=0.1, head_length=0.1,
                    fc='#e74c3c', ec='#e74c3c', alpha=0.8, linewidth=1.5)

            # Add feature label
            ax.text(dx * 1.15, dy * 1.15, feature_names[idx],
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Style
        ax.set_xlabel(f'PC{pc_x + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y + 1}', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Biplot: PC{pc_x + 1} vs PC{pc_y + 1}',
                        fontsize=14, fontweight='bold', pad=20)

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # FEATURE CONTRIBUTION POLAR
    # =========================================================================

    def plot_feature_contribution_polar(
        self,
        loadings_df: pd.DataFrame,
        component: str,
        output_path: Path,
        top_n: int = 10,
        title: str = None
    ) -> Path:
        """
        Create polar plot showing feature contributions to one component

        Args:
            loadings_df: DataFrame with loadings
            component: Component name (e.g., 'PC1')
            output_path: Path to save plot
            top_n: Number of top features to show
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating feature contribution polar plot: {output_path.name}")

        # Get loadings for this component
        loadings = loadings_df[component].abs().nlargest(top_n)
        feature_names = loadings.index.tolist()
        values = loadings_df.loc[feature_names, component].values

        # Setup polar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Angles
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)

        # Colors based on sign
        colors = ['#e74c3c' if v < 0 else '#3498db' for v in values]

        # Create bars
        bars = ax.bar(angles, np.abs(values), color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1)

        # Feature labels
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, size=10)

        # Radial limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=9)

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Feature Contributions to {component}',
                        fontsize=14, fontweight='bold', pad=20)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.7, label='Positive Loading'),
            Patch(facecolor='#e74c3c', alpha=0.7, label='Negative Loading')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # PCA VS ORIGINAL COMPARISON
    # =========================================================================

    def plot_pca_vs_original_comparison(
        self,
        df_original: pd.DataFrame,
        df_pca: pd.DataFrame,
        X_pca: np.ndarray,
        feature_x: str,
        feature_y: str,
        output_path: Path,
        cluster_column: str = 'cluster',
        title: str = None
    ) -> Path:
        """
        Create side-by-side comparison: original vs PCA space

        Args:
            df_original: DataFrame with original features and clusters
            df_pca: DataFrame with PCA results and clusters
            X_pca: PCA-transformed data
            feature_x: Feature name for X-axis in original space
            feature_y: Feature name for Y-axis in original space
            output_path: Path to save plot
            cluster_column: Cluster column name
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating PCA vs original comparison: {output_path.name}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Original space
        unique_clusters = sorted(df_original[cluster_column].unique())
        for i, cluster in enumerate(unique_clusters):
            mask = df_original[cluster_column] == cluster
            ax1.scatter(df_original.loc[mask, feature_x],
                       df_original.loc[mask, feature_y],
                       c=[self.cluster_colors[i]], label=f'Cluster {cluster}',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        ax1.set_xlabel(feature_x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax1.set_ylabel(feature_y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax1.set_title('Original Space', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Right: PCA space
        unique_clusters_pca = sorted(df_pca[cluster_column].unique())
        for i, cluster in enumerate(unique_clusters_pca):
            mask = df_pca[cluster_column] == cluster
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[self.cluster_colors[i]], label=f'Cluster {cluster}',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax2.set_title('PCA Space', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        else:
            fig.suptitle('Clustering: Original Space vs PCA Space',
                        fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path

    # =========================================================================
    # RECONSTRUCTION ERROR HEATMAP
    # =========================================================================

    def plot_reconstruction_error_heatmap(
        self,
        X_original: np.ndarray,
        X_reconstructed: np.ndarray,
        feature_names: List[str],
        output_path: Path,
        sample_indices: List[int] = None,
        title: str = None
    ) -> Path:
        """
        Create heatmap showing reconstruction errors

        Args:
            X_original: Original data
            X_reconstructed: Reconstructed data from inverse PCA transform
            feature_names: List of feature names
            output_path: Path to save plot
            sample_indices: Optional list of sample indices to show
            title: Optional custom title

        Returns:
            Path to saved plot
        """
        logger.info(f"  Creating reconstruction error heatmap: {output_path.name}")

        # Calculate absolute errors
        errors = np.abs(X_original - X_reconstructed)

        # Select samples if specified
        if sample_indices:
            errors = errors[sample_indices]
            row_labels = [f'Sample {i}' for i in sample_indices]
        else:
            # Show first 20 samples
            n_samples = min(20, errors.shape[0])
            errors = errors[:n_samples]
            row_labels = [f'Sample {i}' for i in range(n_samples)]

        # Create DataFrame
        error_df = pd.DataFrame(errors, columns=feature_names, index=row_labels)

        fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.8),
                                        max(8, len(row_labels) * 0.4)))

        # Create heatmap
        sns.heatmap(
            error_df,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Absolute Error"},
            ax=ax
        )

        # Style
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Reconstruction Error: Original vs PCA-Reconstructed',
                        fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample', fontsize=12, fontweight='bold')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"    ✓ Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test PlotEnginePCALoadings
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PLOT ENGINE PCA LOADINGS TEST")
    print("=" * 80)

    # Mock PCA data
    np.random.seed(42)
    n_samples = 100
    n_features = 12
    n_components = 6

    # Mock component loadings
    loadings = np.random.randn(n_features, n_components)
    # Normalize
    for i in range(n_components):
        loadings[:, i] = loadings[:, i] / np.linalg.norm(loadings[:, i])

    feature_names = ['ROA', 'ROE', 'EBIT_Margin', 'Debt_to_Equity', 'Current_Ratio',
                    'Asset_Turnover', 'Revenue_Growth', 'Net_Margin', 'Quick_Ratio',
                    'Interest_Coverage', 'FCF_Margin', 'Equity_Ratio']

    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # Mock PCA-transformed data
    X_pca = np.random.randn(n_samples, n_components)

    # Mock cluster labels
    cluster_labels = np.random.randint(0, 3, n_samples)

    # Mock original data
    X_original = np.random.randn(n_samples, n_features)
    df_original = pd.DataFrame(X_original, columns=feature_names)
    df_original['cluster'] = cluster_labels

    # Mock PCA DataFrame
    df_pca = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    df_pca['cluster'] = cluster_labels

    output_dir = Path('output/test_pca_loadings_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTesting PlotEnginePCALoadings:")
    print("-" * 80)

    try:
        plotter = PlotEnginePCALoadings()

        # Test 1: Component Loadings Heatmap
        print("\n1. Testing component loadings heatmap:")
        path1 = plotter.plot_component_loadings_heatmap(
            loadings_df,
            output_path=output_dir / 'loadings_heatmap.png',
            top_n_features=10
        )
        print(f"   Created: {path1.name}")

        # Test 2: Biplot
        print("\n2. Testing biplot:")
        path2 = plotter.plot_biplot(
            X_pca,
            loadings,
            feature_names,
            cluster_labels,
            output_path=output_dir / 'biplot.png',
            n_features_show=8
        )
        print(f"   Created: {path2.name}")

        # Test 3: Feature Contribution Polar
        print("\n3. Testing feature contribution polar:")
        path3 = plotter.plot_feature_contribution_polar(
            loadings_df,
            'PC1',
            output_path=output_dir / 'feature_contribution_pc1.png',
            top_n=10
        )
        print(f"   Created: {path3.name}")

        # Test 4: PCA vs Original Comparison
        print("\n4. Testing PCA vs original comparison:")
        path4 = plotter.plot_pca_vs_original_comparison(
            df_original,
            df_pca,
            X_pca,
            'ROA',
            'ROE',
            output_path=output_dir / 'pca_vs_original.png'
        )
        print(f"   Created: {path4.name}")

        # Test 5: Reconstruction Error
        print("\n5. Testing reconstruction error heatmap:")
        X_reconstructed = X_original + np.random.randn(n_samples, n_features) * 0.1
        path5 = plotter.plot_reconstruction_error_heatmap(
            X_original,
            X_reconstructed,
            feature_names,
            output_path=output_dir / 'reconstruction_error.png'
        )
        print(f"   Created: {path5.name}")

        print("\n✓ PlotEnginePCALoadings test successful!")
        print(f"\nAll plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ PlotEnginePCALoadings test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
