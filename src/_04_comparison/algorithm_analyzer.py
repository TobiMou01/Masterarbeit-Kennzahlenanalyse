"""
Algorithm Comparison
Vergleicht verschiedene Clustering-Algorithmen untereinander
"""

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class AlgorithmComparison:
    """Vergleicht Clustering-Algorithmen"""

    def __init__(self):
        self.results = {}

    def compare_metrics(
        self,
        metrics_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Vergleicht Metriken verschiedener Algorithmen

        Args:
            metrics_dict: Dict mit {algorithm_name: metrics_dict}

        Returns:
            DataFrame mit Metriken-Vergleich
        """
        logger.info(f"\n{'='*80}")
        logger.info("ALGORITHM COMPARISON - METRICS")
        logger.info(f"{'='*80}")

        comparison_data = []

        for algorithm, metrics in metrics_dict.items():
            row = {
                'algorithm': algorithm,
                'n_clusters': metrics.get('n_clusters', 0),
                'n_samples': metrics.get('n_samples', 0),
                'silhouette': metrics.get('silhouette', np.nan),
                'davies_bouldin': metrics.get('davies_bouldin', np.nan),
                'calinski_harabasz': metrics.get('calinski_harabasz', np.nan),
                'n_noise': metrics.get('n_noise', 0),  # Nur DBSCAN
                'noise_pct': metrics.get('noise_pct', 0.0)
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Print Vergleich
        logger.info("\nMetrics Comparison:")
        logger.info(f"{'Algorithm':<15} {'Clusters':<10} {'Silhouette':<12} {'Davies-B':<12} {'Noise%':<10}")
        logger.info("-" * 80)

        for _, row in df.iterrows():
            noise_str = f"{row['noise_pct']:.1f}%" if row['noise_pct'] > 0 else "-"
            logger.info(
                f"{row['algorithm']:<15} "
                f"{row['n_clusters']:<10} "
                f"{row['silhouette']:<12.3f} "
                f"{row['davies_bouldin']:<12.3f} "
                f"{noise_str:<10}"
            )

        # Best Performers
        logger.info(f"\n{'='*80}")
        logger.info("=== BEST PERFORMERS ===")
        logger.info(f"{'='*80}")

        if df['silhouette'].notna().any():
            best_silhouette = df.loc[df['silhouette'].idxmax()]
            logger.info(f"Best Silhouette Score: {best_silhouette['algorithm']} ({best_silhouette['silhouette']:.3f})")

        if df['davies_bouldin'].notna().any():
            best_db = df.loc[df['davies_bouldin'].idxmin()]
            logger.info(f"Best Davies-Bouldin:   {best_db['algorithm']} ({best_db['davies_bouldin']:.3f})")

        if df['calinski_harabasz'].notna().any():
            best_ch = df.loc[df['calinski_harabasz'].idxmax()]
            logger.info(f"Best Calinski-Harabasz: {best_ch['algorithm']} ({best_ch['calinski_harabasz']:.1f})")

        most_clusters = df.loc[df['n_clusters'].idxmax()]
        logger.info(f"Most Clusters Found:   {most_clusters['algorithm']} ({most_clusters['n_clusters']} clusters)")

        return df

    def compute_cluster_overlap(
        self,
        results_dict: Dict[str, pd.DataFrame],
        company_id_col: str = 'gvkey'
    ) -> pd.DataFrame:
        """
        Berechnet Cluster-Überlappung zwischen Algorithmen

        Args:
            results_dict: Dict mit {algorithm_name: df_with_clusters}
            company_id_col: Spalte mit Company ID

        Returns:
            Overlap-Matrix als DataFrame
        """
        logger.info("\n  Berechne Cluster-Überlappung...")

        algorithms = list(results_dict.keys())
        n_algos = len(algorithms)
        overlap_matrix = np.zeros((n_algos, n_algos))

        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                elif i < j:
                    # Finde gemeinsame Unternehmen
                    df1 = results_dict[algo1]
                    df2 = results_dict[algo2]

                    if company_id_col not in df1.columns or company_id_col not in df2.columns:
                        continue

                    # Merge auf gemeinsame IDs
                    merged = df1[[company_id_col, 'cluster']].merge(
                        df2[[company_id_col, 'cluster']],
                        on=company_id_col,
                        suffixes=('_1', '_2')
                    )

                    # Filtere Noise
                    merged_valid = merged[
                        (merged['cluster_1'] >= 0) &
                        (merged['cluster_2'] >= 0)
                    ]

                    if len(merged_valid) == 0:
                        overlap_matrix[i, j] = 0.0
                        overlap_matrix[j, i] = 0.0
                        continue

                    # Adjusted Rand Index
                    ari = adjusted_rand_score(
                        merged_valid['cluster_1'],
                        merged_valid['cluster_2']
                    )

                    overlap_matrix[i, j] = ari
                    overlap_matrix[j, i] = ari

                    logger.info(f"    {algo1} <-> {algo2}: ARI = {ari:.3f}")

        df_overlap = pd.DataFrame(
            overlap_matrix,
            index=algorithms,
            columns=algorithms
        )

        return df_overlap

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        analysis_type: str = 'static'
    ) -> plt.Figure:
        """
        Erstellt Metrics Comparison Plot

        Args:
            metrics_df: DataFrame mit Metriken
            analysis_type: Analyse-Typ

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        algorithms = metrics_df['algorithm'].tolist()
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(algorithms)]

        # 1. Silhouette Score (höher = besser)
        ax1 = axes[0, 0]
        ax1.bar(algorithms, metrics_df['silhouette'], color=colors)
        ax1.set_title('Silhouette Score (higher = better)', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['silhouette']):
            if not np.isnan(v):
                ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Davies-Bouldin Index (niedriger = besser)
        ax2 = axes[0, 1]
        ax2.bar(algorithms, metrics_df['davies_bouldin'], color=colors)
        ax2.set_title('Davies-Bouldin Index (lower = better)', fontweight='bold')
        ax2.set_ylabel('Index')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['davies_bouldin']):
            if not np.isnan(v):
                ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Number of Clusters
        ax3 = axes[1, 0]
        ax3.bar(algorithms, metrics_df['n_clusters'], color=colors)
        ax3.set_title('Number of Clusters', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['n_clusters']):
            ax3.text(i, v, f'{int(v)}', ha='center', va='bottom', fontweight='bold')

        # 4. Noise Percentage (nur relevant für DBSCAN)
        ax4 = axes[1, 1]
        ax4.bar(algorithms, metrics_df['noise_pct'], color=colors)
        ax4.set_title('Noise Points (%)', fontweight='bold')
        ax4.set_ylabel('Percentage')
        ax4.grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['noise_pct']):
            if v > 0:
                ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        fig.suptitle(f'Algorithm Metrics Comparison - {analysis_type.upper()}',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig

    def plot_overlap_heatmap(
        self,
        overlap_matrix: pd.DataFrame,
        analysis_type: str = 'static'
    ) -> plt.Figure:
        """
        Erstellt Heatmap für Cluster-Überlappung

        Args:
            overlap_matrix: Overlap-Matrix
            analysis_type: Analyse-Typ

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Adjusted Rand Index'},
            ax=ax
        )

        ax.set_title(f'Algorithm Cluster Overlap - {analysis_type.upper()}\nAdjusted Rand Index',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig
