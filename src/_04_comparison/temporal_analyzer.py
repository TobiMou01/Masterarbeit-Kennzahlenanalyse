"""
Temporal Stability
Analysiert zeitliche Stabilität von Clustern (Jahr-zu-Jahr)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalStability:
    """Analysiert zeitliche Cluster-Stabilität"""

    def __init__(self):
        self.results = {}

    def compute_migration_matrix(
        self,
        df: pd.DataFrame,
        year_col: str = 'fyear',
        cluster_col: str = 'cluster',
        company_id_col: str = 'gvkey'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Berechnet Migrations-Matrix (Jahr-zu-Jahr)

        Args:
            df: DataFrame mit Zeitreihen-Daten
            year_col: Spalte mit Jahr
            cluster_col: Spalte mit Cluster-ID
            company_id_col: Spalte mit Company ID

        Returns:
            Tuple of (migration_matrix, statistics)
        """
        logger.info("\n  Berechne Migration Matrix...")

        # Sortiere nach Company und Jahr
        df_sorted = df.sort_values([company_id_col, year_col])

        # Filtere Noise
        df_sorted = df_sorted[df_sorted[cluster_col] >= 0]

        # Finde aufeinanderfolgende Jahre
        df_sorted['prev_cluster'] = df_sorted.groupby(company_id_col)[cluster_col].shift(1)
        df_sorted['year_diff'] = df_sorted.groupby(company_id_col)[year_col].diff()

        # Nur aufeinanderfolgende Jahre
        df_migrations = df_sorted[df_sorted['year_diff'] == 1].copy()

        if len(df_migrations) == 0:
            logger.warning("    ⚠ Keine Migrations-Daten gefunden")
            return None, None

        # Migration Matrix
        migration_counts = pd.crosstab(
            df_migrations['prev_cluster'].astype(int),
            df_migrations[cluster_col].astype(int),
            margins=True
        )

        # Statistiken
        total_migrations = len(df_migrations)
        stayed_same = len(df_migrations[df_migrations['prev_cluster'] == df_migrations[cluster_col]])
        consistency_rate = stayed_same / total_migrations if total_migrations > 0 else 0

        stats = {
            'total_migrations': total_migrations,
            'stayed_same': stayed_same,
            'changed_cluster': total_migrations - stayed_same,
            'consistency_rate': consistency_rate
        }

        logger.info(f"    Total year-to-year observations: {total_migrations}")
        logger.info(f"    Stayed in same cluster: {stayed_same} ({consistency_rate*100:.1f}%)")
        logger.info(f"    Changed cluster: {total_migrations - stayed_same} ({(1-consistency_rate)*100:.1f}%)")

        return migration_counts, stats

    def compute_cluster_stability(
        self,
        df: pd.DataFrame,
        year_col: str = 'fyear',
        cluster_col: str = 'cluster',
        company_id_col: str = 'gvkey'
    ) -> pd.DataFrame:
        """
        Berechnet Stabilität pro Cluster

        Args:
            df: DataFrame mit Zeitreihen
            year_col: Jahr-Spalte
            cluster_col: Cluster-Spalte
            company_id_col: Company ID Spalte

        Returns:
            DataFrame mit Cluster-Stabilitäts-Metriken
        """
        logger.info("\n  Berechne Cluster-Stabilität...")

        df_sorted = df.sort_values([company_id_col, year_col])
        df_sorted = df_sorted[df_sorted[cluster_col] >= 0]

        df_sorted['prev_cluster'] = df_sorted.groupby(company_id_col)[cluster_col].shift(1)
        df_sorted['year_diff'] = df_sorted.groupby(company_id_col)[year_col].diff()

        df_migrations = df_sorted[df_sorted['year_diff'] == 1].copy()

        if len(df_migrations) == 0:
            return None

        # Pro Cluster
        stability_data = []

        for cluster_id in sorted(df_migrations[cluster_col].unique()):
            # Alle die im aktuellen Jahr in diesem Cluster sind
            cluster_data = df_migrations[df_migrations[cluster_col] == cluster_id]

            # Wie viele waren auch letztes Jahr schon im gleichen Cluster?
            stayed = cluster_data[cluster_data['prev_cluster'] == cluster_id]

            total = len(cluster_data)
            stayed_count = len(stayed)
            stability_rate = stayed_count / total if total > 0 else 0

            stability_data.append({
                'cluster': cluster_id,
                'total_observations': total,
                'stayed_same': stayed_count,
                'changed_from_other': total - stayed_count,
                'stability_rate': stability_rate
            })

        stability_df = pd.DataFrame(stability_data).sort_values('stability_rate', ascending=False)

        # Log
        logger.info(f"\n    Cluster Stability Ranking:")
        for _, row in stability_df.iterrows():
            logger.info(f"      Cluster {int(row['cluster'])}: {row['stability_rate']*100:.1f}% stability ({row['stayed_same']}/{row['total_observations']})")

        return stability_df

    def analyze_all_algorithms(
        self,
        results_dict: Dict[str, pd.DataFrame],
        year_col: str = 'fyear',
        company_id_col: str = 'gvkey'
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Analysiert zeitliche Stabilität für alle Algorithmen

        Args:
            results_dict: Dict mit {algorithm_name: df_with_time_series}
            year_col: Jahr-Spalte
            company_id_col: Company ID Spalte

        Returns:
            Tuple of (migration_matrices_dict, stability_metrics_df)
        """
        logger.info(f"\n{'='*80}")
        logger.info("TEMPORAL STABILITY ANALYSIS")
        logger.info(f"{'='*80}")

        migration_matrices = {}
        stability_summary = []

        for algorithm, df in results_dict.items():
            if 'cluster' not in df.columns or year_col not in df.columns:
                logger.warning(f"  ⚠ {algorithm}: Fehlende Spalten")
                continue

            logger.info(f"\n--- {algorithm.upper()} ---")

            # Migration Matrix
            migration_matrix, stats = self.compute_migration_matrix(
                df=df,
                year_col=year_col,
                cluster_col='cluster',
                company_id_col=company_id_col
            )

            if migration_matrix is not None:
                migration_matrices[algorithm] = migration_matrix

                # Cluster Stability
                cluster_stability = self.compute_cluster_stability(
                    df=df,
                    year_col=year_col,
                    cluster_col='cluster',
                    company_id_col=company_id_col
                )

                # Summary
                stability_summary.append({
                    'algorithm': algorithm,
                    'total_migrations': stats['total_migrations'],
                    'consistency_rate': stats['consistency_rate'],
                    'avg_cluster_stability': cluster_stability['stability_rate'].mean() if cluster_stability is not None else np.nan,
                    'most_stable_cluster': cluster_stability.iloc[0]['cluster'] if cluster_stability is not None else np.nan,
                    'most_stable_rate': cluster_stability.iloc[0]['stability_rate'] if cluster_stability is not None else np.nan
                })

        # Summary DataFrame
        stability_df = pd.DataFrame(stability_summary)

        # Print Summary
        if len(stability_df) > 0:
            logger.info(f"\n{'='*80}")
            logger.info("=== TEMPORAL STABILITY SUMMARY ===")
            logger.info(f"{'='*80}")

            for _, row in stability_df.iterrows():
                logger.info(f"\n{row['algorithm'].upper()}:")
                logger.info(f"  Average cluster consistency: {row['consistency_rate']*100:.1f}%")
                if not np.isnan(row['most_stable_cluster']):
                    logger.info(f"  Most stable cluster: Cluster {int(row['most_stable_cluster'])} ({row['most_stable_rate']*100:.1f}% stay)")

        return migration_matrices, stability_df

    def plot_migration_heatmap(
        self,
        migration_matrix: pd.DataFrame,
        algorithm_name: str,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Erstellt Heatmap für Migration Matrix

        Args:
            migration_matrix: Migration Matrix
            algorithm_name: Name des Algorithmus
            normalize: Normalisiere Zeilen auf %

        Returns:
            Matplotlib Figure
        """
        # Entferne 'All' Spalte/Zeile falls vorhanden
        matrix = migration_matrix.copy()
        if 'All' in matrix.index:
            matrix = matrix.drop('All')
        if 'All' in matrix.columns:
            matrix = matrix.drop('All', axis=1)

        if normalize:
            # Normalisiere Zeilen (von Cluster X → % zu anderen Clustern)
            matrix_norm = matrix.div(matrix.sum(axis=1), axis=0) * 100
        else:
            matrix_norm = matrix

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix_norm,
            annot=True,
            fmt='.1f' if normalize else 'd',
            cmap='YlOrRd',
            cbar_kws={'label': '% of companies' if normalize else 'Count'},
            ax=ax
        )

        ax.set_title(f'{algorithm_name.upper()} - Year-to-Year Cluster Migration\n(From → To)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster (Year t)', fontsize=12)
        ax.set_ylabel('Cluster (Year t-1)', fontsize=12)

        plt.tight_layout()
        return fig

    def plot_stability_comparison(
        self,
        stability_df: pd.DataFrame
    ) -> plt.Figure:
        """
        Erstellt Vergleichs-Plot für Stabilität

        Args:
            stability_df: Stability Metrics DataFrame

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = stability_df['algorithm'].tolist()
        consistency_rates = stability_df['consistency_rate'].tolist()
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(algorithms)]

        bars = ax.bar(algorithms, [r*100 for r in consistency_rates], color=colors)

        # Werte anzeigen
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Consistency Rate (%)', fontsize=12)
        ax.set_title('Temporal Cluster Stability\n% of companies staying in same cluster year-to-year',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_year_to_year_changes(
        self,
        df: pd.DataFrame,
        algorithm_name: str,
        year_col: str = 'fyear',
        cluster_col: str = 'cluster',
        company_id_col: str = 'gvkey'
    ) -> plt.Figure:
        """
        Zeigt Cluster-Wechsel über Zeit

        Args:
            df: DataFrame mit Zeitreihen
            algorithm_name: Name des Algorithmus
            year_col: Jahr-Spalte
            cluster_col: Cluster-Spalte
            company_id_col: Company ID Spalte

        Returns:
            Matplotlib Figure
        """
        df_sorted = df.sort_values([company_id_col, year_col])
        df_sorted = df_sorted[df_sorted[cluster_col] >= 0]

        df_sorted['prev_cluster'] = df_sorted.groupby(company_id_col)[cluster_col].shift(1)
        df_sorted['year_diff'] = df_sorted.groupby(company_id_col)[year_col].diff()
        df_sorted['changed'] = (df_sorted['prev_cluster'] != df_sorted[cluster_col]).astype(int)

        df_migrations = df_sorted[df_sorted['year_diff'] == 1].copy()

        if len(df_migrations) == 0:
            return None

        # Aggregiere pro Jahr
        yearly_changes = df_migrations.groupby(year_col).agg({
            'changed': ['sum', 'count']
        })
        yearly_changes.columns = ['changed', 'total']
        yearly_changes['stability_rate'] = (1 - yearly_changes['changed'] / yearly_changes['total']) * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        # Line Plot
        ax.plot(yearly_changes.index, yearly_changes['stability_rate'],
                marker='o', linewidth=2, markersize=8, color='steelblue')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Stability Rate (%)', fontsize=12)
        ax.set_title(f'{algorithm_name.upper()} - Cluster Stability Over Time\n% of companies staying in same cluster',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Werte anzeigen
        for year, rate in zip(yearly_changes.index, yearly_changes['stability_rate']):
            ax.text(year, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9)

        plt.tight_layout()
        return fig
