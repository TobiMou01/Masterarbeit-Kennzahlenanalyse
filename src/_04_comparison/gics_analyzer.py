"""
GICS Comparison
Vergleicht Clustering-Ergebnisse mit GICS-Sektoren
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GICSComparison:
    """Vergleicht Cluster mit GICS-Klassifikation"""

    def __init__(self):
        self.results = {}

    def cramers_v(self, confusion_matrix: np.ndarray) -> float:
        """
        Berechnet Cramér's V für nominale Variablen

        Args:
            confusion_matrix: Kontingenztabelle

        Returns:
            Cramér's V (0 = keine Korrelation, 1 = perfekte Korrelation)
        """
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1

        if min_dim == 0:
            return 0.0

        return np.sqrt(chi2 / (n * min_dim))

    def compare_with_gics(
        self,
        df: pd.DataFrame,
        cluster_col: str = 'cluster',
        gics_col: str = 'gsector',
        algorithm_name: str = 'algorithm'
    ) -> Dict:
        """
        Vergleicht Clustering mit GICS

        Args:
            df: DataFrame mit Cluster und GICS Spalten
            cluster_col: Name der Cluster-Spalte
            gics_col: Name der GICS-Spalte ('gsector', 'gind', etc.)
            algorithm_name: Name des Algorithmus

        Returns:
            Dict mit Ergebnissen
        """
        logger.info(f"\n  Vergleiche {algorithm_name} mit {gics_col}...")

        # Filtere gültige Daten
        df_valid = df[[cluster_col, gics_col]].dropna()

        # Filtere Noise (cluster == -1) aus
        df_valid = df_valid[df_valid[cluster_col] >= 0]

        if len(df_valid) == 0:
            logger.warning(f"    ⚠ Keine gültigen Daten für Vergleich")
            return None

        # Kontingenztabelle
        contingency = pd.crosstab(df_valid[cluster_col], df_valid[gics_col])

        # Statistische Maße
        cramers = self.cramers_v(contingency.values)
        rand_idx = adjusted_rand_score(df_valid[gics_col], df_valid[cluster_col])

        # Chi²-Test
        chi2, p_value, dof, expected = chi2_contingency(contingency.values)

        logger.info(f"    ✓ Cramér's V: {cramers:.3f}")
        logger.info(f"    ✓ Adjusted Rand Index: {rand_idx:.3f}")
        logger.info(f"    ✓ Chi²: {chi2:.2f} (p={p_value:.4f})")

        # Interpretation
        if cramers < 0.2:
            interpretation = "sehr schwache Korrelation ✓ (gut!)"
        elif cramers < 0.3:
            interpretation = "schwache Korrelation ✓"
        elif cramers < 0.5:
            interpretation = "moderate Korrelation"
        else:
            interpretation = "starke Korrelation (Cluster folgen Branchen)"

        logger.info(f"    → {interpretation}")

        return {
            'algorithm': algorithm_name,
            'gics_level': gics_col,
            'cramers_v': cramers,
            'rand_index': rand_idx,
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'interpretation': interpretation,
            'contingency_table': contingency,
            'n_samples': len(df_valid)
        }

    def compare_all_algorithms(
        self,
        results_dict: Dict[str, pd.DataFrame],
        gics_col: str = 'gsector',
        analysis_type: str = 'static'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Vergleicht alle Algorithmen mit GICS

        Args:
            results_dict: Dict mit {algorithm_name: df_with_clusters}
            gics_col: GICS-Spalte
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Tuple of (cramers_df, rand_df, chi2_df, contingency_tables)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"GICS COMPARISON - {analysis_type.upper()}")
        logger.info(f"{'='*80}")

        comparisons = []
        contingency_tables = {}

        for algorithm, df in results_dict.items():
            if 'cluster' not in df.columns or gics_col not in df.columns:
                logger.warning(f"  ⚠ {algorithm}: Fehlende Spalten")
                continue

            result = self.compare_with_gics(
                df=df,
                cluster_col='cluster',
                gics_col=gics_col,
                algorithm_name=algorithm
            )

            if result:
                comparisons.append(result)
                contingency_tables[f"{algorithm}_vs_{gics_col}"] = result['contingency_table']

        if not comparisons:
            logger.warning("  ⚠ Keine Vergleiche möglich")
            return None, None, None, None

        # Zusammenfassung als DataFrames
        cramers_df = pd.DataFrame([
            {'algorithm': c['algorithm'], 'cramers_v': c['cramers_v'],
             'interpretation': c['interpretation']}
            for c in comparisons
        ])

        rand_df = pd.DataFrame([
            {'algorithm': c['algorithm'], 'rand_index': c['rand_index']}
            for c in comparisons
        ])

        chi2_df = pd.DataFrame([
            {'algorithm': c['algorithm'], 'chi2': c['chi2'],
             'p_value': c['p_value'], 'dof': c['dof']}
            for c in comparisons
        ])

        # Print Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"=== GICS COMPARISON SUMMARY ({analysis_type}) ===")
        logger.info(f"{'='*80}")

        for comp in comparisons:
            logger.info(f"{comp['algorithm']:15} vs GICS: Cramér's V = {comp['cramers_v']:.3f} ({comp['interpretation']})")

        # Finde besten (niedrigsten) Cramér's V
        best = min(comparisons, key=lambda x: x['cramers_v'])
        logger.info(f"\n→ Beste GICS-Unabhängigkeit: {best['algorithm']} (V={best['cramers_v']:.3f})")

        return cramers_df, rand_df, chi2_df, contingency_tables

    def plot_contingency_heatmap(
        self,
        contingency_table: pd.DataFrame,
        algorithm_name: str,
        gics_col: str = 'gsector'
    ) -> plt.Figure:
        """
        Erstellt Heatmap für Kontingenztabelle

        Args:
            contingency_table: Kontingenztabelle
            algorithm_name: Name des Algorithmus
            gics_col: GICS-Spalte

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Normalisiere für bessere Visualisierung
        contingency_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        sns.heatmap(
            contingency_norm,
            annot=contingency_table.values,  # Zeige absolute Zahlen
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': '% of cluster'},
            ax=ax
        )

        ax.set_title(f'{algorithm_name.upper()} Clusters vs. {gics_col.upper()}\nContingency Table',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{gics_col.upper()} Sector', fontsize=12)
        ax.set_ylabel('Cluster ID', fontsize=12)

        plt.tight_layout()
        return fig

    def plot_summary_comparison(
        self,
        cramers_df: pd.DataFrame,
        analysis_type: str = 'static'
    ) -> plt.Figure:
        """
        Erstellt Summary-Plot für alle Algorithmen

        Args:
            cramers_df: DataFrame mit Cramér's V Werten
            analysis_type: Analyse-Typ

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar Chart
        bars = ax.bar(
            cramers_df['algorithm'],
            cramers_df['cramers_v'],
            color=['#3498db', '#e74c3c', '#2ecc71'][:len(cramers_df)]
        )

        # Threshold-Linien
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5,
                   label='Very Weak (< 0.2) ✓')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5,
                   label='Weak (< 0.3)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                   label='Moderate (< 0.5)')

        # Beschriftung
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_title(f'GICS Independence Comparison - {analysis_type.upper()}\nCramér\'s V (lower = better)',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Cramér\'s V', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylim(0, max(0.6, cramers_df['cramers_v'].max() * 1.2))
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
