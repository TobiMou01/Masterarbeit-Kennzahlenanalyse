"""
Comparison Handler
Verwaltet die Output-Struktur für Vergleichsanalysen
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class ComparisonHandler:
    """Verwaltet alle Comparison Outputs"""

    def __init__(self, market: str = 'germany', base_dir: str = 'output'):
        """
        Initialisiert Comparison Handler

        Args:
            market: Market-Bezeichnung (germany, usa, etc.)
            base_dir: Basis-Verzeichnis
        """
        self.market = market
        self.base_dir = Path(base_dir) / market / 'comparisons'

        # Verzeichnis-Struktur
        self.dirs = {
            'gics': self.base_dir / '01_gics_comparison',
            'gics_tables': self.base_dir / '01_gics_comparison' / 'contingency_tables',
            'algorithms': self.base_dir / '02_algorithm_comparison',
            'features': self.base_dir / '03_feature_importance',
            'temporal': self.base_dir / '04_temporal_stability',
            'temporal_migrations': self.base_dir / '04_temporal_stability' / 'migration_matrices'
        }

        # Erstelle alle Verzeichnisse
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Comparison-Struktur: {self.base_dir}")

    def save_gics_comparison(
        self,
        cramers_v: pd.DataFrame,
        rand_index: pd.DataFrame,
        chi_square: pd.DataFrame,
        contingency_tables: Dict[str, pd.DataFrame]
    ):
        """
        Speichert GICS-Vergleichsergebnisse

        Args:
            cramers_v: Cramér's V Werte
            rand_index: Adjusted Rand Index Werte
            chi_square: Chi²-Test Ergebnisse
            contingency_tables: Kontingenztabellen pro Algorithmus
        """
        logger.info("\n  Speichere GICS Comparison...")

        # Statistische Maße
        cramers_v.to_csv(self.dirs['gics'] / 'cramers_v.csv')
        logger.info(f"    ✓ Cramér's V: cramers_v.csv")

        rand_index.to_csv(self.dirs['gics'] / 'rand_index.csv')
        logger.info(f"    ✓ Rand Index: rand_index.csv")

        chi_square.to_csv(self.dirs['gics'] / 'chi_square_test.csv')
        logger.info(f"    ✓ Chi²-Test: chi_square_test.csv")

        # Kontingenztabellen
        for name, table in contingency_tables.items():
            safe_name = name.lower().replace(' ', '_')
            table.to_csv(self.dirs['gics_tables'] / f'{safe_name}.csv')
            logger.info(f"    ✓ Contingency Table: {safe_name}.csv")

    def save_algorithm_comparison(
        self,
        metrics: pd.DataFrame,
        overlap_matrix: pd.DataFrame
    ):
        """
        Speichert Algorithmen-Vergleich

        Args:
            metrics: Metriken-Vergleich
            overlap_matrix: Cluster-Überlappungsmatrix
        """
        logger.info("\n  Speichere Algorithm Comparison...")

        metrics.to_csv(self.dirs['algorithms'] / 'metrics_comparison.csv')
        logger.info(f"    ✓ Metrics: metrics_comparison.csv")

        overlap_matrix.to_csv(self.dirs['algorithms'] / 'cluster_overlap_matrix.csv')
        logger.info(f"    ✓ Overlap Matrix: cluster_overlap_matrix.csv")

    def save_feature_importance(
        self,
        importance_data: Dict[str, pd.DataFrame]
    ):
        """
        Speichert Feature Importance

        Args:
            importance_data: Dict mit importance DataFrames pro Algorithmus
        """
        logger.info("\n  Speichere Feature Importance...")

        for algorithm, df in importance_data.items():
            filename = f'{algorithm}_importance.csv'
            df.to_csv(self.dirs['features'] / filename)
            logger.info(f"    ✓ {algorithm.capitalize()}: {filename}")

    def save_temporal_stability(
        self,
        migration_matrices: Dict[str, pd.DataFrame],
        stability_metrics: pd.DataFrame
    ):
        """
        Speichert zeitliche Stabilitätsanalyse

        Args:
            migration_matrices: Migrations-Matrizen pro Algorithmus
            stability_metrics: Stabilitäts-Metriken
        """
        logger.info("\n  Speichere Temporal Stability...")

        # Migrations-Matrizen
        for algorithm, matrix in migration_matrices.items():
            filename = f'{algorithm}_migration.csv'
            matrix.to_csv(self.dirs['temporal_migrations'] / filename)
            logger.info(f"    ✓ Migration ({algorithm}): {filename}")

        # Stabilitäts-Metriken
        stability_metrics.to_csv(self.dirs['temporal'] / 'stability_metrics.csv')
        logger.info(f"    ✓ Stability Metrics: stability_metrics.csv")

    def save_plot(self, fig, filename: str, category: str):
        """
        Speichert Plot in entsprechender Kategorie

        Args:
            fig: Matplotlib/Seaborn Figure
            filename: Dateiname (mit .png)
            category: 'gics', 'algorithms', 'features', oder 'temporal'
        """
        if category == 'gics_tables':
            path = self.dirs['gics_tables'] / filename
        else:
            path = self.dirs.get(category, self.dirs['gics']) / filename

        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"    ✓ Plot: {filename}")

    def get_paths(self) -> Dict[str, Path]:
        """Gibt alle Pfade zurück"""
        return self.dirs.copy()
