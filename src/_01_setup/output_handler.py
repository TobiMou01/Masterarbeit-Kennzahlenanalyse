"""
Output Handler - Option B (Finale Struktur)
Unterscheidet zwischen Comparative und Hierarchical Mode
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import joblib
import json

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Verwaltet alle Outputs in finaler Option B Struktur

    Struktur:
    output/{market}/
    ├── 01_data/                      # Rohdaten
    ├── 02_algorithms/                # Pro Algorithmus
    │   ├── kmeans_comparative/       # K-Means: 3 separate Clusterings
    │   │   ├── static/
    │   │   ├── dynamic/
    │   │   └── combined/
    │   ├── hierarchical/             # Hierarchical: Label-Consistency
    │   │   ├── master_clustering/
    │   │   ├── dynamic_enrichment/
    │   │   └── combined_scores/
    │   └── dbscan/                   # DBSCAN: Label-Consistency
    │       ├── master_clustering/
    │       ├── dynamic_enrichment/
    │       └── combined_scores/
    ├── 03_comparisons/               # Cross-Analysen
    │   ├── algorithms/
    │   ├── gics/
    │   ├── features/
    │   └── temporal/
    └── 99_summary/                   # Executive Reports
    """

    def __init__(
        self,
        market: str = 'germany',
        algorithm: str = 'kmeans',
        mode: str = 'auto',
        base_dir: str = 'output'
    ):
        """
        Initialisiert Output Handler

        Args:
            market: Market-Bezeichnung (germany, usa, etc.)
            algorithm: Clustering-Algorithmus ('kmeans', 'hierarchical', 'dbscan')
            mode: 'comparative' oder 'hierarchical' (oder 'auto' für Auto-Detect)
            base_dir: Basis-Verzeichnis
        """
        self.market = market
        self.algorithm = algorithm

        # Auto-detect mode based on algorithm
        if mode == 'auto':
            if algorithm == 'kmeans':
                self.mode = 'comparative'  # Default für K-Means
            else:
                self.mode = 'hierarchical'  # Hierarchical/DBSCAN nutzen Label-Consistency
        else:
            self.mode = mode

        # Base paths
        self.market_dir = Path(base_dir) / market
        self.data_dir = self.market_dir / '01_data'

        # Algorithm directory name
        if self.mode == 'comparative' and algorithm == 'kmeans':
            self.algorithm_name = 'kmeans_comparative'
        else:
            self.algorithm_name = algorithm

        self.algorithm_dir = self.market_dir / '02_algorithms' / self.algorithm_name
        self.comparisons_dir = self.market_dir / '03_comparisons'
        self.summary_dir = self.market_dir / '99_summary'

        # Analysis type naming based on mode
        if self.mode == 'comparative':
            self.analysis_types = {
                'static': 'static',
                'dynamic': 'dynamic',
                'combined': 'combined'
            }
        else:  # hierarchical mode
            self.analysis_types = {
                'static': 'master_clustering',
                'dynamic': 'dynamic_enrichment',
                'combined': 'combined_scores'
            }

        # Create directory structure
        self._create_directories()

        logger.info(f"✓ OutputHandler: {market} / {self.algorithm_name} ({self.mode} mode)")

    def _create_directories(self):
        """Erstellt die komplette Verzeichnisstruktur"""

        # 01_data/
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 02_algorithms/{algorithm}/
        for analysis_key in ['static', 'dynamic', 'combined']:
            analysis_dir = self.algorithm_dir / self.analysis_types[analysis_key]

            # data/, plots/, reports/
            (analysis_dir / 'data').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'plots').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'reports').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'reports' / 'clusters').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'models').mkdir(parents=True, exist_ok=True)

        # 03_comparisons/
        for comp_type in ['algorithms', 'gics', 'features', 'temporal']:
            (self.comparisons_dir / comp_type / 'data').mkdir(parents=True, exist_ok=True)
            (self.comparisons_dir / comp_type / 'plots').mkdir(parents=True, exist_ok=True)

        # 99_summary/
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def save_cluster_data(
        self,
        df: pd.DataFrame,
        cluster_profiles: pd.DataFrame,
        analysis_type: str = 'static',
        metrics: Dict = None
    ):
        """
        Speichert Cluster-Daten

        Args:
            df: DataFrame mit Cluster-Spalte
            cluster_profiles: Cluster-Profile
            analysis_type: 'static', 'dynamic', oder 'combined'
            metrics: Clustering-Metriken
        """
        analysis_name = self.analysis_types[analysis_type]
        data_dir = self.algorithm_dir / analysis_name / 'data'

        logger.info(f"\n  Speichere {analysis_type} → {analysis_name}/data/")

        # 1. Assignments
        assignments_path = data_dir / 'assignments.csv'
        df.to_csv(assignments_path, index=False)
        logger.info(f"    ✓ {assignments_path.name} ({len(df)} Unternehmen)")

        # 2. Profiles
        profiles_path = data_dir / 'profiles.csv'
        cluster_profiles.to_csv(profiles_path)
        logger.info(f"    ✓ {profiles_path.name} ({len(cluster_profiles)} Cluster)")

        # 3. Metrics
        if metrics:
            metrics_clean = {k: v for k, v in metrics.items()
                           if k not in ['scaler', 'model']}
            metrics_path = data_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_clean, f, indent=2)
            logger.info(f"    ✓ {metrics_path.name}")

    def save_cluster_lists(
        self,
        df: pd.DataFrame,
        analysis_type: str = 'static',
        sort_by: str = 'roa'
    ):
        """Erstellt separate CSV pro Cluster"""
        analysis_name = self.analysis_types[analysis_type]
        clusters_dir = self.algorithm_dir / analysis_name / 'reports' / 'clusters'

        logger.info(f"\n  Erstelle Cluster-Listen ({analysis_type})...")

        valid_df = df[df['cluster'] >= 0]

        for cluster_id in sorted(valid_df['cluster'].unique()):
            cluster_df = valid_df[valid_df['cluster'] == cluster_id].copy()

            cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in cluster_df.columns else f'cluster_{cluster_id}'
            safe_name = cluster_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

            if sort_by in cluster_df.columns:
                cluster_df = cluster_df.sort_values(sort_by, ascending=False)

            filename = f'{cluster_id}_{safe_name}.csv'
            path = clusters_dir / filename
            cluster_df.to_csv(path, index=False)

            logger.info(f"    ✓ Cluster {cluster_id}: {len(cluster_df):3} Unternehmen → {filename}")

    def save_models(
        self,
        scaler,
        model,
        analysis_type: str = 'static',
        pca_model = None
    ):
        """Speichert ML-Modelle"""
        analysis_name = self.analysis_types[analysis_type]
        models_dir = self.algorithm_dir / analysis_name / 'models'

        joblib.dump(scaler, models_dir / 'scaler.pkl')
        joblib.dump(model, models_dir / 'model.pkl')

        if pca_model is not None:
            joblib.dump(pca_model, models_dir / 'pca_model.pkl')
            logger.info(f"    ✓ Models: scaler, model, pca_model")
        else:
            logger.info(f"    ✓ Models: scaler, model")

    def get_plots_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Plots-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'plots'

    def get_reports_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Reports-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'reports'

    def save_processed_features(self, df: pd.DataFrame):
        """Speichert verarbeitete Features in 01_data/"""
        path = self.data_dir / 'processed_features.csv'
        df.to_csv(path, index=False)
        logger.info(f"✓ Processed Features: {path}")

    def save_comparison_data(
        self,
        comp_type: str,
        data: pd.DataFrame,
        filename: str
    ):
        """Speichert Comparison-Daten"""
        data_dir = self.comparisons_dir / comp_type / 'data'
        path = data_dir / filename
        data.to_csv(path, index=False)
        logger.info(f"    ✓ {comp_type}/{filename}")

    def get_comparison_plots_dir(self, comp_type: str) -> Path:
        """Gibt Comparison Plots Dir zurück"""
        return self.comparisons_dir / comp_type / 'plots'

    def create_readme(self):
        """Erstellt README in 99_summary/"""
        readme_content = f"""# Clustering Analysis Results - {self.market.upper()}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Structure

### 01_data/
Processed feature data used for clustering

### 02_algorithms/
Clustering results per algorithm:
- `{self.algorithm_name}/` - {self.algorithm.upper()} results
  - `{self.analysis_types['static']}/` - Static features analysis
  - `{self.analysis_types['dynamic']}/` - Dynamic features analysis
  - `{self.analysis_types['combined']}/` - Combined analysis

### 03_comparisons/
Cross-algorithm comparisons:
- `algorithms/` - Performance metrics comparison
- `gics/` - Independence from GICS sectors
- `features/` - Feature importance analysis
- `temporal/` - Temporal stability

### 99_summary/
Executive summaries and interpretation reports

## Mode

This analysis uses **{self.mode.upper()} Mode**:
"""

        if self.mode == 'comparative':
            readme_content += """
- Static, Dynamic, and Combined are **3 independent clusterings**
- Each uses different feature sets
- Clusters are not directly comparable across analyses
"""
        else:
            readme_content += """
- Static creates **master cluster labels**
- Dynamic and Combined **reuse same labels**
- Only scores change, not cluster assignments
- Allows tracking companies across feature dimensions
"""

        readme_path = self.summary_dir / 'README.md'
        readme_path.write_text(readme_content)
        logger.info(f"✓ README: {readme_path}")

    def __repr__(self):
        return f"OutputHandler(market='{self.market}', algorithm='{self.algorithm_name}', mode='{self.mode}')"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("OUTPUT HANDLER TEST - Option B")
    print("="*80)

    # Test Comparative Mode (K-Means)
    print("\n1. K-Means Comparative Mode:")
    handler_km = OutputHandler(market='germany', algorithm='kmeans', mode='comparative')
    print(f"   Algorithm Dir: {handler_km.algorithm_dir}")
    print(f"   Analysis Types: {handler_km.analysis_types}")

    # Test Hierarchical Mode
    print("\n2. Hierarchical Mode:")
    handler_hc = OutputHandler(market='germany', algorithm='hierarchical')
    print(f"   Algorithm Dir: {handler_hc.algorithm_dir}")
    print(f"   Analysis Types: {handler_hc.analysis_types}")

    # Test DBSCAN Mode
    print("\n3. DBSCAN Mode:")
    handler_db = OutputHandler(market='germany', algorithm='dbscan')
    print(f"   Algorithm Dir: {handler_db.algorithm_dir}")
    print(f"   Analysis Types: {handler_db.analysis_types}")

    # Create README
    handler_km.create_readme()

    print("\n✓ Output Handler Test erfolgreich!")
    print("="*80)
