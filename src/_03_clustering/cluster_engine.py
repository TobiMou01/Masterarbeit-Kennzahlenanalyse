"""
Clustering Engine
Vereinheitlichte 3-stufige Clustering-Analyse mit flexiblen Algorithmen
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

from src._01_setup import config_loader as config
from src._03_clustering.algorithms.factory import ClustererFactory
from src._03_clustering.algorithms.base import BaseClusterer

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Haupt-Engine für alle Clustering-Analysen"""

    def __init__(self, config_dict: dict = None, config_path: str = None):
        """
        Initialisiert Clustering Engine

        Args:
            config_dict: Config dictionary (preferred)
            config_path: Path to config file (legacy support)
        """
        if config_dict is None and config_path is not None:
            # Legacy: load from file
            config_dict = config.load_config(config_path)
        elif config_dict is None:
            # Default: load from default path
            config_dict = config.load_config('config.yaml')

        self.config = config_dict
        self.random_state = config.get_value(config_dict, 'global', 'random_state', default=42)

        # Clustering-Algorithmus aus Config laden
        self.algorithm = config.get_value(config_dict, 'classification', 'algorithm', default='kmeans')
        logger.info(f"Clustering-Algorithmus: {self.algorithm}")

    # =========================================================================
    # CORE CLUSTERING METHODS
    # =========================================================================

    def perform_clustering(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_clusters: int,
        analysis_type: str = 'static'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Führt Clustering durch (für alle Analyse-Typen)

        Args:
            df: DataFrame mit Features
            features: Liste der zu verwendenden Features
            n_clusters: Anzahl Cluster
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            (df mit Clustern, cluster_profiles, metrics)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"{analysis_type.upper()} CLUSTERING")
        logger.info(f"{'='*80}\n")
        logger.info(f"Features: {len(features)}")
        logger.info(f"Cluster: {n_clusters}")

        # 1. Clusterer erstellen mit Factory Pattern
        algo_config = config.get_value(self.config, 'classification', self.algorithm, default={})
        algo_config['n_clusters'] = n_clusters  # n_clusters aus Analyse-Config überschreiben

        clusterer = ClustererFactory.create(
            algorithm=self.algorithm,
            config=algo_config,
            random_state=self.random_state
        )

        # 2. Daten vorbereiten und skalieren
        X_scaled, valid_idx = clusterer.preprocess_data(df, features)

        # 3. Clustering durchführen
        labels = clusterer.fit_predict(X_scaled)

        # 4. Metriken berechnen
        metrics = clusterer.get_metrics(X_scaled, labels)
        metrics['n_companies'] = len(valid_idx)
        metrics['algorithm'] = clusterer.get_algorithm_name()

        logger.info(f"\n✓ Clustering abgeschlossen")
        logger.info(f"  Silhouette Score: {metrics['silhouette']:.3f}")
        logger.info(f"  Davies-Bouldin: {metrics['davies_bouldin']:.3f}")

        # 5. Labels hinzufügen
        df_result = df.copy()
        df_result['cluster'] = -1
        df_result.loc[valid_idx, 'cluster'] = labels

        # 6. Cluster-Profile
        cluster_profiles = self._compute_profiles(
            df_result.loc[valid_idx],
            labels,
            features,
            n_clusters,
            analysis_type
        )

        # 7. Namen hinzufügen
        cluster_names = self._generate_cluster_names(
            cluster_profiles,
            n_clusters,
            analysis_type
        )
        cluster_profiles['cluster_name'] = cluster_profiles.index.map(cluster_names)
        df_result['cluster_name'] = df_result['cluster'].map(cluster_names)

        # 8. Statistik
        self._print_distribution(df_result)

        # Scaler & Model zu metrics
        metrics['scaler'] = clusterer.get_scaler()
        metrics['model'] = clusterer.get_model()

        return df_result, cluster_profiles, metrics


    def _compute_profiles(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        features: List[str],
        n_clusters: int,
        analysis_type: str
    ) -> pd.DataFrame:
        """Berechnet Cluster-Profile (Mittelwerte + Std)"""

        df_temp = df[features].copy()
        df_temp['cluster'] = labels

        profiles = df_temp.groupby('cluster')[features].agg(['mean', 'std'])

        # Flatten MultiIndex columns
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]

        # Nur mean-Spalten für Benennung
        mean_cols = [col for col in profiles.columns if col.endswith('_mean')]
        profiles_mean = profiles[mean_cols].copy()
        profiles_mean.columns = [col.replace('_mean', '') for col in mean_cols]

        return profiles_mean

    def _generate_cluster_names(
        self,
        profiles: pd.DataFrame,
        n_clusters: int,
        analysis_type: str
    ) -> Dict[int, str]:
        """Generiert automatisch Cluster-Namen basierend auf n_clusters"""

        # Performance-Score berechnen (ROA + ROE + EBIT wenn vorhanden)
        score = pd.Series(0, index=profiles.index)
        for metric in ['roa', 'roe', 'ebit_margin']:
            if metric in profiles.columns:
                score += profiles[metric]

        # Sortiere Cluster nach Score
        ranking = score.sort_values(ascending=False).index

        # Namen basierend auf n_clusters
        if analysis_type == 'dynamic':
            # Für dynamische Analyse: Trend-basiert
            return self._name_dynamic_clusters(profiles, ranking, n_clusters)

        # Für static/combined: Performance-basiert
        if n_clusters == 3:
            names = ['High Performers', 'Mid Performers', 'Low Performers']
        elif n_clusters == 4:
            names = ['High Performers', 'Upper-Mid', 'Lower-Mid', 'Low Performers']
        elif n_clusters == 5:
            names = ['High Performers', 'Upper-Mid', 'Mid', 'Lower-Mid', 'Low Performers']
        elif n_clusters == 6:
            names = ['Top Tier', 'Strong Performers', 'Upper-Mid', 'Mid', 'Lower-Mid', 'Challenged']
        elif n_clusters == 7:
            names = ['Excellence', 'High Performers', 'Upper-Mid', 'Solid Mid', 'Lower-Mid', 'Weak', 'Distressed']
        else:
            # Für 8+ Cluster: Tier-System
            names = [f'Tier {i+1}' for i in range(n_clusters)]

        return {cluster_id: names[i] for i, cluster_id in enumerate(ranking)}

    def _name_dynamic_clusters(
        self,
        profiles: pd.DataFrame,
        ranking: pd.Index,
        n_clusters: int
    ) -> Dict[int, str]:
        """Spezielle Benennung für dynamische Cluster"""

        names = {}
        for cluster_id in profiles.index:
            # Durchschnittlicher Trend
            trend_cols = [c for c in profiles.columns if '_trend' in c]
            vol_cols = [c for c in profiles.columns if '_volatility' in c]

            avg_trend = profiles.loc[cluster_id, trend_cols].mean() if trend_cols else 0
            avg_vol = profiles.loc[cluster_id, vol_cols].mean() if vol_cols else 0

            # Klassifizierung
            if avg_trend > 0.5 and avg_vol < 2.0:
                name = "Steady Growers"
            elif avg_trend > 0.3:
                name = "Growing"
            elif avg_trend < -0.3 and avg_vol > 3.0:
                name = "Declining (Volatile)"
            elif avg_trend < -0.3:
                name = "Declining"
            elif avg_vol > 4.0:
                name = "Highly Volatile"
            else:
                name = "Stable"

            names[cluster_id] = name

        return names

    def _print_distribution(self, df: pd.DataFrame):
        """Gibt Cluster-Verteilung aus"""

        valid_df = df[df['cluster'] >= 0]
        total = len(valid_df)

        logger.info(f"\n  Cluster-Verteilung:")
        for cluster_id in sorted(valid_df['cluster'].unique()):
            count = (valid_df['cluster'] == cluster_id).sum()
            name = valid_df[valid_df['cluster'] == cluster_id]['cluster_name'].iloc[0]
            pct = count / total * 100
            logger.info(f"    {cluster_id}: {name:25} {count:4} ({pct:5.1f}%)")

    # =========================================================================
    # TIMESERIES FEATURES
    # =========================================================================

    def compute_timeseries_features(
        self,
        df: pd.DataFrame,
        min_years: int = 5
    ) -> pd.DataFrame:
        """
        Berechnet Zeitreihen-Features (Trend, Volatilität, CAGR)

        Args:
            df: DataFrame mit allen Jahren
            min_years: Minimum Jahre erforderlich

        Returns:
            DataFrame mit Zeitreihen-Features
        """
        logger.info(f"\nBerechne Zeitreihen-Features (min {min_years} Jahre)...")

        metrics = config.get_value(self.config, 'dynamic_analysis', 'features', default=['roa', 'roe', 'ebit_margin', 'revt'])

        # Filter: Nur Unternehmen mit >= min_years
        company_years = df.groupby('gvkey')['fyear'].count()
        valid_companies = company_years[company_years >= min_years].index
        df_filtered = df[df['gvkey'].isin(valid_companies)]

        logger.info(f"  Unternehmen mit >={min_years} Jahren: {len(valid_companies)}")

        # Features pro Unternehmen berechnen
        results = []
        for gvkey, group in df_filtered.groupby('gvkey'):
            group = group.sort_values('fyear')
            years = group['fyear'].values

            features = {
                'gvkey': gvkey,
                'company_name': group['conm'].iloc[0] if 'conm' in group.columns else '',
                'years_available': len(group)
            }

            for metric in metrics:
                if metric not in group.columns:
                    continue

                values = group[metric].values
                mask = ~np.isnan(values)

                if mask.sum() < 2:
                    continue

                clean_years = years[mask]
                clean_values = values[mask]

                # Trend (lineare Regression)
                slope, _, r_value, _, _ = stats.linregress(clean_years, clean_values)
                features[f'{metric}_trend'] = slope
                features[f'{metric}_trend_r2'] = r_value ** 2

                # Volatilität
                features[f'{metric}_volatility'] = np.std(clean_values)

                # CAGR (nur für Revenue/Assets)
                if metric.lower() in ['revt', 'revenue', 'at', 'assets']:
                    if clean_values[0] > 0 and clean_values[-1] > 0:
                        num_years = clean_years[-1] - clean_years[0]
                        if num_years > 0:
                            cagr = (np.power(clean_values[-1] / clean_values[0], 1 / num_years) - 1) * 100
                            features[f'{metric}_cagr'] = cagr

            results.append(features)

        df_timeseries = pd.DataFrame(results)
        logger.info(f"✓ Zeitreihen-Features für {len(df_timeseries)} Unternehmen berechnet")

        return df_timeseries

    def assign_clusters_to_timeseries(
        self,
        df_all: pd.DataFrame,
        df_clustered: pd.DataFrame,
        company_id_col: str = 'gvkey'
    ) -> pd.DataFrame:
        """
        Ordnet Cluster-Labels allen Zeitreihen-Daten zu

        Args:
            df_all: Original DataFrame mit allen Jahren
            df_clustered: DataFrame mit Cluster-Zuordnungen (aggregiert)
            company_id_col: Company ID Spalte

        Returns:
            DataFrame mit allen Jahren + Cluster-Spalten
        """
        # Merge cluster info (gvkey -> cluster, cluster_name)
        cluster_info = df_clustered[[company_id_col, 'cluster', 'cluster_name']].copy()

        # Join auf Original-Daten
        df_timeseries = df_all.merge(
            cluster_info,
            on=company_id_col,
            how='left'
        )

        # Fillna für Unternehmen ohne Cluster
        df_timeseries['cluster'].fillna(-1, inplace=True)
        df_timeseries['cluster'] = df_timeseries['cluster'].astype(int)

        return df_timeseries

    # =========================================================================
    # CROSS ANALYSIS
    # =========================================================================

    def analyze_migration(
        self,
        df_static: pd.DataFrame,
        df_dynamic: pd.DataFrame,
        df_combined: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analysiert Cluster-Migration zwischen Analysen

        Returns:
            DataFrame mit Migration-Pattern
        """
        logger.info(f"\n{'='*80}")
        logger.info("CROSS ANALYSIS - Cluster Migration")
        logger.info(f"{'='*80}\n")

        # Merge
        df_migration = df_combined[['gvkey', 'cluster', 'cluster_name']].copy()
        df_migration = df_migration.rename(columns={
            'cluster': 'combined_cluster',
            'cluster_name': 'combined_name'
        })

        # Static hinzufügen
        if 'cluster' in df_static.columns:
            static_map = df_static.set_index('gvkey')[['cluster', 'cluster_name']].rename(
                columns={'cluster': 'static_cluster', 'cluster_name': 'static_name'}
            )
            df_migration = df_migration.merge(static_map, on='gvkey', how='left')

        # Dynamic hinzufügen
        if 'cluster' in df_dynamic.columns:
            dynamic_map = df_dynamic.set_index('gvkey')[['cluster', 'cluster_name']].rename(
                columns={'cluster': 'dynamic_cluster', 'cluster_name': 'dynamic_name'}
            )
            df_migration = df_migration.merge(dynamic_map, on='gvkey', how='left')

        # Company Name
        for df in [df_combined, df_static, df_dynamic]:
            if 'company_name' in df.columns:
                name_map = df.set_index('gvkey')['company_name']
                if 'company_name' not in df_migration.columns:
                    df_migration['company_name'] = df_migration['gvkey'].map(name_map)
                break

        # Migration Pattern
        df_migration['pattern'] = df_migration.apply(self._classify_pattern, axis=1)
        df_migration['flag'] = df_migration['pattern'].map({
            'Consistent': '✓',
            'Improving': '⭐',
            'Declining': '⚠',
            'Critical': '🔥',
            'Volatile': '⚠',
            'Unknown': '?'
        })

        # Statistik
        logger.info("Migration Patterns:")
        for pattern, count in df_migration['pattern'].value_counts().items():
            pct = count / len(df_migration) * 100
            flag = df_migration[df_migration['pattern'] == pattern]['flag'].iloc[0]
            logger.info(f"  {flag} {pattern:15} {count:4} ({pct:5.1f}%)")

        return df_migration

    def _classify_pattern(self, row: pd.Series) -> str:
        """Klassifiziert Migration-Pattern"""

        static = row.get('static_cluster', -1)
        dynamic = row.get('dynamic_cluster', -1)
        combined = row.get('combined_cluster', -1)

        if static == -1 or dynamic == -1 or combined == -1:
            return 'Unknown'

        # Einfache Heuristik: Niedrige Cluster-ID = besser
        avg_cluster = (static + dynamic + combined) / 3
        variance = np.var([static, dynamic, combined])

        if variance < 0.5 and avg_cluster < 1.5:
            return 'Consistent'
        elif dynamic < static:
            return 'Improving'
        elif dynamic > static + 1:
            return 'Declining'
        elif variance > 2:
            return 'Volatile'
        elif avg_cluster > 3:
            return 'Critical'
        else:
            return 'Consistent'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== CLUSTERING ENGINE ===")
    print("Modul erfolgreich geladen!")
    print("\nVerwendung:")
    print("  engine = ClusteringEngine()")
    print("  df_result, profiles, metrics = engine.perform_clustering(df, features, n_clusters=5)")
