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
from src._02_preprocessing.pca_transformer import PCATransformer
from src._03_clustering.k_selector import KSelector
from src._03_clustering.cluster_postprocessing import enforce_min_cluster_size
from src._03_clustering.robustness_tester import RobustnessTester

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

        # 0. Optional: Automatische k-Bestimmung
        k_selection_mode = config.get_value(self.config, 'cluster_selection', 'mode', default='manual')

        if k_selection_mode == 'auto' and self.algorithm == 'kmeans':
            logger.info("\n🔍 Automatische k-Bestimmung aktiviert")
            n_clusters = self._determine_optimal_k(df, features, analysis_type)
            logger.info(f"  ✓ Optimales k: {n_clusters}\n")

        logger.info(f"Cluster: {n_clusters}")

        # 1. Optional: PCA-Transformation VOR Clustering
        pca_enabled = config.get_value(self.config, 'pca', 'enabled', default=False)
        pca_metadata = None
        original_features = features.copy()

        if pca_enabled and config.get_value(self.config, 'pca', 'run_before_clustering', default=True):
            logger.info("\n🔬 PCA-Preprocessing aktiviert")
            df, features, pca_metadata = self._apply_pca_preprocessing(df, features)
            logger.info(f"  ✓ Features: {len(original_features)} → {len(features)} (PCA)")

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

        # 2.5 DBSCAN Parameter Optimization (if needed)
        if self.algorithm == 'dbscan':
            logger.info(f"\n🔍 DBSCAN Parameter Optimization...")
            logger.info(f"  Current params: eps={clusterer.eps}, min_samples={clusterer.min_samples}")

            # Import grid search function
            from src._03_clustering.algorithms.dbscan import find_optimal_dbscan_params

            # Run grid search
            best_params, grid_results = find_optimal_dbscan_params(
                X=X_scaled,
                eps_range=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
                min_samples_range=[2, 3, 5, 7, 10]
            )

            # Apply best parameters
            clusterer.eps, clusterer.min_samples = best_params
            logger.info(f"  ✓ Optimized params: eps={best_params[0]}, min_samples={best_params[1]}")

            # Show top 3 results
            logger.info(f"\n  Top 3 parameter combinations:")
            for i, result in enumerate(grid_results[:3], 1):
                logger.info(f"    {i}. eps={result['eps']}, min_samples={result['min_samples']}: "
                          f"{result['n_clusters']} clusters, {result['noise_pct']:.1f}% noise, "
                          f"silhouette={result['silhouette']:.3f}")

        # 3. Clustering durchführen
        labels = clusterer.fit_predict(X_scaled)

        # 3.5 CRITICAL: Enforce minimum cluster size constraint
        min_cluster_size = config.get_value(self.config, 'cluster_quality', 'min_cluster_size', default=10)
        min_cluster_percentage = config.get_value(self.config, 'cluster_quality', 'min_cluster_percentage', default=0.06)

        # Calculate minimum based on both absolute and percentage
        min_size_abs = max(min_cluster_size, int(min_cluster_percentage * len(valid_idx)))

        if min_size_abs > 1:
            logger.info(f"\n🔍 Enforcing minimum cluster size: {min_size_abs} companies ({min_cluster_percentage*100:.0f}%)")

            labels, merge_info = enforce_min_cluster_size(
                X=X_scaled,
                labels=labels,
                min_size=min_size_abs,
                merge_strategy='nearest'
            )

            if merge_info['n_merges'] > 0:
                logger.info(f"  ⚠️  Merged {merge_info['n_merges']} small clusters:")
                for merge_entry in merge_info['merged_clusters']:
                    logger.info(f"    Cluster {merge_entry['from_cluster']} ({merge_entry['size']} samples) → "
                              f"Cluster {merge_entry['to_cluster']}")
            else:
                logger.info(f"  ✓ All clusters meet minimum size constraint")

        # 3.6 OPTIONAL: Robustness test for K-Means
        if self.algorithm == 'kmeans' and config.get_value(self.config, 'cluster_quality', 'test_robustness', default=False):
            logger.info(f"\n🔬 Testing K-Means Stability...")

            tester = RobustnessTester(n_runs=10)
            rob_results = tester.test_kmeans_stability(
                X=X_scaled,
                n_clusters=n_clusters,
                base_seed=self.random_state
            )

            logger.info(f"  Mean ARI across 10 runs: {rob_results['mean_ari']:.3f} ± {rob_results['std_ari']:.3f}")
            logger.info(f"  Min/Max ARI: {rob_results['min_ari']:.3f} / {rob_results['max_ari']:.3f}")
            logger.info(f"  → Stability Assessment: {rob_results['stability_assessment']}")

            if rob_results['mean_ari'] < 0.8:
                logger.warning(f"  ⚠️  Low stability! Consider different k or feature selection.")

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
        # Wenn PCA aktiv: Profile für Original-Features berechnen (für Scoring/Naming)
        # Clustering verwendet PCA-Features, aber Scoring braucht Original-Features
        if pca_metadata is not None:
            # Für PCA: Verwende Original-Features für Profile
            cluster_profiles = self._compute_profiles(
                df_result.loc[valid_idx],
                labels,
                original_features,  # ✅ Original-Features statt PCA-Features!
                n_clusters,
                analysis_type
            )
        else:
            # Ohne PCA: Normal
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
        metrics['features_used'] = features
        metrics['original_features'] = original_features
        if pca_metadata:
            metrics['pca_metadata'] = pca_metadata

        return df_result, cluster_profiles, metrics


    def _determine_optimal_k(
        self,
        df: pd.DataFrame,
        features: List[str],
        analysis_type: str
    ) -> int:
        """
        Bestimmt automatisch optimale Clusterzahl mittels KSelector.

        Args:
            df: DataFrame mit Features
            features: Feature-Liste
            analysis_type: Analyse-Typ

        Returns:
            Optimale Clusterzahl
        """
        # Config auslesen
        k_range = config.get_value(self.config, 'cluster_selection', 'k_range', default=[2, 10])
        methods = config.get_value(self.config, 'cluster_selection', 'methods', default=['elbow', 'silhouette', 'gap'])

        # Daten vorbereiten (Standardisierung)
        from sklearn.preprocessing import StandardScaler

        X = df[features].copy()
        X = X.fillna(X.median())  # Handle NaN
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())  # Handle inf

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KSelector anwenden
        selector = KSelector(k_range=k_range, random_state=self.random_state)
        optimal_k, results = selector.find_optimal_k(X_scaled, methods=methods)

        return optimal_k

    def _apply_pca_preprocessing(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Wendet PCA-Transformation VOR Clustering an.

        Args:
            df: DataFrame mit Features
            features: Original Feature-Liste

        Returns:
            Tuple von (df_transformed, pca_feature_names, pca_metadata)
        """
        n_components = config.get_value(self.config, 'pca', 'n_components', default=0.90)

        # PCA Transformer initialisieren
        pca_transformer = PCATransformer(
            n_components=n_components,
            random_state=self.random_state
        )

        # PCA fit & transform
        X_pca, variance_summary = pca_transformer.fit_transform(df, features)

        # PCA-Features als DataFrame
        pca_feature_names = [f'PC{i+1}' for i in range(pca_transformer.pca.n_components_)]

        # ✅ FIX: Behalte ALLE Original-Spalten + füge PCA-Features hinzu
        # Damit kann Scoring/Naming auf Original-Features zugreifen
        # während Clustering die PCA-Features nutzt
        df_pca = df.copy()
        for i, col in enumerate(pca_feature_names):
            df_pca[col] = X_pca[:, i]

        # Metadata
        pca_metadata = {
            'n_components': pca_transformer.pca.n_components_,
            'variance_explained': pca_transformer.pca.explained_variance_ratio_.sum(),
            'variance_summary': variance_summary,
            'component_loadings': pca_transformer.get_component_loadings(features),
            'original_features': features,
            'transformer': pca_transformer
        }

        logger.info(f"  Components: {pca_metadata['n_components']}")
        logger.info(f"  Variance Explained: {pca_metadata['variance_explained']:.1%}")

        return df_pca, pca_feature_names, pca_metadata

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

    def compare_pca_vs_original(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_clusters: int,
        analysis_type: str
    ) -> Dict:
        """
        Compare clustering quality in PCA space vs. Original space

        Args:
            df: DataFrame with features
            features: List of features to use
            n_clusters: Number of clusters
            analysis_type: 'static', 'dynamic', or 'combined'

        Returns:
            Dict with comparison results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PCA VS. ORIGINAL SPACE COMPARISON ({analysis_type.upper()})")
        logger.info(f"{'='*80}\n")

        # 1. Clustering in Original Space
        logger.info("  🔷 Clustering in ORIGINAL space...")
        df_original, profiles_orig, metrics_orig = self.perform_clustering(
            df, features, n_clusters, analysis_type
        )

        silhouette_orig = metrics_orig.get('silhouette_score', -999)
        davies_bouldin_orig = metrics_orig.get('davies_bouldin_score', -999)

        logger.info(f"     Original Space: {len(features)} features")
        logger.info(f"       Silhouette: {silhouette_orig:.3f}")
        logger.info(f"       Davies-Bouldin: {davies_bouldin_orig:.3f}")

        # 2. Clustering in PCA Space
        logger.info("\n  🔶 Clustering in PCA space...")

        # Get PCA config
        n_components = config.get_value(self.config, 'pca', 'n_components', default=0.90)

        # Initialize PCA transformer
        pca_transformer = PCATransformer(n_components=n_components, random_state=self.random_state)

        # Transform to PCA space
        X_pca, pca_metadata = pca_transformer.fit_transform(df, features)

        # Create PCA feature names
        pca_features = [f'PC{i+1}' for i in range(X_pca.shape[1])]

        # Create temporary DataFrame with PCA features
        df_pca = df[['gvkey']].copy() if 'gvkey' in df.columns else pd.DataFrame()
        for i, feat in enumerate(pca_features):
            df_pca[feat] = X_pca[:, i]

        # Copy metadata columns
        metadata_cols = ['company_name', 'conm', 'revt', 'at', 'gsector']
        for col in metadata_cols:
            if col in df.columns:
                df_pca[col] = df[col].values

        # Perform clustering in PCA space
        df_pca_result, profiles_pca, metrics_pca = self.perform_clustering(
            df_pca, pca_features, n_clusters, analysis_type
        )

        silhouette_pca = metrics_pca.get('silhouette_score', -999)
        davies_bouldin_pca = metrics_pca.get('davies_bouldin_score', -999)

        variance_explained = pca_metadata['cumulative_variance'].iloc[-1] if 'cumulative_variance' in pca_metadata else 0

        logger.info(f"     PCA Space: {len(pca_features)} components ({variance_explained:.1%} variance)")
        logger.info(f"       Silhouette: {silhouette_pca:.3f}")
        logger.info(f"       Davies-Bouldin: {davies_bouldin_pca:.3f}")

        # 3. Comparison
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON RESULT")
        logger.info(f"{'='*80}")

        # Silhouette: higher is better
        sil_diff = silhouette_pca - silhouette_orig
        sil_winner = "PCA" if sil_diff > 0 else "Original"

        # Davies-Bouldin: lower is better
        db_diff = davies_bouldin_orig - davies_bouldin_pca  # Reversed!
        db_winner = "PCA" if db_diff > 0 else "Original"

        logger.info(f"\n  Silhouette Score:")
        logger.info(f"    Original: {silhouette_orig:.3f}")
        logger.info(f"    PCA:      {silhouette_pca:.3f}")
        logger.info(f"    Δ:        {sil_diff:+.3f} → {sil_winner} wins")

        logger.info(f"\n  Davies-Bouldin Index:")
        logger.info(f"    Original: {davies_bouldin_orig:.3f}")
        logger.info(f"    PCA:      {davies_bouldin_pca:.3f}")
        logger.info(f"    Δ:        {-db_diff:+.3f} → {db_winner} wins")

        # Overall recommendation
        if sil_winner == db_winner:
            recommendation = f"→ {sil_winner} space provides better separation"
        else:
            recommendation = "→ Mixed results - consider using both for validation"

        logger.info(f"\n  {recommendation}")
        logger.info(f"{'='*80}\n")

        # Return comparison results
        return {
            'original': {
                'n_features': len(features),
                'silhouette': silhouette_orig,
                'davies_bouldin': davies_bouldin_orig,
                'df': df_original,
                'profiles': profiles_orig
            },
            'pca': {
                'n_components': len(pca_features),
                'variance_explained': variance_explained,
                'silhouette': silhouette_pca,
                'davies_bouldin': davies_bouldin_pca,
                'df': df_pca_result,
                'profiles': profiles_pca,
                'metadata': pca_metadata
            },
            'comparison': {
                'silhouette_diff': sil_diff,
                'davies_bouldin_diff': db_diff,
                'silhouette_winner': sil_winner,
                'davies_bouldin_winner': db_winner,
                'recommendation': recommendation
            }
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== CLUSTERING ENGINE ===")
    print("Modul erfolgreich geladen!")
    print("\nVerwendung:")
    print("  engine = ClusteringEngine()")
    print("  df_result, profiles, metrics = engine.perform_clustering(df, features, n_clusters=5)")
