"""
Hierarchical Pipeline - Alternative 1
Static-Labels bleiben konsistent, Dynamic/Combined reichern an
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src._03_clustering.cluster_engine import ClusteringEngine
from src._01_setup.output_handler import OutputHandler
from src._05_visualization.plot_engine import create_all_plots
from src._01_setup import config_loader as config

logger = logging.getLogger(__name__)


class HierarchicalPipeline:
    """
    Pipeline für hierarchisches Vorgehen (Alternative 1)

    1. Static: Master-Clustering → Labels [0, 1, 2, 3, 4]
    2. Dynamic: Keine neuen Labels, nur Dynamic-Features + Scores berechnen
    3. Combined: Gleiche Labels, Combined-Score aus Static + Dynamic
    """

    def __init__(self, config_dict: dict, market: str, skip_plots: bool = False):
        """
        Initialize hierarchical pipeline

        Args:
            config_dict: Configuration dictionary
            market: Market name
            skip_plots: Skip visualization generation
        """
        self.config = config_dict
        self.market = market
        self.skip_plots = skip_plots

        # Get algorithm from config
        self.algorithm = config.get_value(config_dict, 'classification', 'algorithm', default='kmeans')

        # Initialize engine and output handler
        self.engine = ClusteringEngine(config_dict=config_dict)
        self.output = OutputHandler(market=market, algorithm=self.algorithm)

        # Results storage
        self.results = {}
        self.start_time = datetime.now()

        # Master labels from Static (will be reused)
        self.master_labels = None
        self.master_cluster_names = None

    def run_analysis(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame,
        run_static: bool = True,
        run_dynamic: bool = True
    ) -> Dict:
        """
        Run complete hierarchical analysis

        Args:
            df_all: Full dataset (all years)
            df_latest: Latest year only
            run_static: Run static analysis
            run_dynamic: Run dynamic analysis

        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 HIERARCHICAL CLUSTERING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Market: {self.market}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Mode: Hierarchical (Static-Labels bleiben konsistent)\n")

        # Storage for results
        df_static = None
        df_dynamic = None

        # Run analyses
        if run_static:
            df_static = self._run_static_analysis(df_latest)

        if run_dynamic and self.master_labels is not None:
            df_dynamic = self._run_dynamic_analysis(df_all)

        if run_static and run_dynamic and self.master_labels is not None:
            self._run_combined_analysis(df_static, df_dynamic)

        # Print summary
        self._print_summary()

        return self.results

    def _run_static_analysis(self, df_latest: pd.DataFrame) -> pd.DataFrame:
        """Run static analysis - MASTER CLUSTERING"""
        logger.info("\n" + "=" * 80)
        logger.info("STATIC ANALYSIS (Master-Clustering)")
        logger.info("=" * 80 + "\n")

        # Get config - using new feature selection logic
        features = config.get_features_for_analysis(
            self.config, 'static_analysis',
            default_features=['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'current_ratio']
        )
        n_clusters = config.get_value(self.config, 'static_analysis', 'n_clusters', default=5)

        logger.info(f"📊 Selected {len(features)} features for static analysis")

        # Run clustering
        df_result, profiles, metrics = self.engine.perform_clustering(
            df_latest, features, n_clusters, 'static'
        )

        # Save MASTER labels
        self.master_labels = df_result['cluster'].copy()
        self.master_cluster_names = df_result['cluster_name'].copy()

        # Calculate Static Scores (distance to cluster center)
        df_result['static_score'] = self._calculate_scores(df_result, features, profiles)

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, features, 'static', sort_by='roa')

        # Store
        self.results['static'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'profiles': profiles,
            'df': df_result
        }

        logger.info(f"\n✓ Master-Labels gespeichert: {n_clusters} Cluster")

        return df_result

    def _run_dynamic_analysis(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Run dynamic analysis - ANREICHERUNG (kein neues Clustering!)"""
        logger.info("\n" + "=" * 80)
        logger.info("DYNAMIC ANALYSIS (Anreicherung, keine neuen Labels)")
        logger.info("=" * 80 + "\n")

        # Get config
        min_years = config.get_value(self.config, 'dynamic_analysis', 'min_years_required', default=5)

        # Compute timeseries features
        df_timeseries = self.engine.compute_timeseries_features(df_all, min_years=min_years)

        # Auto-detect dynamic features
        features = [col for col in df_timeseries.columns
                    if '_trend' in col or '_volatility' in col or '_cagr' in col]

        # WICHTIG: Keine neues Clustering, sondern Master-Labels zuweisen!
        logger.info("\n→ Verwende Master-Labels von Static Analysis")

        # Merge Master-Labels mit Dynamic-Features
        df_result = df_timeseries.copy()

        # Find companies that exist in both Static and Dynamic
        static_df = self.results['static']['df']
        common_gvkeys = set(df_result['gvkey']).intersection(set(static_df['gvkey']))

        logger.info(f"→ {len(common_gvkeys)} Unternehmen haben Static + Dynamic Daten")

        # Add master labels
        df_result = df_result[df_result['gvkey'].isin(common_gvkeys)].copy()
        label_map = static_df.set_index('gvkey')['cluster'].to_dict()
        name_map = static_df.set_index('gvkey')['cluster_name'].to_dict()

        df_result['cluster'] = df_result['gvkey'].map(label_map)
        df_result['cluster_name'] = df_result['gvkey'].map(name_map)

        # Calculate DYNAMIC Scores (based on dynamic features)
        # Berechne Profile der Dynamic-Features pro Cluster
        dynamic_profiles = df_result.groupby('cluster')[features].mean()
        df_result['dynamic_score'] = self._calculate_scores(df_result, features, dynamic_profiles)

        # Enriched cluster names
        df_result = self._enrich_cluster_names_with_trends(df_result, features)

        # Save results
        profiles = df_result.groupby('cluster')[features + ['static_score', 'dynamic_score']].mean()
        metrics = {
            'n_companies': len(df_result),
            'n_clusters': len(df_result['cluster'].unique()),
            'silhouette_score': None,  # N/A für hierarchical mode
            'davies_bouldin_score': None
        }

        self._save_analysis_results(df_result, profiles, metrics, features, 'dynamic', sort_by='dynamic_score')

        # Assign clusters to ALL timeseries data
        df_all_with_clusters = self.engine.assign_clusters_to_timeseries(
            df_all=df_all,
            df_clustered=df_result
        )

        # Store
        self.results['dynamic'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': metrics['n_clusters'],
            'metrics': metrics,
            'profiles': profiles,
            'df': df_result,
            'df_timeseries': df_all_with_clusters
        }

        return df_result

    def _run_combined_analysis(self, df_static: pd.DataFrame, df_dynamic: pd.DataFrame):
        """Run combined analysis - COMBINED SCORE"""
        logger.info("\n" + "=" * 80)
        logger.info("COMBINED ANALYSIS (Combined-Score aus Static + Dynamic)")
        logger.info("=" * 80 + "\n")

        # Find common companies
        common_gvkeys = set(df_static['gvkey']).intersection(set(df_dynamic['gvkey']))
        logger.info(f"  Common companies: {len(common_gvkeys)}")

        # Get weights from config
        weights = config.get_value(self.config, 'combined_analysis', 'weights',
                                   default={'static': 0.4, 'dynamic': 0.6})

        logger.info(f"  Weights: Static {weights['static']}, Dynamic {weights['dynamic']}")

        # Merge datasets
        df_merged = df_static[df_static['gvkey'].isin(common_gvkeys)][
            ['gvkey', 'conm', 'cluster', 'cluster_name', 'static_score']
        ].merge(
            df_dynamic[df_dynamic['gvkey'].isin(common_gvkeys)][
                ['gvkey', 'dynamic_score']
            ],
            on='gvkey'
        )

        # Calculate COMBINED SCORE
        df_merged['combined_score'] = (
            weights['static'] * df_merged['static_score'] +
            weights['dynamic'] * df_merged['dynamic_score']
        )

        # Enhanced cluster names based on combined score
        df_result = self._enhance_cluster_names_with_scores(df_merged)

        # Profiles
        all_features = []
        features_static = config.get_value(self.config, 'combined_analysis', 'features_static',
                                          default=['roa', 'roe', 'ebit_margin'])
        features_dynamic = config.get_value(self.config, 'combined_analysis', 'features_dynamic',
                                           default=['roa_trend', 'roa_volatility', 'roe_trend', 'revt_cagr'])

        # Get actual feature values
        for feat in features_static:
            if feat in df_static.columns:
                df_result[feat] = df_result['gvkey'].map(df_static.set_index('gvkey')[feat])
                all_features.append(feat)

        for feat in features_dynamic:
            if feat in df_dynamic.columns:
                df_result[feat] = df_result['gvkey'].map(df_dynamic.set_index('gvkey')[feat])
                all_features.append(feat)

        profiles = df_result.groupby('cluster')[all_features + ['static_score', 'dynamic_score', 'combined_score']].mean()

        # Metrics
        n_clusters = len(df_result['cluster'].unique())
        metrics = {
            'n_companies': len(df_result),
            'n_clusters': n_clusters,
            'weights': weights,
            'silhouette_score': None,  # N/A für hierarchical mode
            'davies_bouldin_score': None
        }

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, all_features, 'combined', sort_by='combined_score')

        # Cross-analysis (score evolution)
        df_migration = self._analyze_score_evolution(df_static, df_dynamic, df_result)

        # Save score evolution data to comparisons/temporal
        self.output.save_comparison_data(
            comp_type='temporal',
            data=df_migration,
            filename='score_evolution.csv'
        )

        # Store
        self.results['combined'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'weights': weights,
            'profiles': profiles,
            'df': df_result
        }

        # Print score statistics
        self._print_score_statistics(df_result)

    def _calculate_scores(self, df: pd.DataFrame, features: list, profiles: pd.DataFrame) -> pd.Series:
        """
        Calculate scores (0-100) based on distance to cluster center

        Args:
            df: DataFrame with cluster assignments
            features: List of features to consider
            profiles: Cluster profiles (centers)

        Returns:
            Series with scores (0-100)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import euclidean_distances

        # Prepare features
        X = df[features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate distance to own cluster center
        scores = []
        for i, (idx, row) in enumerate(df.iterrows()):
            cluster_id = row['cluster']
            if cluster_id == -1:  # Noise in DBSCAN
                scores.append(0)
                continue

            # Point coordinates (use enumerate index, not dataframe index)
            point = X_scaled[i].reshape(1, -1)

            # Cluster center
            center_features = profiles.loc[cluster_id, features].fillna(0).values
            center_scaled = scaler.transform(center_features.reshape(1, -1))

            # Euclidean distance
            dist = euclidean_distances(point, center_scaled)[0][0]

            # Convert distance to score (closer = higher score)
            # Use exponential decay: score = 100 * exp(-dist)
            score = 100 * np.exp(-dist / 2)  # Divide by 2 to make scores more spread out
            scores.append(min(100, max(0, score)))  # Clamp to [0, 100]

        return pd.Series(scores, index=df.index)

    def _enrich_cluster_names_with_trends(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Add trend information to cluster names"""
        # Calculate average trend per cluster
        trend_cols = [f for f in features if '_trend' in f]
        if not trend_cols:
            return df

        cluster_trends = df.groupby('cluster')[trend_cols].mean().mean(axis=1)

        # Enhance names
        enhanced_names = {}
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                enhanced_names[cluster_id] = "Noise"
                continue

            base_name = df[df['cluster'] == cluster_id]['cluster_name'].iloc[0]
            avg_trend = cluster_trends[cluster_id]

            if avg_trend > 1.0:
                enhanced_names[cluster_id] = f"{base_name} (Strongly Growing)"
            elif avg_trend > 0.3:
                enhanced_names[cluster_id] = f"{base_name} (Growing)"
            elif avg_trend < -0.5:
                enhanced_names[cluster_id] = f"{base_name} (Declining)"
            else:
                enhanced_names[cluster_id] = f"{base_name} (Stable)"

        df['cluster_name_enriched'] = df['cluster'].map(enhanced_names)
        return df

    def _enhance_cluster_names_with_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance cluster names based on combined scores"""
        # Calculate average combined score per cluster
        cluster_scores = df.groupby('cluster')['combined_score'].mean()

        enhanced_names = {}
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                enhanced_names[cluster_id] = "Noise"
                continue

            base_name = df[df['cluster'] == cluster_id]['cluster_name'].iloc[0]
            avg_score = cluster_scores[cluster_id]

            if avg_score >= 80:
                tier = "Champions"
            elif avg_score >= 65:
                tier = "Strong"
            elif avg_score >= 50:
                tier = "Solid"
            else:
                tier = "Challenged"

            enhanced_names[cluster_id] = f"{base_name} ({tier})"

        df['cluster_name_enhanced'] = df['cluster'].map(enhanced_names)
        return df

    def _analyze_score_evolution(self, df_static: pd.DataFrame, df_dynamic: pd.DataFrame,
                                 df_combined: pd.DataFrame) -> pd.DataFrame:
        """Analyze how scores evolved from Static → Dynamic → Combined"""
        migration_data = []

        for _, row in df_combined.iterrows():
            gvkey = row['gvkey']
            cluster = row['cluster']

            static_score = row['static_score']
            dynamic_score = row['dynamic_score']
            combined_score = row['combined_score']

            # Classify pattern
            if combined_score >= 70:
                pattern = "Strong Overall"
            elif static_score >= 60 and dynamic_score >= 60:
                pattern = "Balanced"
            elif static_score >= 70 > dynamic_score:
                pattern = "Strong Now, Weak Trend"
            elif dynamic_score >= 70 > static_score:
                pattern = "Weak Now, Strong Trend"
            else:
                pattern = "Challenged"

            migration_data.append({
                'gvkey': gvkey,
                'cluster': cluster,
                'cluster_name': row['cluster_name'],
                'static_score': static_score,
                'dynamic_score': dynamic_score,
                'combined_score': combined_score,
                'pattern': pattern
            })

        return pd.DataFrame(migration_data)

    def _print_score_statistics(self, df: pd.DataFrame):
        """Print score statistics per cluster"""
        logger.info("\n📊 Score Statistics per Cluster:")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            name = cluster_df['cluster_name'].iloc[0]

            avg_static = cluster_df['static_score'].mean()
            avg_dynamic = cluster_df['dynamic_score'].mean()
            avg_combined = cluster_df['combined_score'].mean()

            logger.info(f"  Cluster {cluster_id} ({name}):")
            logger.info(f"    Static: {avg_static:.1f}, Dynamic: {avg_dynamic:.1f}, Combined: {avg_combined:.1f}")

    def _save_analysis_results(self, df: pd.DataFrame, profiles: pd.DataFrame,
                               metrics: Dict, features: list, analysis_type: str, sort_by: str):
        """Save analysis results"""
        # Use the same save methods as ClusteringPipeline
        self.output.save_cluster_data(df, profiles, analysis_type, metrics)
        self.output.save_cluster_lists(df, analysis_type, sort_by=sort_by)
        # Note: No models saved for hierarchical mode (scaler/model are None)

        if not self.skip_plots:
            plots_dir = self.output.get_plots_dir(analysis_type)
            create_all_plots(df, profiles, features, analysis_type, plots_dir)

    def _print_summary(self):
        """Print pipeline summary"""
        duration = (datetime.now() - self.start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("HIERARCHICAL PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\n  ⏱️  Duration: {duration:.1f}s")
        logger.info(f"  📁 Output: {self.output.algorithm_dir}\n")

        # Create README in summary directory
        self.output.create_readme()

        print("\n" + "=" * 80)
        print(f"✓ Hierarchical Analysis complete for market: {self.market}")
        print(f"✓ Algorithm: {self.algorithm} ({self.output.mode} mode)")
        print(f"✓ Duration: {duration:.1f}s")
        print(f"\n📂 Output: {self.output.algorithm_dir}/")
        for analysis_key in ['static', 'dynamic', 'combined']:
            analysis_name = self.output.analysis_types[analysis_key]
            print(f"   ├── {analysis_name}/")
        print("\n📌 Next Steps:")
        print(f"   1. Review summary: {self.output.summary_dir}/")
        print(f"   2. Check clusters: {self.output.algorithm_dir}/{self.output.analysis_types['static']}/reports/clusters/")
        print(f"   3. Check score evolution: {self.output.comparisons_dir}/temporal/data/")
        print("=" * 80 + "\n")
