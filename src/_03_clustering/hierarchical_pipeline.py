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
from src._01_setup.output_coordinator import OutputHandler  # Migrated from output_handler
from src._05_visualization.plot_engine import create_all_plots
from src._01_setup import config_loader as config

# New imports for comprehensive scoring
from src._04_scoring import ScoreCalculator
from src._01_setup.feature_selector import FeatureSelector
from src._04_scoring.score_integrator import apply_scoring

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

        # Initialize new modules with config-driven feature toggling
        self.feature_selector = FeatureSelector()

        # Check if scoring is enabled (default: True for backwards compatibility)
        self.scoring_enabled = config.get_value(config_dict, 'scoring', 'enabled', default=True)
        if self.scoring_enabled:
            from src._04_scoring.score_evolution import ScoreEvolutionTracker
            from src._04_scoring.score_analyzer import ScoreAnalyzer
            from src._05_visualization.plot_engine_scores_wrapper import PlotEngineScores  # Unified wrapper combining distributions + analysis

            self.score_calculator = ScoreCalculator(feature_selector=self.feature_selector)
            self.score_tracker = ScoreEvolutionTracker()
            self.score_analyzer = ScoreAnalyzer()
            self.plot_engine_scores = PlotEngineScores()
            logger.info("  ✓ Scoring modules initialized")

        # Check if naming is enabled (default: True)
        self.naming_enabled = config.get_value(config_dict, 'naming', 'enabled', default=True)
        if self.naming_enabled:
            from src._03_clustering.cluster_naming import ClusterNamer
            self.cluster_namer = ClusterNamer(feature_selector=self.feature_selector)
            logger.info("  ✓ Cluster naming initialized")

        # Check if validation is enabled (default: True)
        self.validation_enabled = config.get_value(config_dict, 'validation', 'enabled', default=True)
        if self.validation_enabled:
            from src._06_validation.algorithm_comparison import AlgorithmComparison
            from src._06_validation.external_validation import ExternalValidation
            from src._05_visualization.plot_engine_validation import PlotEngineValidation

            self.algorithm_comparison = AlgorithmComparison()
            self.external_validation = ExternalValidation()
            self.plot_engine_validation = PlotEngineValidation()
            logger.info("  ✓ Validation modules initialized")

        # Check if PCA is enabled (default: False)
        self.pca_enabled = config.get_value(config_dict, 'pca', 'enabled', default=False)
        if self.pca_enabled:
            from src._05_visualization.plot_engine_pca import PlotEnginePCA
            self.plot_engine_pca = PlotEnginePCA()
            logger.info("  ✓ PCA visualization initialized")

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
        run_dynamic: bool = True,
        run_unified: bool = False
    ) -> Dict:
        """
        Run complete hierarchical analysis

        Args:
            df_all: Full dataset (all years)
            df_latest: Latest year only
            run_static: Run static analysis
            run_dynamic: Run dynamic analysis
            run_unified: Run unified analysis (all features together)

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

        # Unified mode: Run all features together
        if run_unified:
            # For unified, hierarchical mode doesn't apply (no master labels)
            logger.warning("⚠️  Unified mode with Hierarchical pipeline: using standard clustering")
            # Fall back to parent class unified analysis if it exists
            if hasattr(super(), '_run_unified_analysis'):
                df_static = self._run_static_analysis(df_latest)
                df_dynamic = self._run_dynamic_analysis(df_all)
                super()._run_unified_analysis(df_static, df_dynamic)
            else:
                logger.warning("⚠️  Unified analysis not supported in Hierarchical mode, skipping")
        else:
            # Sequential mode: Run analyses separately
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

        # Calculate Comprehensive Scores
        df_result = apply_scoring(df=df_result, features=features, cluster_column='cluster', profiles=profiles, analysis_type='static'
        , config=self.config, scoring_enabled=self.scoring_enabled)

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

        # Copy all scores from static analysis
        score_columns = [col for col in static_df.columns if 'score' in col.lower()]
        for score_col in score_columns:
            score_map = static_df.set_index('gvkey')[score_col].to_dict()
            df_result[f'static_{score_col}'] = df_result['gvkey'].map(score_map)

        # Calculate DYNAMIC Scores (based on dynamic features)
        # Berechne Profile der Dynamic-Features pro Cluster
        dynamic_profiles = df_result.groupby('cluster')[features].mean()

        # Apply comprehensive dynamic scoring
        df_result = apply_scoring(df=df_result, features=features, cluster_column='cluster', profiles=dynamic_profiles, analysis_type='dynamic'
        , config=self.config, scoring_enabled=self.scoring_enabled)

        # Enriched cluster names
        df_result = self._enrich_cluster_names_with_trends(df_result, features)

        # Save results
        all_score_cols = [col for col in df_result.columns if 'score' in col.lower()]
        profiles = df_result.groupby('cluster')[features + all_score_cols].mean()
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

        # Get features for combined analysis
        features_static = config.get_value(self.config, 'combined_analysis', 'features_static',
                                          default=['roa', 'roe', 'ebit_margin'])
        features_dynamic = config.get_value(self.config, 'combined_analysis', 'features_dynamic',
                                           default=['roa_trend', 'roa_volatility', 'roe_trend', 'revt_cagr'])

        # Merge datasets with cluster assignments and feature values
        merge_cols_static = ['gvkey', 'conm', 'cluster', 'cluster_name']
        merge_cols_static.extend([f for f in features_static if f in df_static.columns])

        merge_cols_dynamic = ['gvkey']
        merge_cols_dynamic.extend([f for f in features_dynamic if f in df_dynamic.columns])

        df_merged = df_static[df_static['gvkey'].isin(common_gvkeys)][merge_cols_static].merge(
            df_dynamic[df_dynamic['gvkey'].isin(common_gvkeys)][merge_cols_dynamic],
            on='gvkey'
        )

        # Collect all available features for combined analysis
        all_features = []
        for feat in features_static:
            if feat in df_merged.columns:
                all_features.append(feat)
        for feat in features_dynamic:
            if feat in df_merged.columns:
                all_features.append(feat)

        # Calculate profiles for combined features
        combined_profiles = df_merged.groupby('cluster')[all_features].mean()

        # Apply comprehensive combined scoring
        df_result = apply_scoring(df=df_merged, features=all_features, cluster_column='cluster', profiles=combined_profiles, analysis_type='combined'
        , config=self.config, scoring_enabled=self.scoring_enabled)

        # Enhanced cluster names based on combined score
        if 'overall_score' in df_result.columns:
            df_result = self._enhance_cluster_names_with_scores(df_result)

        # Update profiles with all scores
        all_score_cols = [col for col in df_result.columns if 'score' in col.lower()]
        profiles = df_result.groupby('cluster')[all_features + all_score_cols].mean()

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

        # Cross-analysis (score evolution) - DISABLED for now to avoid complexity
        # df_migration = self._analyze_score_evolution(df_static, df_dynamic, df_result)
        df_migration = None  # TODO: Implement if needed

        # Save score evolution data to comparisons/temporal (only if available)
        if df_migration is not None:
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
        if 'overall_score' in df_result.columns:
            logger.info(f"\n  Score Statistics:")
            logger.info(f"     Mean: {df_result['overall_score'].mean():.2f}")
            logger.info(f"     Median: {df_result['overall_score'].median():.2f}")
            logger.info(f"     Std: {df_result['overall_score'].std():.2f}")

    def _enhance_cluster_names_with_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add score information to cluster names (for combined analysis)"""
        if 'overall_score' not in df.columns:
            return df

        # Calculate average score per cluster
        cluster_scores = df.groupby('cluster')['overall_score'].mean()

        # Enhance names based on score
        enhanced_names = {}
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                enhanced_names[cluster_id] = "Noise"
                continue

            base_name = df[df['cluster'] == cluster_id]['cluster_name'].iloc[0]
            avg_score = cluster_scores[cluster_id]

            if avg_score >= 70:
                enhanced_names[cluster_id] = f"{base_name} (High Score: {avg_score:.0f})"
            elif avg_score >= 50:
                enhanced_names[cluster_id] = f"{base_name} (Mid Score: {avg_score:.0f})"
            else:
                enhanced_names[cluster_id] = f"{base_name} (Low Score: {avg_score:.0f})"

        df['cluster_name_enriched'] = df['cluster'].map(enhanced_names)
        return df

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