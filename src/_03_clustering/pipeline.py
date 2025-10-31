"""
Clustering Pipeline
Main orchestration logic for 3-stage clustering analysis
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src._03_clustering.cluster_engine import ClusteringEngine
from src._01_setup.output_handler import OutputHandler
from src._05_visualization.plot_engine import create_all_plots
from src._01_setup import config_loader as config

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """
    Main pipeline for clustering analysis

    Orchestrates the entire 3-stage analysis:
    1. Static Analysis (current state)
    2. Dynamic Analysis (trends over time)
    3. Combined Analysis (integrated view)
    """

    def __init__(self, config_dict: dict, market: str, skip_plots: bool = False):
        """
        Initialize pipeline

        Args:
            config_dict: Configuration dictionary
            market: Market name (e.g., 'germany')
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

    def run_analysis(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame,
        run_static: bool = True,
        run_dynamic: bool = True
    ) -> Dict:
        """
        Run complete clustering analysis

        Args:
            df_all: Full dataset (all years)
            df_latest: Latest year only
            run_static: Run static analysis
            run_dynamic: Run dynamic analysis

        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 CLUSTERING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Market: {self.market}")
        logger.info(f"Algorithm: {self.algorithm}\n")

        # Storage for results
        df_static = None
        df_dynamic = None

        # Run analyses
        if run_static:
            df_static = self._run_static_analysis(df_latest)

        if run_dynamic:
            df_dynamic = self._run_dynamic_analysis(df_all)

        if run_static and run_dynamic:
            self._run_combined_analysis(df_static, df_dynamic)

        # Print summary
        self._print_summary()

        return self.results

    def _run_static_analysis(self, df_latest: pd.DataFrame) -> pd.DataFrame:
        """Run static analysis (current state)"""
        logger.info("\n" + "=" * 80)
        logger.info("STATIC ANALYSIS")
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

        return df_result

    def _run_dynamic_analysis(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Run dynamic analysis (trends over time)"""
        logger.info("\n" + "=" * 80)
        logger.info("DYNAMIC ANALYSIS")
        logger.info("=" * 80 + "\n")

        # Get config
        min_years = config.get_value(self.config, 'dynamic_analysis', 'min_years_required', default=5)
        n_clusters = config.get_value(self.config, 'dynamic_analysis', 'n_clusters', default=5)

        # Compute timeseries features
        df_timeseries = self.engine.compute_timeseries_features(df_all, min_years=min_years)

        # Auto-detect dynamic features
        features = [col for col in df_timeseries.columns
                    if '_trend' in col or '_volatility' in col or '_cagr' in col]

        # Run clustering
        df_result, profiles, metrics = self.engine.perform_clustering(
            df_timeseries, features, n_clusters, 'dynamic'
        )

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, features, 'dynamic', sort_by='roa_trend')

        # Assign clusters to ALL timeseries data (for temporal stability)
        df_all_with_clusters = self.engine.assign_clusters_to_timeseries(
            df_all=df_all,
            df_clustered=df_result
        )

        # Store
        self.results['dynamic'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'profiles': profiles,
            'df': df_result,
            'df_timeseries': df_all_with_clusters  # ← NEU: Für Temporal Stability
        }

        return df_result

    def _run_combined_analysis(self, df_static: pd.DataFrame, df_dynamic: pd.DataFrame):
        """Run combined analysis (static + dynamic)"""
        logger.info("\n" + "=" * 80)
        logger.info("COMBINED ANALYSIS")
        logger.info("=" * 80 + "\n")

        # Find common companies
        common_gvkeys = set(df_static['gvkey']).intersection(set(df_dynamic['gvkey']))
        logger.info(f"  Common companies: {len(common_gvkeys)}")

        # Get config
        features_static = config.get_value(
            self.config, 'combined_analysis', 'features_static',
            default=['roa', 'roe', 'ebit_margin']
        )
        features_dynamic = config.get_value(
            self.config, 'combined_analysis', 'features_dynamic',
            default=['roa_trend', 'roa_volatility', 'roe_trend', 'revt_cagr']
        )
        features_dynamic = [f for f in features_dynamic if f in df_dynamic.columns]
        n_clusters = config.get_value(self.config, 'combined_analysis', 'n_clusters', default=6)

        # Merge datasets
        cols_static = ['gvkey'] + features_static
        # Include company name if available
        if 'company_name' in df_static.columns:
            cols_static.append('company_name')
        elif 'conm' in df_static.columns:
            cols_static.append('conm')

        df_static_sub = df_static[df_static['gvkey'].isin(common_gvkeys)][cols_static]
        df_dynamic_sub = df_dynamic[df_dynamic['gvkey'].isin(common_gvkeys)][['gvkey'] + features_dynamic]
        df_merged = df_static_sub.merge(df_dynamic_sub, on='gvkey')

        # Run clustering
        features_combined = features_static + features_dynamic
        df_result, profiles, metrics = self.engine.perform_clustering(
            df_merged, features_combined, n_clusters, 'combined'
        )

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, features_combined, 'combined', sort_by='roa')

        # Cross-analysis (migration patterns)
        df_migration = self.engine.analyze_migration(df_static, df_dynamic, df_result)
        self.output.save_analysis_results(
            df_static=df_static,
            df_dynamic=df_dynamic,
            df_combined=df_result,
            df_migration=df_migration
        )

        # Store
        weights = config.get_value(self.config, 'combined_analysis', 'weights', default={'static': 0.4, 'dynamic': 0.6})
        self.results['combined'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'weights': weights,
            'profiles': profiles,
            'df': df_result
        }
        self.results['migration'] = {
            'total': len(df_migration),
            'patterns': df_migration['pattern'].value_counts().to_dict() if 'pattern' in df_migration.columns else {}
        }

    def _save_analysis_results(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        metrics: Dict,
        features: list,
        analysis_type: str,
        sort_by: str
    ):
        """Save analysis results (data, models, visualizations)"""
        # Save data
        self.output.save_cluster_data(df, profiles, analysis_type, metrics)
        self.output.save_cluster_lists(df, analysis_type, sort_by=sort_by)
        self.output.save_models(metrics['scaler'], metrics['model'], analysis_type)

        # Save visualizations
        if not self.skip_plots:
            viz_dir = self.output.base_dir / analysis_type / 'visualizations'
            create_all_plots(df, profiles, features, analysis_type=analysis_type, output_dir=viz_dir)

    def _print_summary(self):
        """Print pipeline summary"""
        duration = (datetime.now() - self.start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\n  ⏱️  Duration: {duration:.1f}s")
        logger.info(f"  📁 Output: {self.output.base_dir}\n")

        # Create summary report
        self.output.create_summary_report(self.results)

        print("\n" + "=" * 80)
        print(f"✓ Analysis complete for market: {self.market}")
        print(f"✓ Algorithm: {self.algorithm}")
        print(f"✓ Duration: {duration:.1f}s")
        print(f"\n📂 Output: {self.output.base_dir}/")
        print(f"   ├── static/")
        print(f"   ├── dynamic/")
        print(f"   └── combined/")
        print("\n📌 Next Steps:")
        print(f"   1. Review summary: {self.output.base_dir}/static/reports/data/")
        print(f"   2. Check clusters: {self.output.base_dir}/static/reports/clusters/")
        print(f"   3. Adjust config if needed: config.yaml")
        print("=" * 80 + "\n")
