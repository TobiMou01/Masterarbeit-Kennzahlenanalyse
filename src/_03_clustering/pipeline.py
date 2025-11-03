"""
Clustering Pipeline
Main orchestration logic for 3-stage clustering analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from src._03_clustering.cluster_engine import ClusteringEngine
from src._01_setup.output_handler import OutputHandler
from src._05_visualization.plot_engine import create_all_plots
from src._01_setup import config_loader as config

# New imports for integrated pipeline
from src._04_scoring import ScoreCalculator, ScoreEvolutionTracker, ScoreAnalyzer
from src._06_validation import AlgorithmComparison, ExternalValidation
from src._03_clustering.cluster_naming import ClusterNamer
from src._01_setup.feature_selector import FeatureSelector
from src._05_visualization.plot_engine_scores import PlotEngineScores
from src._05_visualization.plot_engine_validation import PlotEngineValidation
from src._05_visualization.plot_engine_pca import PlotEnginePCA

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

        # Initialize new modules with config-driven feature toggling
        self.feature_selector = FeatureSelector()  # Uses default features_config.yaml

        # Check if scoring is enabled (default: True for backwards compatibility)
        self.scoring_enabled = config.get_value(config_dict, 'scoring', 'enabled', default=True)
        if self.scoring_enabled:
            self.score_calculator = ScoreCalculator(feature_selector=self.feature_selector)
            self.score_tracker = ScoreEvolutionTracker()
            self.score_analyzer = ScoreAnalyzer()
            self.plot_engine_scores = PlotEngineScores()
            logger.info("  ✓ Scoring modules initialized")

        # Check if naming is enabled (default: True)
        self.naming_enabled = config.get_value(config_dict, 'naming', 'enabled', default=True)
        if self.naming_enabled:
            self.cluster_namer = ClusterNamer(feature_selector=self.feature_selector)
            logger.info("  ✓ Cluster naming initialized")

        # Check if validation is enabled (default: True)
        self.validation_enabled = config.get_value(config_dict, 'validation', 'enabled', default=True)
        if self.validation_enabled:
            self.algorithm_comparison = AlgorithmComparison()
            self.external_validation = ExternalValidation()
            self.plot_engine_validation = PlotEngineValidation()
            logger.info("  ✓ Validation modules initialized")

        # Check if PCA is enabled (default: False)
        self.pca_enabled = config.get_value(config_dict, 'pca', 'enabled', default=False)
        if self.pca_enabled:
            self.plot_engine_pca = PlotEnginePCA()
            logger.info("  ✓ PCA visualization initialized")

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

        # Generate summary report (if enabled in config)
        if config.get_value(self.config, 'output', 'create_summary_report', default=True):
            self._generate_summary_report()

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

        # ========== NEW INTEGRATION: Scoring, Naming, Validation ==========

        # 1. Apply Scoring (adds score columns to df_result)
        df_result = self._apply_scoring(
            df=df_result,
            features=features,
            cluster_column='cluster',
            profiles=profiles,
            analysis_type='static'
        )

        # 2. Generate Cluster Names
        cluster_names, naming_summary = self._apply_cluster_naming(
            df=df_result,
            profiles=profiles,
            features=features,
            cluster_column='cluster',
            analysis_type='static'
        )

        # 3. Perform External Validation
        self._perform_validation(
            df=df_result,
            cluster_column='cluster',
            analysis_type='static'
        )

        # 4. Create Score Visualizations
        self._create_score_visualizations(
            df=df_result,
            cluster_column='cluster',
            analysis_type='static'
        )

        # ==================================================================

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, features, 'static', sort_by='roa')

        # Store
        self.results['static'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'profiles': profiles,
            'df': df_result,
            'cluster_names': cluster_names  # NEW: Store names
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

        # ========== NEW INTEGRATION: Scoring, Naming, Validation ==========

        # 1. Apply Scoring
        df_result = self._apply_scoring(
            df=df_result,
            features=features,
            cluster_column='cluster',
            profiles=profiles,
            analysis_type='dynamic'
        )

        # 2. Generate Cluster Names
        cluster_names, naming_summary = self._apply_cluster_naming(
            df=df_result,
            profiles=profiles,
            features=features,
            cluster_column='cluster',
            analysis_type='dynamic'
        )

        # 3. Perform External Validation
        self._perform_validation(
            df=df_result,
            cluster_column='cluster',
            analysis_type='dynamic'
        )

        # 4. Create Score Visualizations
        self._create_score_visualizations(
            df=df_result,
            cluster_column='cluster',
            analysis_type='dynamic'
        )

        # ==================================================================

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
            'df_timeseries': df_all_with_clusters,  # ← NEU: Für Temporal Stability
            'cluster_names': cluster_names  # NEW: Store names
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

        # ========== NEW INTEGRATION: Scoring, Naming, Validation ==========

        # 1. Apply Scoring
        df_result = self._apply_scoring(
            df=df_result,
            features=features_combined,
            cluster_column='cluster',
            profiles=profiles,
            analysis_type='combined'
        )

        # 2. Generate Cluster Names
        cluster_names, naming_summary = self._apply_cluster_naming(
            df=df_result,
            profiles=profiles,
            features=features_combined,
            cluster_column='cluster',
            analysis_type='combined'
        )

        # 3. Perform External Validation
        self._perform_validation(
            df=df_result,
            cluster_column='cluster',
            analysis_type='combined'
        )

        # 4. Create Score Visualizations
        self._create_score_visualizations(
            df=df_result,
            cluster_column='cluster',
            analysis_type='combined'
        )

        # 5. Track Score Evolution (Static → Dynamic → Combined)
        if 'static' in self.results and 'dynamic' in self.results:
            self._track_score_evolution(
                df_static=self.results['static']['df'],
                df_dynamic=self.results['dynamic']['df'],
                df_combined=df_result
            )

        # ==================================================================

        # Save results
        self._save_analysis_results(df_result, profiles, metrics, features_combined, 'combined', sort_by='roa')

        # Cross-analysis (migration patterns)
        df_migration = self.engine.analyze_migration(df_static, df_dynamic, df_result)

        # Save migration data to comparisons/temporal
        self.output.save_comparison_data(
            comp_type='temporal',
            data=df_migration,
            filename='cluster_migrations.csv'
        )

        # Store
        weights = config.get_value(self.config, 'combined_analysis', 'weights', default={'static': 0.4, 'dynamic': 0.6})
        self.results['combined'] = {
            'n_companies': metrics['n_companies'],
            'n_clusters': n_clusters,
            'metrics': metrics,
            'weights': weights,
            'profiles': profiles,
            'df': df_result,
            'cluster_names': cluster_names  # NEW: Store names
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
            plots_dir = self.output.get_plots_dir(analysis_type)
            create_all_plots(df, profiles, features, analysis_type=analysis_type, output_dir=plots_dir)

    # =========================================================================
    # NEW INTEGRATION METHODS
    # =========================================================================

    def _apply_scoring(
        self,
        df: pd.DataFrame,
        features: list,
        cluster_column: str,
        profiles: pd.DataFrame,
        analysis_type: str
    ) -> pd.DataFrame:
        """
        Apply scoring to clustered data

        Calculates:
        - Proximity Score (distance to cluster center)
        - Dimensional Scores (per category: Profitability, Leverage, etc.)
        - Relative Score (Z-score vs cluster average)
        - Overall Score (weighted combination)

        Args:
            df: DataFrame with cluster assignments
            features: List of features used for clustering
            cluster_column: Name of cluster column (default: 'cluster')
            profiles: Cluster profiles DataFrame
            analysis_type: 'static', 'dynamic', or 'combined'

        Returns:
            DataFrame with added score columns
        """
        if not self.scoring_enabled:
            logger.debug(f"  Scoring disabled, skipping...")
            return df

        logger.info(f"\n  💯 Calculating Scores ({analysis_type})...")

        # Calculate all scores
        df_scored = self.score_calculator.calculate_all_scores(
            df=df,
            features=features,
            cluster_column=cluster_column,
            profiles=profiles
        )

        # Save scores to 1_cluster_quality/scores/
        scores_dir = self.output.get_cluster_quality_dir(analysis_type) / 'scores'
        scores_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed scores
        score_columns = [col for col in df_scored.columns if 'score' in col.lower()]
        df_scores = df_scored[['gvkey', 'conm', cluster_column] + score_columns].copy()
        df_scores.to_csv(scores_dir / 'company_scores.csv', index=False)

        logger.info(f"     ✓ Scores calculated and saved")
        logger.info(f"     ✓ Score columns: {len(score_columns)}")

        return df_scored

    def _apply_cluster_naming(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        features: list,
        cluster_column: str,
        analysis_type: str
    ) -> tuple:
        """
        Generate descriptive cluster names

        Args:
            df: DataFrame with cluster assignments
            profiles: Cluster profiles DataFrame
            features: List of features used for clustering
            cluster_column: Name of cluster column
            analysis_type: 'static', 'dynamic', or 'combined'

        Returns:
            Tuple of (cluster_names dict, naming_summary DataFrame)
        """
        if not self.naming_enabled:
            logger.debug(f"  Naming disabled, skipping...")
            return {}, pd.DataFrame()

        logger.info(f"\n  📛 Generating Cluster Names ({analysis_type})...")

        # Get naming method from config (default: 'z_score')
        naming_method = config.get_value(
            self.config, 'naming', 'method', default='z_score'
        )

        # Generate names
        cluster_names, naming_summary = self.cluster_namer.name_clusters(
            profiles=profiles,
            features=features,
            method=naming_method
        )

        # Save naming summary to 1_cluster_quality/naming/
        naming_dir = self.output.get_cluster_quality_dir(analysis_type) / 'naming'
        naming_dir.mkdir(parents=True, exist_ok=True)

        naming_summary.to_csv(naming_dir / 'cluster_names.csv', index=False)

        # Log names
        logger.info(f"     ✓ Naming method: {naming_method}")
        for cluster_id, name in cluster_names.items():
            logger.info(f"     Cluster {cluster_id}: {name}")

        return cluster_names, naming_summary

    def _create_score_visualizations(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        analysis_type: str
    ):
        """
        Create score visualizations

        Generates:
        - Score distributions (box plots)
        - Dimensional heatmaps
        - Score correlations
        - Homogeneity comparisons

        Args:
            df: DataFrame with scores
            cluster_column: Name of cluster column
            analysis_type: 'static', 'dynamic', or 'combined'
        """
        if not self.scoring_enabled or self.skip_plots:
            return

        logger.info(f"\n  📊 Creating Score Visualizations ({analysis_type})...")

        # Output directory: 1_cluster_quality/visualizations/scores/
        viz_dir = self.output.get_cluster_quality_dir(analysis_type) / 'visualizations' / 'scores'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Get score columns
        score_columns = [col for col in df.columns if 'score' in col.lower()]
        dimensional_scores = [col for col in score_columns if col.startswith('dim_')]

        # 1. Score distributions
        if 'overall_score' in df.columns:
            self.plot_engine_scores.plot_score_distribution(
                df=df,
                score_column='overall_score',
                cluster_column=cluster_column,
                output_path=viz_dir / 'score_distribution_overall.png'
            )

        # 2. Dimensional heatmap
        if len(dimensional_scores) > 0:
            self.plot_engine_scores.plot_dimensional_heatmap(
                df=df,
                dimensional_columns=dimensional_scores,
                cluster_column=cluster_column,
                output_path=viz_dir / 'dimensional_heatmap.png'
            )

        # 3. Score correlation matrix
        if len(score_columns) >= 2:
            self.plot_engine_scores.plot_score_correlation_matrix(
                df=df,
                score_columns=score_columns,
                output_path=viz_dir / 'score_correlations.png'
            )

        # 4. Homogeneity comparison
        if 'overall_score' in df.columns:
            self.plot_engine_scores.plot_homogeneity_comparison(
                df=df,
                score_column='overall_score',
                cluster_column=cluster_column,
                output_path=viz_dir / 'cluster_homogeneity.png'
            )

        logger.info(f"     ✓ Score visualizations created in {viz_dir.name}/")

    def _perform_validation(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        analysis_type: str
    ):
        """
        Perform external validation against categorical labels

        Validates clusters against:
        - GICS sectors
        - Company size categories
        - Country (if applicable)

        Args:
            df: DataFrame with cluster assignments and external labels
            cluster_column: Name of cluster column
            analysis_type: 'static', 'dynamic', or 'combined'
        """
        if not self.validation_enabled:
            return

        logger.info(f"\n  🔍 Performing External Validation ({analysis_type})...")

        # Get external labels from config
        external_labels = config.get_value(
            self.config, 'validation', 'external_labels',
            default=['gics_sector', 'size_category']
        )

        # Filter to available columns
        available_labels = [col for col in external_labels if col in df.columns]

        if len(available_labels) == 0:
            logger.warning(f"     ⚠️  No external labels available for validation")
            return

        # Generate validation report
        validation_report = self.external_validation.generate_validation_report(
            df=df,
            cluster_column=cluster_column,
            external_columns=available_labels
        )

        # Save report to 3_external_validation/
        validation_dir = self.output.get_external_validation_dir(analysis_type)
        validation_dir.mkdir(parents=True, exist_ok=True)

        validation_report.to_csv(validation_dir / 'validation_summary.csv', index=False)

        # Create visualizations
        if not self.skip_plots:
            viz_dir = validation_dir / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Cramér's V comparison
            cramers_v_data = validation_report[['external_column', 'cramers_v']].copy()
            self.plot_engine_validation.plot_cramers_v_comparison(
                cramers_v_df=cramers_v_data,
                external_column='external_column',
                value_column='cramers_v',
                output_path=viz_dir / 'cramers_v_comparison.png'
            )

            # Contingency tables for each label
            for label in available_labels:
                self.plot_engine_validation.plot_contingency_heatmap(
                    df=df,
                    cluster_column=cluster_column,
                    external_column=label,
                    output_path=viz_dir / f'contingency_{label}.png'
                )

        logger.info(f"     ✓ Validation report saved")
        logger.info(f"     ✓ External labels validated: {available_labels}")

    def _track_score_evolution(
        self,
        df_static: pd.DataFrame,
        df_dynamic: pd.DataFrame,
        df_combined: pd.DataFrame
    ):
        """
        Track score evolution across analysis phases

        Analyzes:
        - Static → Dynamic changes
        - Dynamic → Combined changes
        - Overall patterns (Improving, Declining, Stable, Volatile)

        Args:
            df_static: Static analysis results with scores
            df_dynamic: Dynamic analysis results with scores
            df_combined: Combined analysis results with scores
        """
        if not self.scoring_enabled:
            return

        logger.info(f"\n  📈 Tracking Score Evolution...")

        # Track evolution
        evolution_df = self.score_tracker.track_evolution(
            df_static=df_static,
            df_dynamic=df_dynamic,
            df_combined=df_combined
        )

        # Analyze patterns
        pattern_summary = self.score_analyzer.analyze_patterns(evolution_df)

        # Save to 1_cluster_quality/scores/evolution/
        evolution_dir = self.output.get_cluster_quality_dir('combined') / 'scores' / 'evolution'
        evolution_dir.mkdir(parents=True, exist_ok=True)

        evolution_df.to_csv(evolution_dir / 'score_evolution.csv', index=False)
        pattern_summary.to_csv(evolution_dir / 'evolution_patterns.csv', index=False)

        # Create visualization
        if not self.skip_plots:
            self.plot_engine_scores.plot_score_evolution_scatter(
                evolution_df=evolution_df,
                output_path=evolution_dir / 'evolution_scatter.png',
                x_column='static_score',
                y_column='dynamic_score',
                pattern_column='pattern'
            )

        logger.info(f"     ✓ Score evolution tracked")
        logger.info(f"     ✓ Patterns: {pattern_summary['pattern'].value_counts().to_dict()}")

    # =========================================================================
    # ORCHESTRATION METHODS
    # =========================================================================

    def run_multi_algorithm_comparison(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame,
        algorithms: list = None
    ):
        """
        Run clustering with multiple algorithms and compare results

        Args:
            df_all: Full dataset (all years)
            df_latest: Latest year only
            algorithms: List of algorithms to compare (default: ['kmeans', 'hierarchical', 'dbscan'])
        """
        if algorithms is None:
            algorithms = ['kmeans', 'hierarchical', 'dbscan']

        logger.info("\n" + "=" * 80)
        logger.info("🔄 MULTI-ALGORITHM COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Algorithms: {algorithms}\n")

        # Storage for results
        results_dict = {}

        # Run each algorithm
        for algo in algorithms:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running: {algo.upper()}")
            logger.info(f"{'='*80}")

            # Create new config with this algorithm
            algo_config = self.config.copy()
            algo_config['classification']['algorithm'] = algo

            # Create new pipeline
            pipeline = ClusteringPipeline(
                config_dict=algo_config,
                market=self.market,
                skip_plots=True  # Skip individual plots
            )

            # Run analysis (combined only for comparison)
            pipeline.run_analysis(
                df_all=df_all,
                df_latest=df_latest,
                run_static=False,
                run_dynamic=False
            )

            # Store results
            if 'combined' in pipeline.results:
                results_dict[algo] = pipeline.results['combined']['df']

        # Compare algorithms
        if len(results_dict) >= 2:
            logger.info("\n" + "=" * 80)
            logger.info("📊 ALGORITHM COMPARISON")
            logger.info("=" * 80)

            comparison_results = self.algorithm_comparison.compare_multiple_algorithms(
                results_dict=results_dict,
                cluster_column='cluster'
            )

            # Save comparison results to 2_algorithm_congruence/
            comparison_dir = self.output.get_algorithm_congruence_dir('combined')
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # Save ARI matrix
            comparison_results['ari_matrix'].to_csv(
                comparison_dir / 'ari_matrix.csv'
            )

            # Save consensus clusters
            comparison_results['consensus'].to_csv(
                comparison_dir / 'consensus_clusters.csv', index=False
            )

            # Save summary
            comparison_results['summary'].to_csv(
                comparison_dir / 'comparison_summary.csv', index=False
            )

            # Create visualizations
            if not self.skip_plots:
                viz_dir = comparison_dir / 'visualizations'
                viz_dir.mkdir(parents=True, exist_ok=True)

                # ARI heatmap
                self.plot_engine_validation.plot_ari_heatmap(
                    ari_matrix=comparison_results['ari_matrix'],
                    output_path=viz_dir / 'ari_heatmap.png'
                )

                # Confusion matrices
                self.plot_engine_validation.plot_multiple_confusion_matrices(
                    confusion_matrices=comparison_results['confusion_matrices'],
                    output_path=viz_dir / 'confusion_matrices.png'
                )

            logger.info(f"\n  ✓ Algorithm comparison completed")
            logger.info(f"  ✓ Mean ARI: {comparison_results['ari_matrix'].values[np.triu_indices_from(comparison_results['ari_matrix'].values, k=1)].mean():.3f}")
            logger.info(f"  ✓ Consensus clusters: {len(comparison_results['consensus'])}")

    def run_with_pca_validation(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame,
        extended_features_preset: str = 'pca_optimized'
    ):
        """
        Run clustering with PCA validation

        Performs parallel clustering in:
        1. Original feature space (base features)
        2. PCA space (extended features)

        Then compares and validates results.

        Args:
            df_all: Full dataset (all years)
            df_latest: Latest year only
            extended_features_preset: Feature preset for extended features
        """
        from src._03_clustering.pca_pipeline import PCAPipeline

        logger.info("\n" + "=" * 80)
        logger.info("🔬 PCA VALIDATION")
        logger.info("=" * 80)

        # Create PCA pipeline
        pca_pipeline = PCAPipeline(
            config_dict=self.config,
            market=self.market,
            extended_features_preset=extended_features_preset
        )

        # Run parallel analysis (static, dynamic, combined)
        pca_pipeline.run_parallel_analysis(
            df=df_latest,
            n_clusters=config.get_value(self.config, 'static_analysis', 'n_clusters', default=5),
            analysis_type='static'
        )

        pca_pipeline.run_parallel_analysis(
            df=df_all,
            n_clusters=config.get_value(self.config, 'dynamic_analysis', 'n_clusters', default=5),
            analysis_type='dynamic'
        )

        # Note: Combined analysis would need merged data
        logger.info(f"\n  ✓ PCA validation completed")
        logger.info(f"  ✓ Results saved to: {self.output.get_pca_analysis_dir('static')}")

    def _generate_summary_report(self, output_path: Path = None):
        """
        Generate comprehensive summary report in Markdown

        Includes:
        - Analysis overview (algorithms, features, clusters)
        - Cluster quality metrics
        - Validation results (Cramér's V, ARI)
        - Key insights and recommendations

        Args:
            output_path: Path to save report (default: summary_dir/analysis_report.md)
        """
        if output_path is None:
            output_path = self.output.summary_dir / 'analysis_report.md'

        logger.info(f"\n  📄 Generating Summary Report...")

        # Build report
        report_lines = []

        # Header
        report_lines.append("# Clustering Analysis Summary Report")
        report_lines.append(f"\n**Market:** {self.market}")
        report_lines.append(f"**Algorithm:** {self.algorithm}")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n---\n")

        # 1. Analysis Overview
        report_lines.append("## 1. Analysis Overview\n")

        for analysis_key in ['static', 'dynamic', 'combined']:
            if analysis_key in self.results:
                result = self.results[analysis_key]
                report_lines.append(f"### {analysis_key.capitalize()} Analysis\n")
                report_lines.append(f"- **Companies:** {result['n_companies']}")
                report_lines.append(f"- **Clusters:** {result['n_clusters']}")
                report_lines.append(f"- **Silhouette Score:** {result['metrics'].get('silhouette_score', 'N/A'):.3f}")
                report_lines.append(f"- **Davies-Bouldin Index:** {result['metrics'].get('davies_bouldin_score', 'N/A'):.3f}\n")

        # 2. Cluster Profiles
        report_lines.append("## 2. Cluster Profiles\n")

        if 'static' in self.results:
            profiles = self.results['static']['profiles']
            report_lines.append("### Static Analysis Profiles\n")
            report_lines.append(profiles.to_markdown())
            report_lines.append("\n")

        # 3. Validation Results
        if self.validation_enabled:
            report_lines.append("## 3. Validation Results\n")
            report_lines.append("*External validation results can be found in 3_external_validation/*\n")

        # 4. Score Analysis
        if self.scoring_enabled:
            report_lines.append("## 4. Score Analysis\n")
            report_lines.append("*Detailed score analysis can be found in 1_cluster_quality/scores/*\n")

        # 5. Output Structure
        report_lines.append("## 5. Output Structure\n")
        report_lines.append("```")
        report_lines.append(f"{self.output.algorithm_dir}/")
        report_lines.append("├── 1_cluster_quality/")
        report_lines.append("│   ├── scores/")
        report_lines.append("│   ├── naming/")
        report_lines.append("│   └── visualizations/")
        report_lines.append("├── 2_algorithm_congruence/")
        report_lines.append("├── 3_external_validation/")
        report_lines.append("├── 4_company_insights/")
        report_lines.append("└── 5_pca_analysis/")
        report_lines.append("```\n")

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"     ✓ Summary report saved: {output_path}")

    # =========================================================================
    # EXISTING METHODS
    # =========================================================================

    def _print_summary(self):
        """Print pipeline summary"""
        duration = (datetime.now() - self.start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\n  ⏱️  Duration: {duration:.1f}s")
        logger.info(f"  📁 Output: {self.output.algorithm_dir}\n")

        print("\n" + "=" * 80)
        print(f"✓ Analysis complete for market: {self.market}")
        print(f"✓ Algorithm: {self.algorithm} ({self.output.mode} mode)")
        print(f"✓ Duration: {duration:.1f}s")
        print(f"\n📂 Output: {self.output.algorithm_dir}/")
        for analysis_key in ['static', 'dynamic', 'combined']:
            analysis_name = self.output.analysis_types[analysis_key]
            print(f"   ├── {analysis_name}/")
        print("\n📌 Next Steps:")
        print(f"   1. Review summary: {self.output.summary_dir}/")
        print(f"   2. Check clusters: {self.output.algorithm_dir}/{self.output.analysis_types['static']}/reports/clusters/")
        print(f"   3. Adjust config if needed: config.yaml")
        print("=" * 80 + "\n")
