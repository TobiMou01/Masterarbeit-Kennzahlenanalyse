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
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Include additional metadata columns needed for external labels
        metadata_cols = ['revt', 'at', 'sale', 'gsector', 'gsubind', 'ggroup', 'gind']
        for col in metadata_cols:
            if col in df_static.columns and col not in cols_static:
                cols_static.append(col)

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

            # Create PCA plots if enabled
            if self.pca_enabled:
                self._create_pca_plots(df, features, analysis_type)

            # Create algorithm congruence plots (robustness check)
            self._create_algorithm_congruence_plots(df, analysis_type)

            # Create company insights plots
            self._create_company_insights_plots(df, profiles, features, analysis_type)

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

        # Build columns list - only include columns that exist
        columns_to_save = ['gvkey']
        if 'conm' in df_scored.columns:
            columns_to_save.append('conm')
        elif 'company_name' in df_scored.columns:
            columns_to_save.append('company_name')

        columns_to_save.append(cluster_column)
        columns_to_save.extend(score_columns)

        df_scores = df_scored[columns_to_save].copy()
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

        # Get naming style from config (default: 'hybrid')
        # Map old 'method' values to new 'style' values
        naming_method = config.get_value(
            self.config, 'naming', 'method', default='hybrid'
        )

        # Map method names to style names
        method_to_style = {
            'z_score': 'technical',
            'top_features': 'technical',
            'percentile': 'technical',
            'hybrid': 'hybrid',
            'business': 'business',
            'technical': 'technical'
        }

        style = method_to_style.get(naming_method, 'hybrid')

        # Generate names
        cluster_names = self.cluster_namer.generate_names(
            profiles=profiles,
            features=features,
            style=style,
            top_n=2
        )

        # Create naming summary DataFrame
        naming_summary = pd.DataFrame([
            {'cluster_id': cid, 'cluster_name': name}
            for cid, name in cluster_names.items()
        ])

        # Save naming summary to 1_cluster_quality/naming/
        naming_dir = self.output.get_cluster_quality_dir(analysis_type) / 'naming'
        naming_dir.mkdir(parents=True, exist_ok=True)

        naming_summary.to_csv(naming_dir / 'cluster_names.csv', index=False)

        # Log names
        logger.info(f"     ✓ Naming style: {style}")
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

        # Output directory: 1_cluster_quality/plots/
        viz_dir = self.output.get_cluster_quality_dir(analysis_type) / 'plots'
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
                dimensional_score_columns=dimensional_scores,
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
            # First analyze homogeneity
            homogeneity_df = self.score_analyzer.analyze_cluster_homogeneity(
                df=df,
                score_column='overall_score',
                cluster_column=cluster_column
            )

            # Then plot it
            self.plot_engine_scores.plot_homogeneity_comparison(
                homogeneity_df=homogeneity_df,
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

        # Add missing external labels if needed
        df = self._add_external_labels(df)

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

        # Create summary DataFrame from report dictionary
        summary_data = []
        for ext_label in available_labels:
            if ext_label in validation_report.get('cramers_v', {}):
                cramers_data = validation_report['cramers_v'][ext_label]
                chi2_data = validation_report.get('chi_square', {}).get(ext_label, {})

                summary_data.append({
                    'external_column': ext_label,
                    'cramers_v': cramers_data.get('value', 0),
                    'interpretation': cramers_data.get('interpretation', 'N/A'),
                    'chi2_statistic': chi2_data.get('chi2', 0),
                    'p_value': chi2_data.get('p_value', 1.0),
                    'significant': chi2_data.get('significant', False)
                })

        validation_summary = pd.DataFrame(summary_data)

        # Save summary
        if not validation_summary.empty:
            validation_summary.to_csv(validation_dir / 'validation_summary.csv', index=False)

        # Create visualizations
        if not self.skip_plots and not validation_summary.empty:
            viz_dir = validation_dir / 'plots'
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Cramér's V comparison
            cramers_v_data = validation_summary[['external_column', 'cramers_v', 'interpretation']].copy()
            # Rename column to match plot_engine expectation
            cramers_v_data = cramers_v_data.rename(columns={'external_column': 'external_label'})
            self.plot_engine_validation.plot_cramers_v_comparison(
                cramers_v_df=cramers_v_data,
                output_path=viz_dir / 'cramers_v_comparison.png'
            )

            # Contingency tables for each label
            for label in available_labels:
                if label in validation_report.get('contingency_tables', {}):
                    contingency = validation_report['contingency_tables'][label]
                    self.plot_engine_validation.plot_contingency_heatmap(
                        contingency_table=contingency,
                        cluster_name='Cluster',
                        external_name=label,
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

        # Classify evolution patterns
        patterns = self.score_tracker.classify_evolution_pattern(evolution_df)
        evolution_df['pattern'] = patterns

        # Create pattern summary
        pattern_counts = patterns.value_counts()
        pattern_summary = pd.DataFrame({
            'pattern': pattern_counts.index,
            'count': pattern_counts.values,
            'percentage': (pattern_counts.values / len(patterns) * 100).round(1)
        })

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
        logger.info(f"     ✓ Patterns: {pattern_counts.to_dict()}")

    def _add_external_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing external labels for validation

        Creates:
        - gics_sector: From gsector column (GICS Sector names)
        - size_category: From revenue/assets (Small, Medium, Large)

        Args:
            df: DataFrame

        Returns:
            DataFrame with added external labels
        """
        df = df.copy()

        # 1. Add gics_sector from gsector
        if 'gics_sector' not in df.columns and 'gsector' in df.columns:
            # GICS Sector mapping (standard GICS sectors)
            gics_mapping = {
                10: 'Energy',
                15: 'Materials',
                20: 'Industrials',
                25: 'Consumer Discretionary',
                30: 'Consumer Staples',
                35: 'Health Care',
                40: 'Financials',
                45: 'Information Technology',
                50: 'Communication Services',
                55: 'Utilities',
                60: 'Real Estate'
            }

            df['gics_sector'] = df['gsector'].map(gics_mapping)

            # Fill unmapped values
            df['gics_sector'] = df['gics_sector'].fillna('Other')

            logger.info(f"     ✓ Created gics_sector from gsector")

        # 2. Add size_category from revenue or assets
        if 'size_category' not in df.columns:
            # Try revenue first, then assets
            if 'revt' in df.columns:
                size_col = 'revt'
            elif 'at' in df.columns:
                size_col = 'at'
            elif 'sale' in df.columns:
                size_col = 'sale'
            else:
                logger.warning(f"     ⚠️  Cannot create size_category: no revenue/assets column")
                return df

            # Calculate size categories based on terciles
            valid_sizes = df[df[size_col].notna()][size_col]

            if len(valid_sizes) > 0:
                tercile_33 = valid_sizes.quantile(0.33)
                tercile_67 = valid_sizes.quantile(0.67)

                def categorize_size(value):
                    if pd.isna(value):
                        return 'Unknown'
                    elif value < tercile_33:
                        return 'Small'
                    elif value < tercile_67:
                        return 'Medium'
                    else:
                        return 'Large'

                df['size_category'] = df[size_col].apply(categorize_size)

                logger.info(f"     ✓ Created size_category from {size_col}")
                logger.info(f"       Small: < {tercile_33:.0f}, Medium: {tercile_33:.0f}-{tercile_67:.0f}, Large: > {tercile_67:.0f}")

        return df

    def _create_pca_plots(
        self,
        df: pd.DataFrame,
        features: List[str],
        analysis_type: str
    ):
        """
        Create PCA visualization plots

        Generates:
        - Scree plot (variance explained)
        - Component loadings heatmap
        - Biplot (observations + features)
        - Cluster separation in PCA space

        Args:
            df: DataFrame with features and cluster assignments
            features: List of feature names used for clustering
            analysis_type: 'static', 'dynamic', or 'combined'
        """
        if not self.pca_enabled or self.skip_plots:
            return

        logger.info(f"\n  🔬 Creating PCA Plots ({analysis_type})...")

        from src._02_preprocessing.pca_transformer import PCATransformer

        # Output directory: 5_pca_analysis/plots/
        pca_dir = self.output.get_pca_analysis_dir(analysis_type) / 'plots'
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Validate features exist in DataFrame
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < 2:
            logger.warning(f"     ⚠️  Not enough features ({len(available_features)}) for PCA")
            return

        # Apply PCA transformation
        pca_transformer = PCATransformer(n_components=0.85)
        try:
            X_pca, variance_summary = pca_transformer.fit_transform(
                df=df,
                features=available_features
            )
        except Exception as e:
            logger.warning(f"     ⚠️  PCA transformation failed: {e}")
            return

        n_components = pca_transformer.pca.n_components_
        explained_variance = pca_transformer.pca.explained_variance_ratio_

        logger.info(f"     ✓ PCA complete: {len(available_features)} features → {n_components} components")

        # Get cluster labels
        cluster_labels = df['cluster'].values if 'cluster' in df.columns else np.zeros(len(df))

        # 1. Scree Plot
        try:
            self.plot_engine_pca.plot_scree_plot(
                explained_variance,
                output_path=pca_dir / 'scree_plot.png',
                cumulative=True,
                threshold=0.85
            )
        except Exception as e:
            logger.warning(f"     ⚠️  Scree plot failed: {e}")

        # 2. Component Loadings Heatmap
        try:
            component_loadings = pca_transformer.get_component_loadings()
            self.plot_engine_pca.plot_component_loadings_heatmap(
                component_loadings,
                output_path=pca_dir / 'component_loadings.png',
                top_n_features=min(15, len(available_features))
            )
        except Exception as e:
            logger.warning(f"     ⚠️  Component loadings plot failed: {e}")

        # 3. Biplot (if we have at least 2 components)
        if n_components >= 2:
            try:
                loadings = pca_transformer.pca.components_.T
                self.plot_engine_pca.plot_biplot(
                    X_pca,
                    loadings,
                    available_features,
                    cluster_labels,
                    output_path=pca_dir / 'biplot_pc1_pc2.png',
                    pc_x=0,
                    pc_y=1,
                    n_features_show=min(10, len(available_features))
                )
            except Exception as e:
                logger.warning(f"     ⚠️  Biplot failed: {e}")

        # 4. Cluster Separation in PCA Space (if we have clusters)
        if n_components >= 2 and len(np.unique(cluster_labels)) > 1:
            try:
                cluster_names = [f'Cluster {i}' for i in sorted(np.unique(cluster_labels))]

                # Determine which component pairs to plot
                components_to_plot = [(0, 1)]  # Always PC1 vs PC2
                if n_components >= 3:
                    components_to_plot.extend([(0, 2), (1, 2)])

                self.plot_engine_pca.plot_cluster_separation_in_pca_space(
                    X_pca,
                    cluster_labels,
                    cluster_names,
                    output_path=pca_dir / 'cluster_separation.png',
                    components=components_to_plot
                )
            except Exception as e:
                logger.warning(f"     ⚠️  Cluster separation plot failed: {e}")

        logger.info(f"     ✓ PCA plots saved to {pca_dir.name}/")

    def _create_algorithm_congruence_plots(
        self,
        df: pd.DataFrame,
        analysis_type: str
    ):
        """
        Create algorithm congruence plots

        Tests clustering robustness by running multiple initializations
        and comparing results using ARI (Adjusted Rand Index).

        Generates:
        - ARI heatmap (multiple runs)
        - Stability analysis
        - Confusion matrix

        Args:
            df: DataFrame with cluster assignments
            analysis_type: 'static', 'dynamic', or 'combined'
        """
        if not self.validation_enabled or self.skip_plots:
            return

        logger.info(f"\n  🔄 Creating Algorithm Congruence Plots ({analysis_type})...")

        # Output directory: 2_algorithm_congruence/plots/
        congruence_dir = self.output.get_algorithm_congruence_dir(analysis_type) / 'plots'
        congruence_dir.mkdir(parents=True, exist_ok=True)

        # Get features and cluster column
        if 'cluster' not in df.columns:
            logger.warning(f"     ⚠️  No cluster column found")
            return

        # Run multiple clusterings with different seeds to test robustness
        from sklearn.metrics import adjusted_rand_score
        from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

        n_runs = 5  # Number of random initializations
        cluster_assignments = {}
        cluster_assignments['original'] = df['cluster'].values

        # Store features used for clustering (try to infer from DataFrame)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_candidates = [col for col in numeric_cols
                            if col not in ['cluster', 'gvkey', 'fyear', 'datadate', 'gsector', 'sic', 'naics']
                            and not col.endswith('_outlier')
                            and not col.endswith('_score')]

        features = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates

        if len(features) < 2:
            logger.warning(f"     ⚠️  Not enough features ({len(features)}) for re-clustering")
            # Create placeholder info file
            info_path = congruence_dir / 'info.txt'
            with open(info_path, 'w') as f:
                f.write("Algorithm Congruence Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write("Algorithm congruence plots are only available when:\n")
                f.write("1. Multiple clustering algorithms are compared, OR\n")
                f.write("2. The same algorithm runs with different parameters\n\n")
                f.write(f"Current run: Single {self.algorithm} clustering\n")
                f.write(f"For multi-algorithm comparison, use:\n")
                f.write("  pipeline.run_multi_algorithm_comparison()\n")
            logger.info(f"     ℹ️  Created info file: {info_path.name}")
            return

        # Perform multiple runs with different seeds
        logger.info(f"     Running {n_runs} clusterings with different initializations...")

        for i in range(n_runs):
            try:
                # Re-run clustering with different seed
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                X = df[features].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_clusters = len(df['cluster'].unique())
                model = KMeans(n_clusters=n_clusters, random_state=42 + i, n_init=10)
                labels = model.fit_predict(X_scaled)

                cluster_assignments[f'run_{i+1}'] = labels

            except Exception as e:
                logger.warning(f"     ⚠️  Run {i+1} failed: {e}")

        if len(cluster_assignments) < 2:
            logger.warning(f"     ⚠️  Not enough successful runs for comparison")
            return

        # Calculate ARI matrix
        run_names = list(cluster_assignments.keys())
        n_comparisons = len(run_names)
        ari_matrix = np.ones((n_comparisons, n_comparisons))

        for i, name1 in enumerate(run_names):
            for j, name2 in enumerate(run_names):
                if i < j:
                    ari = adjusted_rand_score(
                        cluster_assignments[name1],
                        cluster_assignments[name2]
                    )
                    ari_matrix[i, j] = ari
                    ari_matrix[j, i] = ari

        ari_df = pd.DataFrame(ari_matrix, index=run_names, columns=run_names)

        logger.info(f"     ✓ ARI Matrix calculated (mean ARI: {ari_df.values[np.triu_indices_from(ari_df.values, k=1)].mean():.3f})")

        # Create ARI heatmap
        try:
            self.plot_engine_validation.plot_ari_heatmap(
                ari_df,
                output_path=congruence_dir / 'ari_heatmap_robustness.png',
                title=f'Clustering Robustness: ARI Across {n_runs+1} Runs'
            )
        except Exception as e:
            logger.warning(f"     ⚠️  ARI heatmap failed: {e}")

        # Create confusion matrix (original vs run_1)
        if 'run_1' in cluster_assignments:
            try:
                conf_matrix = sklearn_confusion_matrix(
                    cluster_assignments['original'],
                    cluster_assignments['run_1']
                )
                conf_df = pd.DataFrame(conf_matrix)

                self.plot_engine_validation.plot_confusion_matrix(
                    conf_df,
                    'Original',
                    'Run 1',
                    output_path=congruence_dir / 'confusion_matrix.png',
                    title='Cluster Assignment Consistency'
                )
            except Exception as e:
                logger.warning(f"     ⚠️  Confusion matrix failed: {e}")

        # Save ARI matrix
        ari_df.to_csv(congruence_dir.parent / 'ari_matrix.csv')

        logger.info(f"     ✓ Algorithm congruence plots saved to {congruence_dir.name}/")

    def _create_company_insights_plots(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        features: List[str],
        analysis_type: str
    ):
        """
        Create company insights plots

        Generates:
        - Top/Bottom performers per cluster
        - Outlier detection (companies far from cluster center)
        - Feature distributions per cluster
        - Cluster size distribution

        Args:
            df: DataFrame with cluster assignments and features
            profiles: Cluster profiles (means)
            features: List of features used for clustering
            analysis_type: 'static', 'dynamic', or 'combined'
        """
        if self.skip_plots:
            return

        logger.info(f"\n  💼 Creating Company Insights Plots ({analysis_type})...")

        # Output directory: 4_company_insights/plots/
        insights_dir = self.output.get_company_insights_dir(analysis_type) / 'plots'
        insights_dir.mkdir(parents=True, exist_ok=True)

        if 'cluster' not in df.columns:
            logger.warning(f"     ⚠️  No cluster column found")
            return

        # 1. Cluster Size Distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            cluster_sizes = df['cluster'].value_counts().sort_index()
            colors = sns.color_palette("husl", n_colors=len(cluster_sizes))

            bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors,
                         edgecolor='black', linewidth=1.5, alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
            ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(insights_dir / 'cluster_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"     ✓ Cluster size distribution plot created")

        except Exception as e:
            logger.warning(f"     ⚠️  Cluster size plot failed: {e}")

        # 2. Feature Distributions per Cluster
        try:
            # Select top 6 most important features (highest variance across clusters)
            available_features = [f for f in features if f in df.columns][:6]

            if len(available_features) >= 2:
                n_features = len(available_features)
                n_cols = 2
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
                axes = axes.flatten() if n_features > 1 else [axes]

                for idx, feature in enumerate(available_features):
                    ax = axes[idx]

                    # Create violin plot
                    data_for_plot = []
                    labels_for_plot = []

                    for cluster_id in sorted(df['cluster'].unique()):
                        cluster_data = df[df['cluster'] == cluster_id][feature].dropna()
                        data_for_plot.append(cluster_data)
                        labels_for_plot.append(f'C{cluster_id}')

                    parts = ax.violinplot(data_for_plot, positions=range(len(data_for_plot)),
                                         showmeans=True, showmedians=True)

                    # Color the violins
                    colors_violin = sns.color_palette("husl", n_colors=len(data_for_plot))
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors_violin[i])
                        pc.set_alpha(0.7)

                    ax.set_xticks(range(len(labels_for_plot)))
                    ax.set_xticklabels(labels_for_plot)
                    ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
                    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                    ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')

                # Hide unused subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].axis('off')

                fig.suptitle('Feature Distributions Across Clusters', fontsize=16, fontweight='bold', y=0.995)
                plt.tight_layout()
                plt.savefig(insights_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"     ✓ Feature distribution plots created")

        except Exception as e:
            logger.warning(f"     ⚠️  Feature distribution plots failed: {e}")

        # 3. Top/Bottom Performers (based on overall_score if available)
        try:
            score_column = None
            for col in ['overall_score', 'proximity_score', 'roa', 'roe']:
                if col in df.columns:
                    score_column = col
                    break

            if score_column:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Top Performers
                ax_top = axes[0]
                top_n = min(10, len(df))
                top_performers = df.nlargest(top_n, score_column)[['conm' if 'conm' in df.columns else 'gvkey', score_column, 'cluster']]

                y_pos = np.arange(top_n)
                colors_top = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in top_performers['cluster'].values]

                ax_top.barh(y_pos, top_performers[score_column].values, color=colors_top, edgecolor='black', alpha=0.8)
                ax_top.set_yticks(y_pos)

                company_col = 'conm' if 'conm' in df.columns else 'gvkey'
                labels_top = [f"{name[:20]}... (C{int(c)})" if len(str(name)) > 20 else f"{name} (C{int(c)})"
                            for name, c in zip(top_performers[company_col].values, top_performers['cluster'].values)]
                ax_top.set_yticklabels(labels_top, fontsize=9)

                ax_top.set_xlabel(score_column.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax_top.set_title(f'Top {top_n} Performers', fontsize=12, fontweight='bold')
                ax_top.grid(True, alpha=0.3, axis='x')

                # Bottom Performers
                ax_bottom = axes[1]
                bottom_performers = df.nsmallest(top_n, score_column)[['conm' if 'conm' in df.columns else 'gvkey', score_column, 'cluster']]

                colors_bottom = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in bottom_performers['cluster'].values]

                ax_bottom.barh(y_pos, bottom_performers[score_column].values, color=colors_bottom, edgecolor='black', alpha=0.8)
                ax_bottom.set_yticks(y_pos)

                labels_bottom = [f"{name[:20]}... (C{int(c)})" if len(str(name)) > 20 else f"{name} (C{int(c)})"
                               for name, c in zip(bottom_performers[company_col].values, bottom_performers['cluster'].values)]
                ax_bottom.set_yticklabels(labels_bottom, fontsize=9)

                ax_bottom.set_xlabel(score_column.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax_bottom.set_title(f'Bottom {top_n} Performers', fontsize=12, fontweight='bold')
                ax_bottom.grid(True, alpha=0.3, axis='x')

                fig.suptitle(f'Company Performance Ranking (by {score_column.replace("_", " ").title()})',
                            fontsize=14, fontweight='bold', y=0.98)

                plt.tight_layout()
                plt.savefig(insights_dir / 'top_bottom_performers.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"     ✓ Top/bottom performers plot created")

        except Exception as e:
            logger.warning(f"     ⚠️  Top/bottom performers plot failed: {e}")

        # 4. Outlier Detection (Distance from Cluster Center)
        try:
            from sklearn.preprocessing import StandardScaler
            from scipy.spatial.distance import cdist

            available_features = [f for f in features if f in df.columns]
            if len(available_features) >= 2:
                X = df[available_features].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Calculate distance to cluster center for each point
                distances = []
                for cluster_id in df['cluster'].unique():
                    mask = df['cluster'].values == cluster_id
                    cluster_points = X_scaled[mask]

                    if len(cluster_points) > 0:
                        center = cluster_points.mean(axis=0).reshape(1, -1)
                        dists = cdist(cluster_points, center, metric='euclidean').flatten()
                        distances.extend(dists)
                    else:
                        distances.extend([0] * mask.sum())

                df_temp = df.copy()
                df_temp['distance_to_center'] = distances

                # Identify outliers (top 5% furthest from center)
                threshold = df_temp['distance_to_center'].quantile(0.95)
                outliers = df_temp[df_temp['distance_to_center'] > threshold].nlargest(15, 'distance_to_center')

                if len(outliers) > 0:
                    fig, ax = plt.subplots(figsize=(12, 7))

                    y_pos = np.arange(len(outliers))
                    colors_outliers = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in outliers['cluster'].values]

                    ax.barh(y_pos, outliers['distance_to_center'].values, color=colors_outliers,
                           edgecolor='black', alpha=0.8, linewidth=1.5)

                    ax.set_yticks(y_pos)
                    company_col = 'conm' if 'conm' in df_temp.columns else 'gvkey'
                    labels_outliers = [f"{name[:25]}... (C{int(c)})" if len(str(name)) > 25 else f"{name} (C{int(c)})"
                                     for name, c in zip(outliers[company_col].values, outliers['cluster'].values)]
                    ax.set_yticklabels(labels_outliers, fontsize=9)

                    ax.set_xlabel('Distance to Cluster Center', fontsize=11, fontweight='bold')
                    ax.set_title('Outlier Companies (Furthest from Cluster Center)', fontsize=13, fontweight='bold', pad=20)
                    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'95th Percentile ({threshold:.2f})', alpha=0.7)
                    ax.legend(loc='lower right')
                    ax.grid(True, alpha=0.3, axis='x')

                    plt.tight_layout()
                    plt.savefig(insights_dir / 'outliers.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    logger.info(f"     ✓ Outlier detection plot created")

        except Exception as e:
            logger.warning(f"     ⚠️  Outlier detection plot failed: {e}")

        logger.info(f"     ✓ Company insights plots saved to {insights_dir.name}/")

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

                # Format metrics with proper type checking
                silhouette = result['metrics'].get('silhouette_score', 'N/A')
                if isinstance(silhouette, (int, float)):
                    report_lines.append(f"- **Silhouette Score:** {silhouette:.3f}")
                else:
                    report_lines.append(f"- **Silhouette Score:** {silhouette}")

                davies_bouldin = result['metrics'].get('davies_bouldin_score', 'N/A')
                if isinstance(davies_bouldin, (int, float)):
                    report_lines.append(f"- **Davies-Bouldin Index:** {davies_bouldin:.3f}\n")
                else:
                    report_lines.append(f"- **Davies-Bouldin Index:** {davies_bouldin}\n")

        # 2. Cluster Profiles
        report_lines.append("## 2. Cluster Profiles\n")

        if 'static' in self.results:
            profiles = self.results['static']['profiles']
            report_lines.append("### Static Analysis Profiles\n")

            # Convert to markdown table manually (no tabulate dependency needed)
            report_lines.append("```")
            report_lines.append(profiles.to_string())
            report_lines.append("```")
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
