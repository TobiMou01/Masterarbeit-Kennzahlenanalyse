"""
Comparison Pipeline
Führt Vergleiche zwischen Algorithmen, GICS, Features und zeitlicher Stabilität durch
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from src._01_setup import config_loader as config
from src._03_clustering.pipeline import ClusteringPipeline
from src._04_comparison.gics_analyzer import GICSComparison
from src._04_comparison.algorithm_analyzer import AlgorithmComparison
from src._04_comparison.feature_analyzer import FeatureImportance
from src._04_comparison.temporal_analyzer import TemporalStability
# Note: ComparisonHandler logic integrated into this class

logger = logging.getLogger(__name__)


class ComparisonPipeline:
    """
    Pipeline für umfassende Vergleichsanalysen

    Führt aus:
    1. GICS Comparison - Vergleich mit Branchen-Klassifikation
    2. Algorithm Comparison - Vergleich zwischen K-Means, Hierarchical, DBSCAN
    3. Feature Importance - Welche Features trennen die Cluster?
    4. Temporal Stability - Wie stabil sind Cluster über Zeit?
    """

    def __init__(
        self,
        config_dict: dict,
        market: str,
        algorithms: List[str] = None,
        skip_plots: bool = False
    ):
        """
        Initialize Comparison Pipeline

        Args:
            config_dict: Configuration dictionary
            market: Market name (e.g., 'germany')
            algorithms: Liste der zu vergleichenden Algorithmen (default: all)
            skip_plots: Skip visualization generation
        """
        self.config = config_dict
        self.market = market
        self.skip_plots = skip_plots

        # Algorithms to compare
        if algorithms is None:
            self.algorithms = ['kmeans', 'hierarchical', 'dbscan']
        else:
            self.algorithms = algorithms

        # Initialize handlers
        self.comparison_handler = ComparisonHandler(market=market)
        self.gics_comparison = GICSComparison()
        self.algo_comparison = AlgorithmComparison()
        self.feature_importance = FeatureImportance()
        self.temporal_stability = TemporalStability()

        # Results storage
        self.algorithm_results = {}  # {algorithm: {static: df, dynamic: df, combined: df}}
        self.comparison_results = {}

    def run_all_algorithms(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame
    ) -> Dict:
        """
        Führt Clustering für alle Algorithmen aus

        Args:
            df_all: DataFrame mit allen Jahren
            df_latest: DataFrame mit aktuellem Jahr

        Returns:
            Dict mit Ergebnissen pro Algorithmus
        """
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING ALL ALGORITHMS")
        logger.info(f"{'='*80}")

        for algorithm in self.algorithms:
            logger.info(f"\n--- Running {algorithm.upper()} ---")

            # Update config für aktuellen Algorithmus
            algo_config = self.config.copy()
            if 'classification' not in algo_config:
                algo_config['classification'] = {}
            algo_config['classification']['algorithm'] = algorithm

            # Create pipeline für diesen Algorithmus
            pipeline = ClusteringPipeline(
                config_dict=algo_config,
                market=self.market,
                skip_plots=True  # Vermeide individuelle Plots, nur Comparison-Plots
            )

            # Run pipeline und extrahiere Ergebnisse
            pipeline.run_analysis(
                df_all=df_all,
                df_latest=df_latest,
                run_static=True,
                run_dynamic=True
            )

            # Speichere Ergebnisse aus Pipeline
            self.algorithm_results[algorithm] = {
                'static': pipeline.results.get('static'),
                'dynamic': pipeline.results.get('dynamic'),
                'combined': pipeline.results.get('combined')
            }

        logger.info(f"\n✓ All algorithms completed")
        return self.algorithm_results

    def run_comparisons(self) -> Dict:
        """
        Führt alle Vergleichsanalysen durch

        Returns:
            Dict mit Comparison-Ergebnissen
        """
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON ANALYSES")
        logger.info(f"{'='*80}")

        # 1. GICS Comparison
        self._run_gics_comparison()

        # 2. Algorithm Comparison
        self._run_algorithm_comparison()

        # 3. Feature Importance
        self._run_feature_importance()

        # 4. Temporal Stability
        self._run_temporal_stability()

        logger.info(f"\n✓ All comparisons completed")
        return self.comparison_results

    def _run_gics_comparison(self):
        """GICS Comparison für alle Algorithmen und Analyse-Stadien"""
        logger.info(f"\n{'='*80}")
        logger.info("1. GICS COMPARISON")
        logger.info(f"{'='*80}")

        gics_col = 'gsector'  # Haupt-Sektor (10 Sektoren)

        # Für jedes Analyse-Stadium
        for stage in ['static', 'dynamic', 'combined']:
            logger.info(f"\n--- GICS Comparison: {stage.upper()} ---")

            # Sammle Daten aller Algorithmen für dieses Stadium
            stage_results = {}
            for algorithm, results in self.algorithm_results.items():
                if results[stage] and 'df' in results[stage]:
                    df = results[stage]['df']
                    if gics_col in df.columns:
                        stage_results[algorithm] = df

            if not stage_results:
                logger.warning(f"  ⚠ Keine Daten für {stage}")
                continue

            # Vergleiche
            cramers_df, rand_df, chi2_df, contingency_tables = \
                self.gics_comparison.compare_all_algorithms(
                    results_dict=stage_results,
                    gics_col=gics_col,
                    analysis_type=stage
                )

            if cramers_df is not None:
                # Speichere Ergebnisse
                self.comparison_handler.save_gics_comparison(
                    cramers_v=cramers_df,
                    rand_index=rand_df,
                    chi_square=chi2_df,
                    contingency_tables=contingency_tables
                )

                # Plots
                if not self.skip_plots:
                    # Heatmaps pro Algorithmus
                    for name, table in contingency_tables.items():
                        algorithm = name.split('_vs_')[0]
                        fig = self.gics_comparison.plot_contingency_heatmap(
                            contingency_table=table,
                            algorithm_name=algorithm,
                            gics_col=gics_col
                        )
                        self.comparison_handler.save_plot(
                            fig, f'{name}.png', 'gics_tables'
                        )

                    # Summary Plot
                    fig_summary = self.gics_comparison.plot_summary_comparison(
                        cramers_df=cramers_df,
                        analysis_type=stage
                    )
                    self.comparison_handler.save_plot(
                        fig_summary, f'summary_gics_{stage}.png', 'gics'
                    )

                # Speichere in Results
                self.comparison_results[f'gics_{stage}'] = {
                    'cramers_v': cramers_df,
                    'rand_index': rand_df,
                    'chi_square': chi2_df
                }

    def _run_algorithm_comparison(self):
        """Algorithm Comparison für Metriken und Überlappung"""
        logger.info(f"\n{'='*80}")
        logger.info("2. ALGORITHM COMPARISON")
        logger.info(f"{'='*80}")

        # Für jedes Stadium
        for stage in ['static', 'dynamic', 'combined']:
            logger.info(f"\n--- Algorithm Comparison: {stage.upper()} ---")

            # Sammle Metriken
            metrics_dict = {}
            results_dict = {}

            for algorithm, results in self.algorithm_results.items():
                if results[stage] and 'metrics' in results[stage]:
                    metrics_dict[algorithm] = results[stage]['metrics']
                    results_dict[algorithm] = results[stage]['df']

            if not metrics_dict:
                logger.warning(f"  ⚠ Keine Metriken für {stage}")
                continue

            # Metrics Comparison
            metrics_df = self.algo_comparison.compare_metrics(metrics_dict)

            # Cluster Overlap
            overlap_matrix = self.algo_comparison.compute_cluster_overlap(results_dict)

            # Speichere
            self.comparison_handler.save_algorithm_comparison(
                metrics=metrics_df,
                overlap_matrix=overlap_matrix
            )

            # Plots
            if not self.skip_plots:
                fig_metrics = self.algo_comparison.plot_metrics_comparison(
                    metrics_df=metrics_df,
                    analysis_type=stage
                )
                self.comparison_handler.save_plot(
                    fig_metrics, f'metrics_comparison_{stage}.png', 'algorithms'
                )

                fig_overlap = self.algo_comparison.plot_overlap_heatmap(
                    overlap_matrix=overlap_matrix,
                    analysis_type=stage
                )
                self.comparison_handler.save_plot(
                    fig_overlap, f'algorithm_overlap_{stage}.png', 'algorithms'
                )

            # Speichere in Results
            self.comparison_results[f'algorithm_{stage}'] = {
                'metrics': metrics_df,
                'overlap': overlap_matrix
            }

    def _run_feature_importance(self):
        """Feature Importance Analysis"""
        logger.info(f"\n{'='*80}")
        logger.info("3. FEATURE IMPORTANCE")
        logger.info(f"{'='*80}")

        # Hole Feature-Liste aus Config
        static_features = config.get_value(
            self.config, 'static_analysis', 'features', default=[]
        )
        dynamic_features = config.get_value(
            self.config, 'dynamic_analysis', 'features', default=[]
        )

        # Für jedes Stadium
        for stage in ['static', 'combined']:  # Dynamic hat zu viele aggregierte Features
            logger.info(f"\n--- Feature Importance: {stage.upper()} ---")

            # Wähle passende Features
            if stage == 'static':
                feature_cols = static_features
            else:
                feature_cols = static_features + dynamic_features

            # Sammle Daten
            results_dict = {}
            for algorithm, results in self.algorithm_results.items():
                if results[stage] and 'df' in results[stage]:
                    results_dict[algorithm] = results[stage]['df']

            if not results_dict:
                logger.warning(f"  ⚠ Keine Daten für {stage}")
                continue

            # Berechne Importance
            importance_dict = self.feature_importance.compute_all_algorithms(
                results_dict=results_dict,
                feature_cols=feature_cols,
                analysis_type=stage
            )

            if importance_dict:
                # Speichere
                self.comparison_handler.save_feature_importance(importance_dict)

                # Plots
                if not self.skip_plots:
                    # Pro Algorithmus
                    for algorithm, importance_df in importance_dict.items():
                        fig = self.feature_importance.plot_importance(
                            importance_df=importance_df,
                            algorithm_name=algorithm
                        )
                        self.comparison_handler.save_plot(
                            fig, f'{algorithm}_importance_{stage}.png', 'features'
                        )

                    # Combined Plot
                    fig_combined = self.feature_importance.plot_combined_importance(
                        importance_dict=importance_dict
                    )
                    self.comparison_handler.save_plot(
                        fig_combined, f'combined_importance_{stage}.png', 'features'
                    )

                # Speichere in Results
                self.comparison_results[f'importance_{stage}'] = importance_dict

    def _run_temporal_stability(self):
        """Temporal Stability Analysis"""
        logger.info(f"\n{'='*80}")
        logger.info("4. TEMPORAL STABILITY")
        logger.info(f"{'='*80}")

        # Verwende Dynamic-Daten (haben Zeitreihen)
        results_dict = {}
        for algorithm, results in self.algorithm_results.items():
            if results['dynamic']:
                # Verwende df_timeseries falls vorhanden (hat fyear), sonst df
                if 'df_timeseries' in results['dynamic']:
                    results_dict[algorithm] = results['dynamic']['df_timeseries']
                elif 'df' in results['dynamic']:
                    results_dict[algorithm] = results['dynamic']['df']

        if not results_dict:
            logger.warning("  ⚠ Keine Dynamic-Daten für Temporal Analysis")
            return

        # Analyze
        migration_matrices, stability_df = self.temporal_stability.analyze_all_algorithms(
            results_dict=results_dict,
            year_col='fyear',
            company_id_col='gvkey'
        )

        if migration_matrices and stability_df is not None:
            # Speichere
            self.comparison_handler.save_temporal_stability(
                migration_matrices=migration_matrices,
                stability_metrics=stability_df
            )

            # Plots
            if not self.skip_plots:
                # Migration Heatmaps
                for algorithm, matrix in migration_matrices.items():
                    fig_migration = self.temporal_stability.plot_migration_heatmap(
                        migration_matrix=matrix,
                        algorithm_name=algorithm
                    )
                    self.comparison_handler.save_plot(
                        fig_migration, f'{algorithm}_migration_heatmap.png', 'temporal'
                    )

                    # Year-to-year changes
                    df = results_dict[algorithm]
                    fig_yearly = self.temporal_stability.plot_year_to_year_changes(
                        df=df,
                        algorithm_name=algorithm
                    )
                    if fig_yearly:
                        self.comparison_handler.save_plot(
                            fig_yearly, f'{algorithm}_yearly_stability.png', 'temporal'
                        )

                # Stability Comparison
                fig_stability = self.temporal_stability.plot_stability_comparison(
                    stability_df=stability_df
                )
                self.comparison_handler.save_plot(
                    fig_stability, 'algorithm_stability_comparison.png', 'temporal'
                )

            # Speichere in Results
            self.comparison_results['temporal'] = {
                'migration_matrices': migration_matrices,
                'stability_metrics': stability_df
            }

    def _find_common_companies(
        self,
        df_static: pd.DataFrame,
        df_dynamic: pd.DataFrame,
        company_id_col: str = 'gvkey'
    ) -> pd.DataFrame:
        """Findet Unternehmen die in beiden DataFrames vorhanden sind"""
        if company_id_col not in df_static.columns or company_id_col not in df_dynamic.columns:
            return pd.DataFrame()

        common_ids = set(df_static[company_id_col]) & set(df_dynamic[company_id_col])
        df_common = df_static[df_static[company_id_col].isin(common_ids)].copy()

        return df_common

    def run_full_comparison_pipeline(
        self,
        df_all: pd.DataFrame,
        df_latest: pd.DataFrame
    ) -> Dict:
        """
        Führt komplette Comparison Pipeline aus

        Args:
            df_all: DataFrame mit allen Jahren
            df_latest: DataFrame mit aktuellem Jahr

        Returns:
            Dict mit allen Ergebnissen
        """
        start_time = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info("STARTING FULL COMPARISON PIPELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Market: {self.market}")
        logger.info(f"Algorithms: {', '.join(self.algorithms)}")

        # 1. Run all algorithms
        self.run_all_algorithms(df_all, df_latest)

        # 2. Run all comparisons
        self.run_comparisons()

        # Duration
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON PIPELINE COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"  ⏱️  Duration: {duration:.1f}s")
        logger.info(f"  📁 Output: {self.comparison_handler.base_dir}")

        return {
            'algorithm_results': self.algorithm_results,
            'comparison_results': self.comparison_results,
            'duration': duration
        }
