"""
Main Entry Point with Interactive Mode
Simplified orchestration with user-friendly terminal interface
"""

import sys
from pathlib import Path

# ============================================================================
# VENV CHECK - MUST BE FIRST, BEFORE ANY OTHER IMPORTS!
# ============================================================================

# Add project root to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

from src._01_setup.environment import check_environment

# Check venv BEFORE importing anything that needs packages
check_environment()

# ============================================================================
# NOW SAFE TO IMPORT - venv is guaranteed to be active
# ============================================================================

import argparse
import logging
import pandas as pd

from src._01_setup import config_loader
from src._01_setup.interactive_menu import InteractiveMenu
from src._02_processing import data_cleaner as preprocessing
from src._03_clustering.pipeline import ClusteringPipeline
from src._03_clustering.hierarchical_pipeline import HierarchicalPipeline
from src._04_comparison.comparison_pipeline import ComparisonPipeline

# Import config module functions
config = config_loader

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3-Stage Clustering Analysis')

    parser.add_argument('--market', type=str, default='germany',
                        help='Market (germany, usa, etc.)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file')
    parser.add_argument('--skip-prep', action='store_true',
                        help='Skip preprocessing')
    parser.add_argument('--only-static', action='store_true',
                        help='Run only static analysis')
    parser.add_argument('--only-dynamic', action='store_true',
                        help='Run only dynamic analysis')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip visualizations')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison analysis across all algorithms')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['kmeans', 'hierarchical', 'dbscan'],
                        help='Algorithms to compare (default: kmeans hierarchical dbscan)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable interactive mode (for automated runs)')

    return parser.parse_args()


def is_interactive_mode(args) -> bool:
    """
    Check if interactive mode should be enabled

    Interactive mode is enabled when:
    - No command-line arguments provided (just running python src/main.py)
    - --quiet flag NOT set

    Returns:
        True if interactive mode should be used
    """
    # Check if any non-default args were provided
    has_args = (
        args.market != 'germany' or
        args.config != 'config.yaml' or
        args.skip_prep or
        args.only_static or
        args.only_dynamic or
        args.skip_plots or
        args.compare or
        args.algorithms != ['kmeans', 'hierarchical', 'dbscan'] or
        args.quiet
    )

    return not has_args


def main():
    """Main entry point"""
    args = parse_args()

    try:
        # Load config
        cfg = config.load_config(args.config)

        # Check if interactive mode
        if is_interactive_mode(args):
            # Run interactive menu
            menu = InteractiveMenu(cfg)
            session_config = menu.run()

            # Apply session config to args
            args.market = cfg.get('data', {}).get('market', 'germany')
            args.algorithms = session_config.get('algorithms', ['kmeans'])
            args.skip_prep = session_config.get('skip_preprocessing', False)
            args.skip_plots = not session_config.get('create_plots', False)
            args.compare = session_config.get('comparison_mode', False)

            # Store session config for pipeline
            analyses = session_config.get('analyses', ['static'])
            kmeans_mode = session_config.get('kmeans_mode', 'comparative')
            hierarchical_mode = session_config.get('hierarchical_mode', 'hierarchical')
            dbscan_mode = session_config.get('dbscan_mode', 'hierarchical')
            auto_tune_dbscan = session_config.get('auto_tune_dbscan', False)

        else:
            # Direct mode (command-line args)
            logger.info("\n" + "=" * 80)
            logger.info("🚀 CLUSTERING ANALYSIS PIPELINE")
            logger.info("=" * 80)
            logger.info(f"Market: {args.market}")
            logger.info(f"Config: {args.config}\n")

            # Parse algorithms from config if not specified
            if args.algorithms == ['kmeans', 'hierarchical', 'dbscan']:
                config_algorithms = config.parse_algorithms_from_config(cfg)
                if config_algorithms:
                    args.algorithms = config_algorithms

            # Auto-detect analyses
            if args.only_static:
                analyses = ['static']
            elif args.only_dynamic:
                analyses = ['dynamic']
            else:
                analyses = ['static', 'dynamic', 'combined']

            # Default modes
            kmeans_mode = 'comparative'
            hierarchical_mode = 'hierarchical'
            dbscan_mode = 'hierarchical'
            auto_tune_dbscan = False

        # Preprocessing
        if not args.skip_prep:
            input_dir = config.get_value(cfg, 'data', 'input_dir', default='data/raw')
            df_features = preprocessing.run_preprocessing(input_dir, args.market)
        else:
            logger.info("⏭️  Skipping preprocessing")
            df_features = preprocessing.load_processed_data(args.market)

        # Prepare time data
        df_all, df_latest = preprocessing.prepare_time_data(df_features, args.market)

        # Run analyses
        if args.compare or len(args.algorithms) > 1:
            # Comparison mode (multiple algorithms)
            run_comparison_mode(
                cfg=cfg,
                market=args.market,
                algorithms=args.algorithms,
                analyses=analyses,
                df_all=df_all,
                df_latest=df_latest,
                skip_plots=args.skip_plots,
                interactive=is_interactive_mode(args),
                kmeans_mode=kmeans_mode,
                hierarchical_mode=hierarchical_mode,
                dbscan_mode=dbscan_mode,
                auto_tune_dbscan=auto_tune_dbscan
            )
        else:
            # Single algorithm mode
            run_single_algorithm_mode(
                cfg=cfg,
                market=args.market,
                algorithm=args.algorithms[0],
                analyses=analyses,
                df_all=df_all,
                df_latest=df_latest,
                skip_plots=args.skip_plots,
                interactive=is_interactive_mode(args),
                kmeans_mode=kmeans_mode,
                hierarchical_mode=hierarchical_mode,
                dbscan_mode=dbscan_mode,
                auto_tune_dbscan=auto_tune_dbscan
            )

        return 0

    except KeyboardInterrupt:
        logger.info("\n\n👋 Pipeline abgebrochen durch Benutzer.")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


def run_single_algorithm_mode(cfg, market, algorithm, analyses, df_all, df_latest,
                               skip_plots, interactive, kmeans_mode, hierarchical_mode,
                               dbscan_mode, auto_tune_dbscan):
    """Run single algorithm analysis"""

    # Determine which mode to use for this algorithm
    if algorithm == 'kmeans':
        mode = kmeans_mode
    elif algorithm == 'hierarchical':
        mode = hierarchical_mode
    elif algorithm == 'dbscan':
        mode = dbscan_mode
    else:
        mode = 'comparative'  # Default

    logger.info(f"\n🔬 Running {algorithm.upper()} analysis")
    logger.info(f"   Mode: {mode.capitalize()}")
    logger.info(f"   Analyses: {' → '.join([a.capitalize() for a in analyses])}\n")

    # Override algorithm in config for this run
    cfg_copy = cfg.copy()
    if 'classification' not in cfg_copy:
        cfg_copy['classification'] = {}
    cfg_copy['classification']['algorithm'] = algorithm

    # Choose pipeline based on mode
    if mode == 'hierarchical':
        # Use Hierarchical Pipeline (Alternative 1)
        pipeline = HierarchicalPipeline(
            config_dict=cfg_copy,
            market=market,
            skip_plots=skip_plots
        )

        # Run full analysis (handles staging internally)
        pipeline.run_analysis(
            df_all=df_all,
            df_latest=df_latest,
            run_static='static' in analyses,
            run_dynamic='dynamic' in analyses or 'combined' in analyses
        )

        # Pause after each stage if interactive
        if interactive:
            if 'static' in analyses:
                output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['static']
                InteractiveMenu.pause_for_review('Static', output_dir)
            if 'dynamic' in analyses:
                output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['dynamic']
                InteractiveMenu.pause_for_review('Dynamic', output_dir)
            if 'combined' in analyses:
                output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['combined']
                InteractiveMenu.pause_for_review('Combined', output_dir)

    elif mode == 'both':
        # K-Means: Run BOTH modes and compare
        logger.info("→ Running BOTH modes for comparison\n")

        # Mode 1: Comparative
        logger.info("\n" + "="*80)
        logger.info("MODE 1: COMPARATIVE (3 separate clusterings)")
        logger.info("="*80)

        pipeline_comp = ClusteringPipeline(
            config_dict=cfg_copy,
            market=market + '_comparative',
            skip_plots=skip_plots
        )

        pipeline_comp.run_analysis(
            df_all=df_all,
            df_latest=df_latest,
            run_static='static' in analyses,
            run_dynamic='dynamic' in analyses or 'combined' in analyses
        )

        # Mode 2: Hierarchical
        logger.info("\n" + "="*80)
        logger.info("MODE 2: HIERARCHICAL (consistent labels)")
        logger.info("="*80)

        pipeline_hier = HierarchicalPipeline(
            config_dict=cfg_copy,
            market=market + '_hierarchical',
            skip_plots=skip_plots
        )

        pipeline_hier.run_analysis(
            df_all=df_all,
            df_latest=df_latest,
            run_static='static' in analyses,
            run_dynamic='dynamic' in analyses or 'combined' in analyses
        )

        # Compare results
        logger.info("\n" + "="*80)
        logger.info("COMPARISON: Comparative vs. Hierarchical")
        logger.info("="*80)

        comp_static = pipeline_comp.results.get('static', {})
        hier_static = pipeline_hier.results.get('static', {})

        logger.info(f"\nComparative - Silhouette: {comp_static.get('metrics', {}).get('silhouette_score', 'N/A')}")
        logger.info(f"Hierarchical - Static Score Avg: {hier_static.get('df', pd.DataFrame()).get('static_score', pd.Series()).mean():.1f}" if not hier_static.get('df', pd.DataFrame()).empty else "N/A")

        if interactive:
            logger.info(f"\n📂 Outputs:")
            logger.info(f"   Comparative:  {pipeline_comp.output.algorithm_dir}/")
            logger.info(f"   Hierarchical: {pipeline_hier.output.algorithm_dir}/")
            InteractiveMenu.pause_for_review('Both Modes', Path(f'output/{market}/'))

    else:
        # Use standard Comparative Pipeline
        pipeline = ClusteringPipeline(
            config_dict=cfg_copy,
            market=market,
            skip_plots=skip_plots
        )

        # Run analyses stage by stage
        for stage in analyses:
            if stage == 'static':
                pipeline._run_static_analysis(df_latest)
                if interactive:
                    output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['static']
                    InteractiveMenu.pause_for_review('Static', output_dir)

            elif stage == 'dynamic':
                pipeline._run_dynamic_analysis(df_all)
                if interactive:
                    output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['dynamic']
                    InteractiveMenu.pause_for_review('Dynamic', output_dir)

            elif stage == 'combined':
                df_static = pipeline.results.get('static', {}).get('df')
                df_dynamic = pipeline.results.get('dynamic', {}).get('df')
                if df_static is not None and df_dynamic is not None:
                    pipeline._run_combined_analysis(df_static, df_dynamic)
                    if interactive:
                        output_dir = pipeline.output.algorithm_dir / pipeline.output.analysis_types['combined']
                        InteractiveMenu.pause_for_review('Combined', output_dir)

        # Print final summary
        pipeline._print_summary()


def run_comparison_mode(cfg, market, algorithms, analyses, df_all, df_latest,
                        skip_plots, interactive, kmeans_mode, hierarchical_mode,
                        dbscan_mode, auto_tune_dbscan):
    """Run comparison analysis across multiple algorithms"""
    logger.info("\n🔬 Running COMPARISON ANALYSIS mode")
    logger.info(f"   Algorithms: {', '.join(algorithms)}")
    logger.info(f"   Analyses: {' → '.join([a.capitalize() for a in analyses])}\n")

    comparison_pipeline = ComparisonPipeline(
        config_dict=cfg,
        market=market,
        algorithms=algorithms,
        skip_plots=skip_plots
    )

    results = comparison_pipeline.run_full_comparison_pipeline(
        df_all=df_all,
        df_latest=df_latest
    )

    if interactive:
        output_dir = Path(f'output/{market}/comparisons')
        InteractiveMenu.pause_for_review('Comparison', output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Comparison Analysis complete")
    logger.info(f"✓ Algorithms compared: {', '.join(algorithms)}")
    logger.info(f"✓ Duration: {results['duration']:.1f}s")
    logger.info(f"\n📂 Output: output/{market}/comparisons/")
    logger.info("   ├── 01_gics_comparison/")
    logger.info("   ├── 02_algorithm_comparison/")
    logger.info("   ├── 03_feature_importance/")
    logger.info("   └── 04_temporal_stability/")
    logger.info("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
