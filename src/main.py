"""
Main Entry Point
Simplified orchestration - delegates to Pipeline class
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

from src._01_setup import config_loader
from src._02_processing import data_cleaner as preprocessing
from src.pipeline import ClusteringPipeline
from src.comparison_pipeline import ComparisonPipeline

# Import config module functions
config = config_loader

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("🚀 CLUSTERING ANALYSIS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Market: {args.market}")
    logger.info(f"Config: {args.config}\n")

    try:
        # Load config
        cfg = config.load_config(args.config)

        # Parse algorithms from config
        config_algorithms = config.parse_algorithms_from_config(cfg)

        # Auto-detect compare mode if multiple algorithms in config
        auto_compare = len(config_algorithms) > 1

        if auto_compare and not args.compare:
            logger.info(f"📋 Multiple algorithms detected in config: {', '.join(config_algorithms)}")
            logger.info("🔬 Automatically enabling COMPARISON MODE\n")
            args.compare = True
            args.algorithms = config_algorithms

        # Preprocessing
        if not args.skip_prep:
            input_dir = config.get_value(cfg, 'data', 'input_dir', default='data/raw')
            df_features = preprocessing.run_preprocessing(input_dir, args.market)
        else:
            logger.info("⏭️  Skipping preprocessing")
            df_features = preprocessing.load_processed_data(args.market)

        # Prepare time data
        df_all, df_latest = preprocessing.prepare_time_data(df_features, args.market)

        # Check if comparison mode
        if args.compare:
            # Run comparison pipeline (all algorithms)
            logger.info("\n🔬 Running COMPARISON ANALYSIS mode")
            logger.info(f"   Algorithms: {', '.join(args.algorithms)}\n")

            comparison_pipeline = ComparisonPipeline(
                config_dict=cfg,
                market=args.market,
                algorithms=args.algorithms,
                skip_plots=args.skip_plots
            )

            results = comparison_pipeline.run_full_comparison_pipeline(
                df_all=df_all,
                df_latest=df_latest
            )

            logger.info("\n" + "=" * 80)
            logger.info("✓ Comparison Analysis complete")
            logger.info(f"✓ Algorithms compared: {', '.join(args.algorithms)}")
            logger.info(f"✓ Duration: {results['duration']:.1f}s")
            logger.info(f"\n📂 Output: output/{args.market}/comparisons/")
            logger.info("   ├── 01_gics_comparison/")
            logger.info("   ├── 02_algorithm_comparison/")
            logger.info("   ├── 03_feature_importance/")
            logger.info("   └── 04_temporal_stability/")
            logger.info("=" * 80)
        else:
            # Run single algorithm pipeline
            pipeline = ClusteringPipeline(
                config_dict=cfg,
                market=args.market,
                skip_plots=args.skip_plots
            )

            results = pipeline.run_analysis(
                df_all=df_all,
                df_latest=df_latest,
                run_static=not args.only_dynamic,
                run_dynamic=not args.only_static
            )

        # Note: reorganize_output functionality moved to _archive/reorganization_utils.py
        # Can be re-added if needed

        return 0

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
