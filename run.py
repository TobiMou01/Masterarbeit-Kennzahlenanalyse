#!/usr/bin/env python3
"""
CLUSTERING ANALYSIS - COMMAND LINE RUNNER
==========================================

Führt komplette Clustering-Analyse basierend auf config.yaml durch.
Nutze dieses Script für reproduzierbare, konfigurierbare Analysen.

USAGE:
    # Standard-Analyse (nutzt config.yaml)
    python run.py

    # Mit custom config
    python run.py --config my_config.yaml

    # Nur bestimmte Analysen
    python run.py --analyses static combined

    # Nur K-Means
    python run.py --algorithms kmeans

    # Mit custom Preset
    python run.py --preset minimal

    # Override market
    python run.py --market germany --files dax40_proxy.csv mdax_proxy.csv

WICHTIG FÜR BESSERE ERGEBNISSE:
- Passe config.yaml an (feature_selection.preset, pca.enabled, etc.)
- Schließe Outlier aus (preprocessing.outlier_detection)
- Nutze PCA bei vielen Features (pca.enabled: true)
- Teste verschiedene Feature-Sets in features_config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src._01_setup import config_loader, environment
from src._02_preprocessing import data_loader, data_cleaner
from src._03_clustering import pipeline as clustering_pipeline
from src._04_comparison import comparison_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('clustering_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    cfg = config_loader.load_config(PROJECT_ROOT / config_path)
    return cfg


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config with command-line arguments."""

    if args.market:
        config_loader.set_value(cfg, 'data', 'market', args.market)

    if args.files:
        market = config_loader.get_value(cfg, 'data', 'market')
        if 'file_selection' not in cfg['data']:
            cfg['data']['file_selection'] = {}
        cfg['data']['file_selection'][market] = args.files

    if args.analyses:
        # Not directly in config, will be handled in pipeline
        pass

    if args.algorithms:
        algo_str = ', '.join(args.algorithms)
        config_loader.set_value(cfg, 'classification', 'algorithm', algo_str)

    if args.preset:
        config_loader.set_value(cfg, 'feature_selection', 'preset', args.preset)

    if args.n_clusters:
        # Override all n_clusters settings
        config_loader.set_value(cfg, 'static_analysis', 'n_clusters', args.n_clusters)
        config_loader.set_value(cfg, 'dynamic_analysis', 'n_clusters', args.n_clusters)
        config_loader.set_value(cfg, 'combined_analysis', 'n_clusters', args.n_clusters)

    return cfg


def run_analysis(cfg: dict, args: argparse.Namespace):
    """Run complete clustering analysis."""

    # Extract settings
    market = config_loader.get_value(cfg, 'data', 'market', default='germany')
    input_dir = PROJECT_ROOT / config_loader.get_value(cfg, 'data', 'input_dir', default='data/raw')
    output_dir = PROJECT_ROOT / config_loader.get_value(cfg, 'data', 'output_dir', default='output')

    logger.info("="*80)
    logger.info("CLUSTERING ANALYSIS - PRODUCTION RUN")
    logger.info("="*80)
    logger.info(f"Market: {market}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)

    # 1. LOAD DATA
    logger.info("\n📥 STEP 1/4: Loading Data...")

    file_selection = None
    if args.files:
        file_selection = args.files
    elif 'file_selection' in cfg.get('data', {}) and market in cfg['data']['file_selection']:
        file_selection = cfg['data']['file_selection'][market]

    df_raw = data_loader.load_market_data(
        market=market,
        data_dir=str(input_dir),
        file_selection=file_selection
    )

    logger.info(f"✓ Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
    if 'gvkey' in df_raw.columns:
        logger.info(f"✓ Companies: {df_raw['gvkey'].nunique()}")

    # 2. PREPROCESS
    logger.info("\n🔧 STEP 2/4: Preprocessing...")

    impute_enabled = config_loader.get_value(cfg, 'preprocessing', 'imputation', 'enabled', default=True)
    impute_method = config_loader.get_value(cfg, 'preprocessing', 'imputation', 'method', default='median')
    impute_threshold = config_loader.get_value(cfg, 'preprocessing', 'imputation', 'threshold', default=0.5)
    smooth_static = config_loader.get_value(cfg, 'preprocessing', 'cagr_smoothing', 'enabled', default=False)
    cagr_years = config_loader.get_value(cfg, 'preprocessing', 'cagr_smoothing', 'years', default=3)

    df_features = data_cleaner.run_preprocessing(
        input_dir=str(input_dir),
        market=market,
        impute=impute_enabled,
        impute_method=impute_method,
        impute_threshold=impute_threshold,
        smooth_static=smooth_static,
        cagr_years=cagr_years,
        df_raw=df_raw
    )

    logger.info(f"✓ Processed {len(df_features):,} rows, {len(df_features.columns)} features")

    # 3. CLUSTERING
    logger.info("\n🔬 STEP 3/4: Clustering Analysis...")

    algorithms = args.algorithms if args.algorithms else ['kmeans']
    analyses = args.analyses if args.analyses else ['static', 'dynamic', 'combined']

    logger.info(f"Algorithms: {', '.join(algorithms)}")
    logger.info(f"Analyses:   {', '.join(analyses)}")

    # Run clustering pipeline for each algorithm and analysis type
    from src._03_clustering.cluster_engine import ClusterEngine
    from src._01_setup.output_coordinator import OutputHandler

    results = {}

    for algorithm in algorithms:
        logger.info(f"\n  🔹 Running {algorithm.upper()}...")

        # Create output handler
        output_handler = OutputHandler(
            market=market,
            algorithm=algorithm,
            mode='auto',
            base_dir=str(output_dir)
        )

        algo_results = {}

        for analysis_type in analyses:
            logger.info(f"    → {analysis_type} analysis...")

            try:
                # Get features for this analysis type
                if analysis_type == 'static':
                    features = config_loader.get_value(cfg, 'static_analysis', 'features', default=[])
                elif analysis_type == 'dynamic':
                    features = config_loader.get_value(cfg, 'dynamic_analysis', 'features', default=[])
                else:  # combined
                    features_static = config_loader.get_value(cfg, 'combined_analysis', 'features_static', default=[])
                    features_dynamic = config_loader.get_value(cfg, 'combined_analysis', 'features_dynamic', default=[])
                    features = features_static + features_dynamic

                # Filter available features
                available_features = [f for f in features if f in df_features.columns]

                if not available_features:
                    logger.warning(f"      ⚠️  No features available for {analysis_type}, skipping...")
                    continue

                logger.info(f"      Features: {len(available_features)} available")

                # Create cluster engine
                engine = ClusterEngine(
                    algorithm=algorithm,
                    config=cfg,
                    output_handler=output_handler
                )

                # Run clustering
                cluster_result = engine.fit_predict(
                    X=df_features[available_features],
                    df_full=df_features,
                    analysis_type=analysis_type
                )

                algo_results[analysis_type] = cluster_result

                # Save results
                output_handler.save_cluster_results(
                    df=df_features,
                    cluster_labels=cluster_result['labels'],
                    analysis_type=analysis_type,
                    algorithm=algorithm
                )

                logger.info(f"      ✓ {cluster_result['n_clusters']} clusters created")

            except Exception as e:
                logger.error(f"      ✗ Error in {analysis_type}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())

        results[algorithm] = algo_results

    # 4. COMPARISON (optional)
    if len(algorithms) > 1 and not args.no_comparison:
        logger.info("\n📊 STEP 4/4: Algorithm Comparison...")

        try:
            # Run comparison pipeline
            comparison_pipeline.run_comparison(
                results=results,
                df=df_features,
                config=cfg,
                market=market,
                output_dir=output_dir
            )
            logger.info("✓ Comparison complete")
        except Exception as e:
            logger.error(f"✗ Comparison failed: {str(e)}")

    # SUMMARY
    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"📁 Output: {output_dir / market}")
    logger.info(f"📊 Algorithms: {len(algorithms)}")
    logger.info(f"📈 Analyses: {len(analyses)}")
    logger.info("="*80)

    return results


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Run clustering analysis with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    # Market settings
    parser.add_argument(
        '--market', '-m',
        type=str,
        help='Market to analyze (overrides config.yaml)'
    )

    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='Specific CSV files to load (e.g., dax40_proxy.csv mdax_proxy.csv)'
    )

    # Analysis settings
    parser.add_argument(
        '--analyses', '-a',
        nargs='+',
        choices=['static', 'dynamic', 'combined'],
        help='Which analyses to run (default: all)'
    )

    parser.add_argument(
        '--algorithms', '-A',
        nargs='+',
        choices=['kmeans', 'hierarchical', 'dbscan'],
        help='Which algorithms to run (default: kmeans)'
    )

    # Feature settings
    parser.add_argument(
        '--preset', '-p',
        type=str,
        help='Feature preset from features_config.yaml (e.g., minimal, standard, extended)'
    )

    parser.add_argument(
        '--n-clusters', '-k',
        type=int,
        help='Number of clusters (overrides auto-selection)'
    )

    # Output settings
    parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Skip algorithm comparison (faster)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running analysis'
    )

    args = parser.parse_args()

    try:
        # Load config
        cfg = load_config(args.config)

        # Override with CLI args
        cfg = override_config(cfg, args)

        # Dry run?
        if args.dry_run:
            logger.info("DRY RUN MODE - Configuration:")
            logger.info(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
            return 0

        # Check environment
        environment.check_environment()

        # Run analysis
        results = run_analysis(cfg, args)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Analysis interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
