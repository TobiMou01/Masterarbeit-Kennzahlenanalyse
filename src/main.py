"""
Main Entry Point
Simplified orchestration - delegates to Pipeline class
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, preprocessing
from src.pipeline import ClusteringPipeline
from src.comparison_pipeline import ComparisonPipeline

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def is_venv_active():
    """Check if a virtual environment is active"""
    return sys.prefix != sys.base_prefix


def check_venv_setup():
    """
    Interactive venv setup checker.
    If no venv is active, prompts user with options.
    Returns True to continue, False to exit.
    """
    if is_venv_active():
        # venv is active, continue silently
        return True

    # No venv active - show interactive dialog
    print("\n" + "=" * 80)
    print("⚠️  KEINE VIRTUELLE UMGEBUNG (venv) AKTIV!")
    print("=" * 80)
    print("\nOptionen:")
    print("[1] venv im Projekt-Ordner nutzen (./venv)")
    print("[2] Externe venv nutzen (/Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit)")
    print("[3] Dependencies prüfen/installieren")
    print("[4] Abbrechen")
    print()

    choice = input("Wahl [1/2/3/4]: ").strip()

    if choice == "1":
        # Check project venv
        venv_path = Path("venv")
        print()
        if venv_path.exists():
            print("✓ venv gefunden!")
            print("\nFühre folgenden Befehl aus:\n")
            print("    source venv/bin/activate")
            print("\nDanach starte das Skript erneut.")
        else:
            print("✗ venv nicht gefunden!")
            print("\nErstelle sie mit:\n")
            print("    python3 -m venv venv")
            print("    source venv/bin/activate")
            print("    pip install -r requirements.txt")
            print("\nDanach starte das Skript erneut.")
        print()
        return False

    elif choice == "2":
        # Check external venv
        external_venv = Path("/Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit")
        print()
        if external_venv.exists():
            print("✓ Externe venv gefunden!")
            print("\nFühre folgenden Befehl aus:\n")
            print("    source /Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit/bin/activate")
            print("\nDanach starte das Skript erneut.")
        else:
            print("✗ Externe venv nicht gefunden unter:")
            print(f"   {external_venv}")
            print("\nBitte prüfe den Pfad oder wähle Option [1].")
        print()
        return False

    elif choice == "3":
        # Show dependency info
        print()
        print("📦 Dependencies aus requirements.txt:")
        print()
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            print("Installiere alle mit:\n")
            print("    pip install -r requirements.txt")
            print()
            print("Oder einzeln prüfen:\n")
            print("    pip list | grep -E \"pandas|numpy|scikit-learn|matplotlib|seaborn|openpyxl|pyyaml|scipy|joblib\"")
            print()
            print("\nBenötigte Pakete:")
            with open(requirements_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        print(f"    - {line.strip()}")
        else:
            print("✗ requirements.txt nicht gefunden!")
        print()
        return False

    elif choice == "4":
        # Cancel
        print("\n✓ Abgebrochen.\n")
        return False

    else:
        print(f"\n✗ Ungültige Wahl: '{choice}'")
        print("Bitte wähle 1, 2, 3 oder 4.\n")
        return False


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


def parse_algorithms_from_config(cfg: dict) -> list:
    """
    Parse algorithm(s) from config.
    Supports single or multiple algorithms (comma-separated).

    Examples:
        'kmeans' -> ['kmeans']
        'kmeans, hierarchical, dbscan' -> ['kmeans', 'hierarchical', 'dbscan']

    Returns:
        List of algorithm names
    """
    algorithm_str = config.get_value(cfg, 'classification', 'algorithm', default='kmeans')

    if isinstance(algorithm_str, str):
        # Split by comma and strip whitespace
        algorithms = [alg.strip() for alg in algorithm_str.split(',')]
        # Filter out empty strings
        algorithms = [alg for alg in algorithms if alg]
        return algorithms
    elif isinstance(algorithm_str, list):
        # Already a list
        return algorithm_str
    else:
        # Fallback
        return ['kmeans']


def main():
    """Main entry point"""
    args = parse_args()

    # Check venv setup before starting pipeline
    if not check_venv_setup():
        # User chose to exit or needs to setup venv
        return 0

    logger.info("\n" + "=" * 80)
    logger.info("🚀 CLUSTERING ANALYSIS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Market: {args.market}")
    logger.info(f"Config: {args.config}\n")

    try:
        # Load config
        cfg = config.load_config(args.config)

        # Parse algorithms from config
        config_algorithms = parse_algorithms_from_config(cfg)

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

        return 0

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
