#!/usr/bin/env python3
"""
FMP to CSV Converter
====================

Eigenständiges Tool zum Laden von FMP API Daten und Speichern als WRDS-kompatible CSV.

Workflow:
1. Dieses Script ausführen → CSV erstellen
2. Deine normale Pipeline nutzen (UNVERÄNDERT!)

Usage:
    # Manual symbol list
    python fmp_to_csv.py --symbols AAPL MSFT --years 10 --market us_tech

    # From file (recommended for many symbols)
    python fmp_to_csv.py --symbols-file data/symbols/sp500.txt --market sp500 --years 10

    # Load S&P 500 (full workflow)
    python fmp_get_symbols.py --source sp500 --output data/symbols/sp500.txt
    python fmp_to_csv.py --symbols-file data/symbols/sp500.txt --market sp500
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src._02_preprocessing import fmp_data_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_symbols_from_file(file_path: str) -> list:
    """
    Read symbols from file (one per line)

    Args:
        file_path: Path to file with symbols

    Returns:
        List of symbols (stripped and filtered for empty lines)
    """
    symbols = []
    with open(file_path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):  # Skip empty lines and comments
                symbols.append(symbol)

    logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
    return symbols


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Load FMP API data and save as WRDS-compatible CSV'
    )

    # Input options
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of ticker symbols (e.g., AAPL MSFT JNJ)'
    )
    parser.add_argument(
        '--symbols-file',
        type=str,
        help='Path to file with symbols (one per line)'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Number of years of historical data (default: 10)'
    )
    parser.add_argument(
        '--market',
        type=str,
        required=True,
        help='Market name (used for output directory, e.g., "us_tech", "dax")'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Base output directory (default: data/raw)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='fmp_data.csv',
        help='Output filename (default: fmp_data.csv)'
    )

    # API options
    parser.add_argument(
        '--api-key',
        type=str,
        help='FMP API Key (or set FMP_API_KEY env variable)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between companies in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    # Get symbols from either --symbols or --symbols-file
    symbols = None

    if args.symbols_file:
        try:
            symbols = read_symbols_from_file(args.symbols_file)
        except FileNotFoundError:
            logger.error(f"Symbols file not found: {args.symbols_file}")
            return 1
        except Exception as e:
            logger.error(f"Failed to read symbols file: {e}")
            return 1
    elif args.symbols:
        symbols = args.symbols
    else:
        logger.error("No symbols provided!")
        logger.error("Use --symbols AAPL MSFT ... OR --symbols-file path/to/file.txt")
        return 1

    if not symbols:
        logger.error("Symbol list is empty!")
        return 1

    # Get API key
    api_key = args.api_key or os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("No API key provided!")
        logger.error("Set FMP_API_KEY environment variable or use --api-key")
        return 1

    logger.info("="*80)
    logger.info("FMP TO CSV CONVERTER")
    logger.info("="*80)
    logger.info(f"Market: {args.market}")
    logger.info(f"Companies: {len(symbols)}")
    if len(symbols) <= 10:
        logger.info(f"Symbols: {symbols}")
    else:
        logger.info(f"Symbols (first 10): {symbols[:10]} ...")
    logger.info(f"Years: {args.years}")
    logger.info(f"Output: {args.output_dir}/{args.market}/{args.filename}")
    logger.info("="*80)

    # Load data from FMP (already converts to WRDS format internally)
    logger.info("\n[STEP 1/2] Loading data from FMP API...")

    try:
        df_wrds = fmp_data_loader.fetch_multiple_companies(
            symbols=symbols,
            api_key=api_key,
            years=args.years,
            rate_limit_delay=args.rate_limit,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Failed to load data from FMP: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if df_wrds.empty:
        logger.error("No data loaded!")
        return 1

    logger.info(f"✓ Loaded and converted {len(df_wrds)} rows")
    logger.info(f"✓ Columns: {len(df_wrds.columns)}")

    # Save to CSV
    logger.info("\n[STEP 2/2] Saving to CSV...")

    output_path = Path(args.output_dir) / args.market / args.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_wrds.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUCCESS!")
    logger.info("="*80)
    logger.info(f"CSV created: {output_path}")
    logger.info(f"Rows: {len(df_wrds)}")
    logger.info(f"Columns: {len(df_wrds.columns)}")
    logger.info(f"Companies: {df_wrds['tic'].nunique() if 'tic' in df_wrds.columns else 'N/A'}")
    logger.info(f"Years: {df_wrds['fyear'].min()}-{df_wrds['fyear'].max()}" if 'fyear' in df_wrds.columns else "")
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Verify CSV manually (if needed):")
    logger.info(f"   head {output_path}")
    logger.info("")
    logger.info("2. Use with your EXISTING pipeline (NO CHANGES!):")
    logger.info(f"   python run.py --market {args.market}")
    logger.info("")
    logger.info("   OR:")
    logger.info(f"   from src._02_preprocessing import data_cleaner")
    logger.info(f"   df = data_cleaner.run_preprocessing(")
    logger.info(f"       input_dir='{args.output_dir}',")
    logger.info(f"       market='{args.market}'")
    logger.info(f"   )")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
