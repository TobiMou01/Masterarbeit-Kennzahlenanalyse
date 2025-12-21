#!/usr/bin/env python3
"""
Wikipedia Symbol Scraper
========================

Holt Ticker-Symbole von Wikipedia für große Indizes.
100% KOSTENLOS - keine API benötigt!

Verfügbar:
- S&P 500 (US)
- NASDAQ-100 (US Tech)
- Dow Jones (US)
- DAX (Deutschland)
- FTSE 100 (UK)
- CAC 40 (Frankreich)
- EURO STOXX 50 (Europa)

Usage:
    python get_symbols_from_wikipedia.py --index sp500 --output data/symbols/sp500.txt
    python get_symbols_from_wikipedia.py --index dax --output data/symbols/dax.txt
    python get_symbols_from_wikipedia.py --index all --output-dir data/symbols/
"""

import argparse
import sys
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Index Konfiguration
INDEXES = {
    'sp500': {
        'name': 'S&P 500',
        'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'table_index': 0,
        'symbol_column': 'Symbol',
        'suffix': '',  # US Symbole ohne Suffix
    },
    'nasdaq100': {
        'name': 'NASDAQ-100',
        'url': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'table_index': 4,  # Die "Components" Tabelle
        'symbol_column': 'Ticker',
        'suffix': '',
    },
    'dowjones': {
        'name': 'Dow Jones',
        'url': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
        'table_index': 1,
        'symbol_column': 'Symbol',
        'suffix': '',
    },
    'dax': {
        'name': 'DAX 40',
        'url': 'https://en.wikipedia.org/wiki/DAX',
        'table_index': 2,
        'symbol_column': 'Ticker',
        'suffix': '.DE',  # Deutsche Börse Suffix für FMP
    },
    'ftse100': {
        'name': 'FTSE 100',
        'url': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
        'table_index': 3,
        'symbol_column': 'Ticker',
        'suffix': '.L',  # London Stock Exchange
    },
    'cac40': {
        'name': 'CAC 40',
        'url': 'https://en.wikipedia.org/wiki/CAC_40',
        'table_index': 1,
        'symbol_column': 'Ticker',
        'suffix': '.PA',  # Paris Stock Exchange
    },
    'eurostoxx50': {
        'name': 'EURO STOXX 50',
        'url': 'https://en.wikipedia.org/wiki/EURO_STOXX_50',
        'table_index': 1,
        'symbol_column': 'Ticker',
        'suffix': '',  # Mixed, meist ohne Suffix
    }
}


def fetch_symbols_from_wikipedia(index_key: str) -> list:
    """
    Holt Ticker-Symbole von Wikipedia

    Args:
        index_key: Key aus INDEXES dict (z.B. 'sp500', 'dax')

    Returns:
        Liste von Ticker-Symbolen
    """
    if index_key not in INDEXES:
        raise ValueError(f"Unknown index: {index_key}. Available: {list(INDEXES.keys())}")

    config = INDEXES[index_key]

    logger.info(f"Fetching {config['name']} from Wikipedia...")
    logger.info(f"URL: {config['url']}")

    try:
        # Fetch HTML with User-Agent (Wikipedia requires this)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(config['url'], headers=headers, timeout=30)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the right table
        tables = soup.find_all('table', {'class': 'wikitable'})

        if len(tables) <= config['table_index']:
            raise ValueError(f"Table index {config['table_index']} not found. Found {len(tables)} tables.")

        table = tables[config['table_index']]

        # Extract symbols
        symbols = []
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if not cells:
                continue

            # Try to find symbol column by header name
            # Get headers
            headers = [th.get_text(strip=True) for th in table.find_all('tr')[0].find_all('th')]

            try:
                symbol_col_index = headers.index(config['symbol_column'])
            except ValueError:
                # Fallback: meist erste Spalte
                symbol_col_index = 0

            if len(cells) > symbol_col_index:
                symbol = cells[symbol_col_index].get_text(strip=True)

                # Clean symbol
                symbol = symbol.replace('\n', '').strip()

                # Remove special characters aber behalte Punkt
                symbol = re.sub(r'[^\w\.\-]', '', symbol)

                if symbol:
                    # Add suffix if needed
                    if config['suffix'] and not symbol.endswith(config['suffix']):
                        symbol = symbol + config['suffix']

                    symbols.append(symbol)

        logger.info(f"✓ Found {len(symbols)} symbols for {config['name']}")

        if symbols:
            logger.info(f"Preview: {', '.join(symbols[:5])}...")

        return symbols

    except Exception as e:
        logger.error(f"Failed to fetch {config['name']}: {e}")
        return []


def save_symbols(symbols: list, output_path: str):
    """Speichert Symbole in Datei (eins pro Zeile)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")

    logger.info(f"✓ Saved {len(symbols)} symbols to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Get ticker symbols from Wikipedia (FREE!)'
    )

    parser.add_argument(
        '--index',
        type=str,
        required=True,
        choices=list(INDEXES.keys()) + ['all'],
        help='Index to fetch (or "all" for all indexes)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (required if not using --output-dir with "all")'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (for --index all)'
    )

    args = parser.parse_args()

    # Fetch symbols
    if args.index == 'all':
        if not args.output_dir:
            logger.error("--output-dir required when using --index all")
            return 1

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*60)
        logger.info("Fetching ALL indexes...")
        logger.info("="*60)

        for index_key in INDEXES.keys():
            symbols = fetch_symbols_from_wikipedia(index_key)
            if symbols:
                output_file = output_dir / f"{index_key}.txt"
                save_symbols(symbols, output_file)
            print()  # Spacing

        logger.info("="*60)
        logger.info("✓ Done! All symbol lists saved.")
        logger.info("="*60)

    else:
        if not args.output:
            logger.error("--output required when fetching single index")
            return 1

        symbols = fetch_symbols_from_wikipedia(args.index)

        if not symbols:
            logger.error("No symbols found!")
            return 1

        save_symbols(symbols, args.output)

        print("\n" + "="*60)
        print(f"✓ Successfully fetched {len(symbols)} symbols")
        print(f"✓ Saved to: {args.output}")
        print("="*60)
        print("\nNext step:")
        print(f"  python fmp_to_csv.py --symbols-file {args.output} --market {args.index} --years 10")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
