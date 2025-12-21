#!/usr/bin/env python3
"""
FMP Symbol List Generator
=========================

Holt Listen von Unternehmen aus verschiedenen Quellen:
- Indizes (S&P 500, NASDAQ, Dow Jones)
- Stock Screener (filtern nach Market Cap, Sektor, Land, etc.)
- Alle verfügbaren Aktien

Usage:
    # S&P 500 Unternehmen
    python fmp_get_symbols.py --source sp500 --output data/symbols/sp500.txt

    # NASDAQ Unternehmen
    python fmp_get_symbols.py --source nasdaq --output data/symbols/nasdaq.txt

    # Stock Screener: Deutsche Unternehmen mit Market Cap > 1B
    python fmp_get_symbols.py --source screener --country Germany --min-market-cap 1000000000

    # Alle aktiv gehandelten Aktien
    python fmp_get_symbols.py --source actively-trading
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict
import requests
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FMPSymbolFetcher:
    """Fetches company symbols from FMP API"""

    BASE_URL = "https://financialmodelingprep.com"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make API request with error handling"""
        if params is None:
            params = {}

        params['apikey'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            logger.info(f"Fetching: {endpoint}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Handle error responses
            if isinstance(data, dict) and 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []

    def get_sp500_constituents(self) -> List[str]:
        """Get S&P 500 companies"""
        data = self._make_request("stable/sp500-constituent")
        symbols = [item['symbol'] for item in data if 'symbol' in item]
        logger.info(f"Found {len(symbols)} S&P 500 companies")
        return symbols

    def get_nasdaq_constituents(self) -> List[str]:
        """Get NASDAQ companies"""
        data = self._make_request("stable/nasdaq-constituent")
        symbols = [item['symbol'] for item in data if 'symbol' in item]
        logger.info(f"Found {len(symbols)} NASDAQ companies")
        return symbols

    def get_dowjones_constituents(self) -> List[str]:
        """Get Dow Jones companies"""
        data = self._make_request("stable/dowjones-constituent")
        symbols = [item['symbol'] for item in data if 'symbol' in item]
        logger.info(f"Found {len(symbols)} Dow Jones companies")
        return symbols

    def get_actively_trading(self) -> List[str]:
        """Get all actively trading stocks"""
        data = self._make_request("stable/actively-trading-list")
        symbols = [item['symbol'] for item in data if 'symbol' in item]
        logger.info(f"Found {len(symbols)} actively trading stocks")
        return symbols

    def get_stock_list(self) -> List[str]:
        """Get complete stock list"""
        data = self._make_request("stable/stock-list")
        symbols = [item['symbol'] for item in data if 'symbol' in item]
        logger.info(f"Found {len(symbols)} stocks in complete list")
        return symbols

    def screen_stocks(
        self,
        market_cap_min: Optional[int] = None,
        market_cap_max: Optional[int] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        volume_min: Optional[int] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 1000
    ) -> List[str]:
        """
        Screen stocks by various criteria

        Examples:
            # Large cap US tech stocks
            screen_stocks(market_cap_min=10e9, sector='Technology', country='US')

            # German stocks
            screen_stocks(country='Germany', exchange='FRA')
        """
        params = {'limit': limit}

        if market_cap_min:
            params['marketCapMoreThan'] = market_cap_min
        if market_cap_max:
            params['marketCapLowerThan'] = market_cap_max
        if price_min:
            params['priceMoreThan'] = price_min
        if price_max:
            params['priceLowerThan'] = price_max
        if volume_min:
            params['volumeMoreThan'] = volume_min
        if sector:
            params['sector'] = sector
        if industry:
            params['industry'] = industry
        if country:
            params['country'] = country
        if exchange:
            params['exchange'] = exchange

        data = self._make_request("stable/company-screener", params)
        symbols = [item['symbol'] for item in data if 'symbol' in item]

        logger.info(f"Screener found {len(symbols)} stocks matching criteria")
        return symbols

    def get_available_sectors(self) -> List[str]:
        """Get list of available sectors"""
        data = self._make_request("stable/available-sectors")
        return data

    def get_available_countries(self) -> List[str]:
        """Get list of available countries"""
        data = self._make_request("stable/available-countries")
        return data

    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        data = self._make_request("stable/available-exchanges")
        return data


def save_symbols(symbols: List[str], output_path: str, verbose: bool = True):
    """Save symbols to file (one per line)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")

    if verbose:
        logger.info(f"✓ Saved {len(symbols)} symbols to {output_path}")
        logger.info(f"Preview: {', '.join(symbols[:10])}...")


def main():
    parser = argparse.ArgumentParser(
        description='Get company symbol lists from FMP API'
    )

    # Data source
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['sp500', 'nasdaq', 'dowjones', 'actively-trading', 'stock-list', 'screener'],
        help='Data source for symbols'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (e.g., data/symbols/sp500.txt)'
    )

    # Screener filters (only for --source screener)
    parser.add_argument('--min-market-cap', type=float, help='Minimum market cap')
    parser.add_argument('--max-market-cap', type=float, help='Maximum market cap')
    parser.add_argument('--min-price', type=float, help='Minimum stock price')
    parser.add_argument('--max-price', type=float, help='Maximum stock price')
    parser.add_argument('--min-volume', type=int, help='Minimum trading volume')
    parser.add_argument('--sector', type=str, help='Sector filter (e.g., Technology)')
    parser.add_argument('--industry', type=str, help='Industry filter')
    parser.add_argument('--country', type=str, help='Country filter (e.g., US, Germany)')
    parser.add_argument('--exchange', type=str, help='Exchange filter (e.g., NYSE, NASDAQ, FRA)')

    # API
    parser.add_argument(
        '--api-key',
        type=str,
        help='FMP API Key (or set FMP_API_KEY env variable)'
    )

    # Options
    parser.add_argument('--limit', type=int, default=1000, help='Max results for screener')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("No API key provided!")
        logger.error("Set FMP_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Create fetcher
    fetcher = FMPSymbolFetcher(api_key)

    # Get symbols based on source
    if args.source == 'sp500':
        symbols = fetcher.get_sp500_constituents()

    elif args.source == 'nasdaq':
        symbols = fetcher.get_nasdaq_constituents()

    elif args.source == 'dowjones':
        symbols = fetcher.get_dowjones_constituents()

    elif args.source == 'actively-trading':
        symbols = fetcher.get_actively_trading()

    elif args.source == 'stock-list':
        symbols = fetcher.get_stock_list()

    elif args.source == 'screener':
        symbols = fetcher.screen_stocks(
            market_cap_min=args.min_market_cap,
            market_cap_max=args.max_market_cap,
            price_min=args.min_price,
            price_max=args.max_price,
            volume_min=args.min_volume,
            sector=args.sector,
            industry=args.industry,
            country=args.country,
            exchange=args.exchange,
            limit=args.limit
        )

    else:
        logger.error(f"Unknown source: {args.source}")
        sys.exit(1)

    # Save symbols
    if symbols:
        save_symbols(symbols, args.output, verbose=True)

        # Show summary
        print("\n" + "="*50)
        print(f"✓ Successfully fetched {len(symbols)} symbols")
        print(f"✓ Saved to: {args.output}")
        print("="*50)
        print("\nNext steps:")
        print(f"  1. Review symbols: cat {args.output}")
        print(f"  2. Load data: python fmp_to_csv.py --symbols-file {args.output} --market your_market_name")
        print()
    else:
        logger.error("No symbols found!")
        sys.exit(1)


if __name__ == '__main__':
    main()
