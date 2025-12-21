"""
FMP Data Loader
Lädt Finanzdaten von Financial Modeling Prep API und konvertiert sie in WRDS-kompatibles Format

Financial Modeling Prep API (Stable API - for new Free Tier users):
- Income Statement: /stable/income-statement?symbol={symbol}&limit={years}
- Balance Sheet: /stable/balance-sheet-statement?symbol={symbol}&limit={years}
- Cash Flow: /stable/cash-flow-statement?symbol={symbol}&limit={years}
- Company Profile: /stable/profile?symbol={symbol}

IMPORTANT: New API keys use the "Stable API" endpoints (/stable/...)
Legacy endpoints (/api/v3/...) are only available for accounts created before Aug 31, 2025

Rate Limits (Free Tier):
- 250 Requests/Tag
- 5 Requests/Minute

Usage:
    # Einzelnes Unternehmen
    df = fetch_company_data('AAPL', api_key='YOUR_KEY', years=10)
    df_wrds = convert_to_wrds_format(df)

    # Mehrere Unternehmen
    symbols = ['AAPL', 'MSFT', 'JNJ']
    df = fetch_multiple_companies(symbols, api_key='YOUR_KEY')

    # Integration mit bestehender Pipeline
    from src._02_preprocessing import data_cleaner
    df_features = data_cleaner.run_preprocessing(..., df_raw=df_wrds)
"""

import pandas as pd
import numpy as np
import requests
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# FMP → WRDS COLUMN MAPPING
# ============================================================================

FMP_TO_WRDS_MAPPING = {
    # ===== INCOME STATEMENT =====
    'revenue': 'revt',                              # Revenue Total
    'costOfRevenue': 'cogs',                        # Cost of Goods Sold
    'operatingIncome': 'ebit',                      # EBIT (Operating Income)
    'ebitda': 'ebitda',                             # EBITDA
    'netIncome': 'ni',                              # Net Income
    'incomeBeforeTax': 'ib',                        # Income Before Tax
    'interestExpense': 'xint',                      # Interest Expense
    'researchAndDevelopmentExpenses': 'xrd',        # R&D Expenses
    'sellingGeneralAndAdministrativeExpenses': 'xsga',  # SG&A
    'operatingIncomeBeforeDepreciation': 'oibdp',   # Operating Income Before Depreciation

    # ===== BALANCE SHEET - ASSETS =====
    'totalAssets': 'at',                            # Total Assets
    'totalCurrentAssets': 'act',                    # Current Assets
    'cashAndCashEquivalents': 'che',                # Cash & Equivalents
    'netReceivables': 'rect',                       # Receivables
    'inventory': 'invt',                            # Inventory
    'propertyPlantEquipmentNet': 'ppent',           # PP&E Net
    'shortTermInvestments': 'ivst',                 # Short-term Investments

    # ===== BALANCE SHEET - LIABILITIES & EQUITY =====
    'totalLiabilities': 'lt',                       # Total Liabilities
    'totalCurrentLiabilities': 'lct',               # Current Liabilities
    'shortTermDebt': 'dlc',                         # Debt in Current Liabilities
    'longTermDebt': 'dltt',                         # Long-term Debt
    'totalStockholdersEquity': 'seq',               # Stockholders Equity
    'commonStock': 'ceq',                           # Common Equity (approximation)

    # ===== CASH FLOW STATEMENT =====
    'operatingCashFlow': 'oancf',                   # Operating Cash Flow
    'capitalExpenditure': 'capx',                   # CapEx (usually negative in FMP)
    'depreciationAndAmortization': 'dp',            # Depreciation & Amortization
    'dividendsPaid': 'dvt',                         # Dividends (usually negative in FMP)

    # ===== COMPANY PROFILE (Meta-Data) =====
    'symbol': 'tic',                                # Ticker Symbol
    'companyName': 'conm',                          # Company Name
    'country': 'loc',                               # Location/Country
    'sector': 'sector_fmp',                         # FMP Sector (not GICS!)
    'industry': 'industry_fmp',                     # FMP Industry
    'isin': 'isin',                                 # ISIN (if available)
}

# Felder die in FMP NICHT direkt verfügbar sind (müssen berechnet oder approximiert werden)
MISSING_WRDS_FIELDS = {
    'gvkey': 'symbol',           # WRDS Company ID → nutze Symbol als Ersatz
    'gsector': None,             # GICS Sector → nicht verfügbar, muss gemappt werden
    'ggroup': None,              # GICS Group → nicht verfügbar
    'gind': None,                # GICS Industry → nicht verfügbar
    'gsubind': None,             # GICS Sub-Industry → nicht verfügbar
    'sale': 'revt',              # Sales = Revenue (redundant in WRDS)
}

# FMP Sector → GICS Sector Mapping (approximativ, da GICS sehr granular ist)
FMP_SECTOR_TO_GICS = {
    'Technology': 45,            # Information Technology
    'Healthcare': 35,            # Health Care
    'Financial Services': 40,    # Financials
    'Consumer Cyclical': 25,     # Consumer Discretionary
    'Industrials': 20,           # Industrials
    'Communication Services': 50,# Communication Services
    'Consumer Defensive': 30,    # Consumer Staples
    'Energy': 10,                # Energy
    'Real Estate': 60,           # Real Estate
    'Materials': 15,             # Materials
    'Utilities': 55,             # Utilities
}


# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def _make_api_request(url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
    """
    Führt API-Request mit Error-Handling und Retries durch.

    Args:
        url: API Endpoint URL
        params: Query-Parameter
        max_retries: Anzahl Wiederholungen bei Fehlern

    Returns:
        JSON Response als dict oder None bei Fehler
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes

            data = response.json()

            # FMP gibt manchmal {"Error Message": "..."} zurück
            if isinstance(data, dict) and 'Error Message' in data:
                logger.error(f"FMP API Error: {data['Error Message']}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.warning(f"API Request failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"API Request failed after {max_retries} attempts")
                return None

    return None


def _fetch_income_statement(symbol: str, api_key: str, limit: int = 10) -> pd.DataFrame:
    """Lädt Income Statement von FMP API (Stable API)."""
    url = "https://financialmodelingprep.com/stable/income-statement"
    params = {'symbol': symbol, 'apikey': api_key, 'limit': limit}

    logger.info(f"  Fetching Income Statement for {symbol}...")
    data = _make_api_request(url, params)

    if data is None or len(data) == 0:
        logger.warning(f"  No Income Statement data for {symbol}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _fetch_balance_sheet(symbol: str, api_key: str, limit: int = 10) -> pd.DataFrame:
    """Lädt Balance Sheet von FMP API (Stable API)."""
    url = "https://financialmodelingprep.com/stable/balance-sheet-statement"
    params = {'symbol': symbol, 'apikey': api_key, 'limit': limit}

    logger.info(f"  Fetching Balance Sheet for {symbol}...")
    data = _make_api_request(url, params)

    if data is None or len(data) == 0:
        logger.warning(f"  No Balance Sheet data for {symbol}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _fetch_cash_flow(symbol: str, api_key: str, limit: int = 10) -> pd.DataFrame:
    """Lädt Cash Flow Statement von FMP API (Stable API)."""
    url = "https://financialmodelingprep.com/stable/cash-flow-statement"
    params = {'symbol': symbol, 'apikey': api_key, 'limit': limit}

    logger.info(f"  Fetching Cash Flow Statement for {symbol}...")
    data = _make_api_request(url, params)

    if data is None or len(data) == 0:
        logger.warning(f"  No Cash Flow data for {symbol}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _fetch_company_profile(symbol: str, api_key: str) -> dict:
    """Lädt Company Profile (Sector, Industry, Country, etc.) von FMP API (Stable API)."""
    url = "https://financialmodelingprep.com/stable/profile"
    params = {'symbol': symbol, 'apikey': api_key}

    logger.info(f"  Fetching Company Profile for {symbol}...")
    data = _make_api_request(url, params)

    if data is None or len(data) == 0:
        logger.warning(f"  No Company Profile data for {symbol}")
        return {}

    return data[0] if isinstance(data, list) else data


# ============================================================================
# MAIN DATA LOADING FUNCTIONS
# ============================================================================

def fetch_company_data(symbol: str, api_key: str, years: int = 10) -> pd.DataFrame:
    """
    Lädt alle Finanzdaten für ein Unternehmen von FMP API.

    Ruft ab:
    - Income Statement (GuV)
    - Balance Sheet (Bilanz)
    - Cash Flow Statement
    - Company Profile (Meta-Daten)

    Args:
        symbol: Ticker-Symbol (z.B. 'AAPL', 'SAP.DE')
        api_key: FMP API Key
        years: Anzahl Jahre historische Daten (default: 10)

    Returns:
        DataFrame mit allen kombinierten Finanzdaten (noch im FMP-Format)

    Example:
        >>> df = fetch_company_data('AAPL', api_key='YOUR_KEY', years=5)
        >>> print(df.shape)
        (5, 150)  # 5 Jahre, ~150 Spalten
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading data for: {symbol}")
    logger.info(f"{'='*60}")

    # 1. Lade alle Statements
    df_income = _fetch_income_statement(symbol, api_key, limit=years)
    df_balance = _fetch_balance_sheet(symbol, api_key, limit=years)
    df_cashflow = _fetch_cash_flow(symbol, api_key, limit=years)
    profile = _fetch_company_profile(symbol, api_key)

    # Prüfe ob wir Daten haben
    if df_income.empty and df_balance.empty and df_cashflow.empty:
        logger.error(f"No data available for {symbol}")
        return pd.DataFrame()

    # 2. Merge alle DataFrames auf 'date' Spalte
    # FMP nutzt 'date' oder 'calendarYear' als gemeinsamen Key
    date_col = 'date' if 'date' in df_income.columns else 'calendarYear'

    # Starte mit Income Statement als Basis
    df_combined = df_income.copy() if not df_income.empty else df_balance.copy()

    # Merge Balance Sheet (drop duplicates)
    if not df_balance.empty and not df_combined.empty:
        # Find columns that exist in both (except merge key)
        overlap_cols = set(df_combined.columns) & set(df_balance.columns) - {date_col}
        # Drop overlapping columns from balance sheet before merge
        df_balance_clean = df_balance.drop(columns=list(overlap_cols), errors='ignore')

        df_combined = df_combined.merge(
            df_balance_clean,
            on=date_col,
            how='outer',
            suffixes=('', '_bs')
        )

    # Merge Cash Flow (drop duplicates)
    if not df_cashflow.empty and not df_combined.empty:
        # Find columns that exist in both (except merge key)
        overlap_cols = set(df_combined.columns) & set(df_cashflow.columns) - {date_col}
        # Drop overlapping columns from cashflow before merge
        df_cashflow_clean = df_cashflow.drop(columns=list(overlap_cols), errors='ignore')

        df_combined = df_combined.merge(
            df_cashflow_clean,
            on=date_col,
            how='outer',
            suffixes=('', '_cf')
        )

    # 3. Füge Company Profile Daten hinzu (als konstante Spalten)
    if profile:
        df_combined['symbol'] = profile.get('symbol', symbol)
        df_combined['companyName'] = profile.get('companyName', '')
        df_combined['sector'] = profile.get('sector', '')
        df_combined['industry'] = profile.get('industry', '')
        df_combined['country'] = profile.get('country', '')
        df_combined['isin'] = profile.get('isin', '')
    else:
        # Fallback wenn kein Profile verfügbar
        df_combined['symbol'] = symbol

    logger.info(f"✓ Loaded {len(df_combined)} years for {symbol}")
    logger.info(f"  Columns: {len(df_combined.columns)}")

    return df_combined


def convert_to_wrds_format(df_fmp: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert FMP-Daten in WRDS-kompatibles Format.

    Schritte:
    1. Rename Spalten gemäß FMP_TO_WRDS_MAPPING
    2. Erstelle fehlende WRDS-Felder (gvkey, fyear, datadate)
    3. Berechne approximative GICS-Codes
    4. Standardisiere CapEx/Dividends (FMP gibt negative Werte)
    5. Sortiere Spalten in WRDS-Reihenfolge

    Args:
        df_fmp: DataFrame im FMP-Format (von fetch_company_data())

    Returns:
        DataFrame im WRDS-Format (kompatibel mit bestehender Pipeline)

    Example:
        >>> df = fetch_company_data('AAPL', api_key='KEY')
        >>> df_wrds = convert_to_wrds_format(df)
        >>> # Jetzt mit Pipeline nutzbar:
        >>> from src._02_preprocessing import data_cleaner
        >>> df_features = data_cleaner.run_preprocessing(df_raw=df_wrds, ...)
    """
    logger.info("\n" + "="*60)
    logger.info("Converting FMP → WRDS format")
    logger.info("="*60)

    df = df_fmp.copy()

    # 1. Rename Spalten gemäß Mapping
    logger.info("  Renaming columns...")
    rename_dict = {}
    for fmp_col, wrds_col in FMP_TO_WRDS_MAPPING.items():
        if fmp_col in df.columns:
            rename_dict[fmp_col] = wrds_col

    df = df.rename(columns=rename_dict)
    logger.info(f"    ✓ Renamed {len(rename_dict)} columns")

    # 2. Erstelle WRDS-spezifische Felder
    logger.info("  Creating WRDS-specific fields...")

    # gvkey: Nutze Symbol als Company-ID (tic wurde bereits gemappt)
    if 'tic' in df.columns and 'gvkey' not in df.columns:
        df['gvkey'] = df['tic']

    # datadate: Konvertiere FMP 'date' zu datetime
    if 'date' in df.columns:
        df['datadate'] = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    elif 'calendarYear' in df.columns:
        # Fallback: Nutze calendarYear für fyear
        df['fyear'] = df['calendarYear']
        df['datadate'] = pd.to_datetime(df['calendarYear'].astype(str) + '-12-31')

    # fyear: Extrahiere Jahr aus datadate
    if 'datadate' in df.columns:
        df['fyear'] = df['datadate'].dt.year

    # 3. GICS Sector Approximation
    if 'sector_fmp' in df.columns:
        logger.info("  Mapping FMP Sector → GICS...")
        df['gsector'] = df['sector_fmp'].map(FMP_SECTOR_TO_GICS)
        # Behalte auch FMP Sector für Referenz, aber nicht als WRDS-Feld
        df = df.drop(columns=['sector_fmp', 'industry_fmp'], errors='ignore')

    # 4. Standardisiere Vorzeichen (FMP: CapEx & Dividends sind negativ)
    logger.info("  Standardizing signs for CapEx and Dividends...")

    if 'capx' in df.columns:
        # WRDS erwartet positive CapEx, FMP liefert negative
        df['capx'] = df['capx'].abs()

    if 'dvt' in df.columns:
        # WRDS erwartet positive Dividends, FMP liefert negative
        df['dvt'] = df['dvt'].abs()

    # 5. Fehlende kritische Felder prüfen
    logger.info("  Checking for missing critical fields...")
    critical_fields = ['gvkey', 'datadate', 'fyear', 'at', 'revt']
    missing = [f for f in critical_fields if f not in df.columns]

    if missing:
        logger.warning(f"  ⚠ Missing critical fields: {missing}")
    else:
        logger.info("    ✓ All critical fields present")

    # 6. Sortiere nach gvkey und fyear (wie WRDS)
    if 'gvkey' in df.columns and 'fyear' in df.columns:
        df = df.sort_values(['gvkey', 'fyear'], ascending=[True, True])

    logger.info(f"\n✓ Conversion complete")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Years: {df['fyear'].min()}-{df['fyear'].max()}" if 'fyear' in df.columns else "")
    logger.info("="*60 + "\n")

    return df


def fetch_multiple_companies(
    symbols: List[str],
    api_key: str,
    years: int = 10,
    rate_limit_delay: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Lädt Daten für mehrere Unternehmen mit Rate Limiting.

    Args:
        symbols: Liste von Ticker-Symbolen
        api_key: FMP API Key
        years: Anzahl Jahre pro Unternehmen
        rate_limit_delay: Wartezeit zwischen Requests in Sekunden (default: 1.0)
                         Free Tier: Max 5 req/min → min 12 sec delay empfohlen
        verbose: Zeige Fortschritt (default: True)

    Returns:
        Kombinierter DataFrame mit allen Unternehmen im WRDS-Format

    Example:
        >>> symbols = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM']
        >>> df = fetch_multiple_companies(symbols, api_key='KEY', rate_limit_delay=12)
        >>> # → ~20 API Requests (4 pro Unternehmen × 5)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"FETCHING MULTIPLE COMPANIES")
    logger.info(f"{'='*80}")
    logger.info(f"  Companies: {len(symbols)}")
    logger.info(f"  Years per company: {years}")
    logger.info(f"  Estimated API calls: {len(symbols) * 4} (4 per company)")
    logger.info(f"  Rate limit delay: {rate_limit_delay}s between companies")
    logger.info(f"{'='*80}\n")

    dataframes = []
    failed_symbols = []

    for i, symbol in enumerate(symbols, 1):
        if verbose:
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            # Lade Daten für ein Unternehmen
            df = fetch_company_data(symbol, api_key, years)

            if df.empty:
                logger.warning(f"  ⚠ No data for {symbol} - skipping")
                failed_symbols.append(symbol)
                continue

            # Konvertiere zu WRDS-Format
            df_wrds = convert_to_wrds_format(df)
            dataframes.append(df_wrds)

            if verbose:
                logger.info(f"  ✓ {symbol}: {len(df_wrds)} years loaded\n")

            # Rate Limiting (außer beim letzten Symbol)
            if i < len(symbols):
                time.sleep(rate_limit_delay)

        except Exception as e:
            logger.error(f"  ✗ Error processing {symbol}: {e}")
            failed_symbols.append(symbol)
            continue

    # Kombiniere alle DataFrames
    if not dataframes:
        logger.error("No data loaded for any company!")
        return pd.DataFrame()

    df_combined = pd.concat(dataframes, ignore_index=True)

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  ✓ Successfully loaded: {len(dataframes)}/{len(symbols)} companies")
    logger.info(f"  Total rows: {len(df_combined)}")
    logger.info(f"  Total columns: {len(df_combined.columns)}")

    if failed_symbols:
        logger.warning(f"  ⚠ Failed symbols: {failed_symbols}")

    logger.info(f"{'='*80}\n")

    return df_combined


# ============================================================================
# INDEX CONSTITUENTS (Phase 3 - Future)
# ============================================================================

def get_index_constituents(index: str = 'sp500', api_key: str = None) -> List[str]:
    """
    Holt Liste von Ticker-Symbolen für einen Index.

    Args:
        index: Index-Name ('sp500', 'nasdaq', 'dowjones')
        api_key: FMP API Key

    Returns:
        Liste von Ticker-Symbolen

    Note:
        Nicht im PoC implementiert - für Phase 3
    """
    logger.warning("get_index_constituents() not implemented yet - Phase 3 feature")
    return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Speichert DataFrame als CSV.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to: {output_file}")


def load_fmp_market_data(
    market: str,
    api_key: str,
    symbols: Optional[List[str]] = None,
    years: int = 10,
    output_dir: str = 'data/raw',
    save_csv: bool = False
) -> pd.DataFrame:
    """
    Hauptfunktion: Lädt Marktdaten via FMP API.

    Integration-Optionen:

    Option 1: Direkter DataFrame-Rückgabe (empfohlen für erste Tests)
        >>> df = load_fmp_market_data(market='test', symbols=['AAPL', 'MSFT'], api_key=KEY)
        >>> from src._02_preprocessing import data_cleaner
        >>> df_features = data_cleaner.run_preprocessing(df_raw=df, market='test')

    Option 2: CSV-Export für Offline-Nutzung
        >>> df = load_fmp_market_data(market='sp500', symbols=sp500_list,
        ...                           api_key=KEY, save_csv=True)
        >>> # Später: Normale Pipeline ohne API-Calls
        >>> python src/main.py --market sp500

    Args:
        market: Marktname (z.B. 'sp500', 'test', 'dax')
        api_key: FMP API Key
        symbols: Liste von Ticker-Symbolen (erforderlich!)
        years: Anzahl Jahre historische Daten
        output_dir: Ausgabe-Verzeichnis (default: 'data/raw')
        save_csv: Speichere als CSV (default: False)

    Returns:
        DataFrame im WRDS-Format

    Example:
        >>> # PoC Test mit 3 Unternehmen
        >>> symbols = ['AAPL', 'MSFT', 'JNJ']
        >>> df = load_fmp_market_data(market='poc_test', symbols=symbols, api_key=KEY)
        >>> print(f"Loaded {len(df)} rows for {df['gvkey'].nunique()} companies")
    """
    if symbols is None:
        raise ValueError("symbols parameter is required! Provide a list of ticker symbols.")

    logger.info(f"\n{'='*80}")
    logger.info(f"LOAD FMP MARKET DATA: {market}")
    logger.info(f"{'='*80}")

    # Lade alle Unternehmen
    df = fetch_multiple_companies(symbols, api_key, years=years)

    if df.empty:
        logger.error("No data loaded!")
        return pd.DataFrame()

    # Optional: Save to CSV
    if save_csv:
        output_path = Path(output_dir) / market / 'fmp_data.csv'
        save_to_csv(df, str(output_path))

    return df


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test-Script für Proof of Concept.

    Usage:
        export FMP_API_KEY="your_key_here"
        python src/_02_preprocessing/fmp_data_loader.py
    """
    import os

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Get API key from environment
    api_key = os.getenv('FMP_API_KEY')

    if not api_key:
        logger.error("FMP_API_KEY environment variable not set!")
        logger.info("Set it with: export FMP_API_KEY='your_key_here'")
        exit(1)

    # Test symbols (3-5 große US-Unternehmen mit guten Daten)
    test_symbols = [
        'AAPL',  # Apple - Technology
        'MSFT',  # Microsoft - Technology
        'JNJ',   # Johnson & Johnson - Healthcare
        'JPM',   # JP Morgan - Financials
        'XOM',   # Exxon Mobil - Energy
    ]

    logger.info("\n" + "="*80)
    logger.info("FMP DATA LOADER - PROOF OF CONCEPT TEST")
    logger.info("="*80)
    logger.info(f"Testing with {len(test_symbols)} companies: {test_symbols}")
    logger.info("="*80 + "\n")

    # Lade Daten
    df = load_fmp_market_data(
        market='poc_test',
        symbols=test_symbols,
        api_key=api_key,
        years=5,  # Nur 5 Jahre für schnelleren Test
        save_csv=True
    )

    # Zeige Ergebnisse
    if not df.empty:
        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Companies: {df['gvkey'].nunique()}")
        logger.info(f"Years range: {df['fyear'].min()}-{df['fyear'].max()}")
        logger.info(f"\nAvailable WRDS columns:")
        wrds_cols = [c for c in df.columns if c in ['gvkey', 'fyear', 'datadate', 'conm',
                                                      'at', 'revt', 'ebit', 'ni', 'seq',
                                                      'oancf', 'capx', 'roa', 'roe']]
        logger.info(f"  {wrds_cols}")
        logger.info(f"\nSample data (first company, latest year):")
        sample = df[df['gvkey'] == df['gvkey'].iloc[0]].head(1)
        logger.info(f"\n{sample[['gvkey', 'fyear', 'conm', 'revt', 'at', 'ni']].to_string()}")
        logger.info("="*80)

        logger.info("\n✅ Proof of Concept successful!")
        logger.info("Next steps:")
        logger.info("  1. Check data/raw/poc_test/fmp_data.csv")
        logger.info("  2. Validate with: df_features = data_cleaner.run_preprocessing(df_raw=df, ...)")
        logger.info("  3. Compare calculated ratios with expected values")
    else:
        logger.error("❌ Test failed - no data loaded")
