#!/usr/bin/env python3
"""
Test Script für FMP Data Loader (Phase 1: Proof of Concept)

Testet ob FMP API die nötigen Daten liefert und ob das WRDS-Mapping funktioniert.

Usage:
    1. Setze API Key als Umgebungsvariable:
       export FMP_API_KEY="your_key_here"

    2. Führe Script aus:
       python test_fmp_loader.py

    3. Prüfe Ausgabe:
       - Werden alle Unternehmen erfolgreich geladen?
       - Sind die WRDS-Spalten vorhanden?
       - Sind die Werte plausibel?
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src._02_preprocessing import fmp_data_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_single_company(api_key: str):
    """Test 1: Lade einzelnes Unternehmen (Apple)"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Single Company (AAPL)")
    logger.info("="*80)

    df = fmp_data_loader.fetch_company_data('AAPL', api_key, years=5)
    df_wrds = fmp_data_loader.convert_to_wrds_format(df)

    logger.info("\nResults:")
    logger.info(f"  Rows: {len(df_wrds)}")
    logger.info(f"  Columns: {len(df_wrds.columns)}")
    logger.info(f"  Years: {df_wrds['fyear'].min()}-{df_wrds['fyear'].max()}")

    # Check kritische WRDS-Spalten
    critical_cols = ['gvkey', 'fyear', 'datadate', 'conm', 'at', 'revt', 'ebit', 'ni', 'seq', 'oancf', 'capx']
    missing = [c for c in critical_cols if c not in df_wrds.columns]

    if missing:
        logger.warning(f"  ⚠ Missing columns: {missing}")
    else:
        logger.info(f"  ✓ All critical columns present")

    # Zeige Sample-Daten (neuestes Jahr)
    logger.info("\nSample data (latest year):")
    latest = df_wrds.nlargest(1, 'fyear')
    display_cols = ['gvkey', 'fyear', 'conm', 'revt', 'at', 'ni', 'ebit', 'seq']
    available_cols = [c for c in display_cols if c in latest.columns]
    logger.info(f"\n{latest[available_cols].to_string(index=False)}")

    return df_wrds


def test_multiple_companies(api_key: str):
    """Test 2: Lade 5 Unternehmen aus verschiedenen Sektoren"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Multiple Companies (5 different sectors)")
    logger.info("="*80)

    symbols = [
        'AAPL',  # Technology
        'MSFT',  # Technology
        'JNJ',   # Healthcare
        'JPM',   # Financials
        'XOM',   # Energy
    ]

    df = fmp_data_loader.fetch_multiple_companies(
        symbols=symbols,
        api_key=api_key,
        years=5,
        rate_limit_delay=1.0,  # Kurze Wartezeit für Test
        verbose=True
    )

    logger.info("\nResults:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Companies loaded: {df['gvkey'].nunique()}/{len(symbols)}")
    logger.info(f"  Columns: {len(df.columns)}")

    # Zeige Verteilung pro Unternehmen
    logger.info("\nRows per company:")
    for company, count in df['gvkey'].value_counts().items():
        company_name = df[df['gvkey'] == company]['conm'].iloc[0] if 'conm' in df.columns else company
        logger.info(f"  {company}: {count} years - {company_name}")

    # Zeige verfügbare WRDS-Felder
    wrds_standard_fields = [
        'gvkey', 'fyear', 'datadate', 'conm', 'loc',  # Identifikation
        'gsector', 'at', 'revt', 'ebit', 'ebitda', 'ni',  # Basics
        'seq', 'lt', 'act', 'lct', 'che',  # Bilanz
        'oancf', 'capx', 'dp', 'dvt',  # Cashflow
        'cogs', 'xint', 'xrd', 'ppent', 'invt', 'rect',  # Details
    ]

    available = [f for f in wrds_standard_fields if f in df.columns]
    missing = [f for f in wrds_standard_fields if f not in df.columns]

    logger.info(f"\nWRDS fields available: {len(available)}/{len(wrds_standard_fields)}")
    if missing:
        logger.warning(f"  Missing fields: {missing[:10]}...")  # Nur erste 10 zeigen

    return df


def test_data_quality(df):
    """Test 3: Prüfe Datenqualität und plausible Werte"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Data Quality Check")
    logger.info("="*80)

    # 1. Missing Values
    logger.info("\nMissing values (top 10 columns):")
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(10)
    for col, pct in missing_pct.items():
        logger.info(f"  {col}: {pct:.1f}%")

    # 2. Plausibilitäts-Checks für Apple (neuestes Jahr)
    if 'AAPL' in df['gvkey'].values:
        logger.info("\nPlausibility check for AAPL (latest year):")
        aapl_latest = df[(df['gvkey'] == 'AAPL')].nlargest(1, 'fyear')

        checks = {
            'Revenue (revt)': aapl_latest['revt'].values[0] if 'revt' in aapl_latest.columns and len(aapl_latest) > 0 else None,
            'Total Assets (at)': aapl_latest['at'].values[0] if 'at' in aapl_latest.columns and len(aapl_latest) > 0 else None,
            'Net Income (ni)': aapl_latest['ni'].values[0] if 'ni' in aapl_latest.columns and len(aapl_latest) > 0 else None,
            'EBIT': aapl_latest['ebit'].values[0] if 'ebit' in aapl_latest.columns and len(aapl_latest) > 0 else None,
        }

        for metric, value in checks.items():
            # Handle both scalar and array values
            if value is not None:
                # Check if it's a scalar or convert to scalar
                try:
                    value_scalar = float(value) if not isinstance(value, (list, tuple, pd.Series)) else value
                    if pd.notna(value_scalar):
                        logger.info(f"  {metric}: ${value_scalar:,.0f}M")
                    else:
                        logger.warning(f"  {metric}: MISSING")
                except (TypeError, ValueError):
                    logger.warning(f"  {metric}: INVALID VALUE")
            else:
                logger.warning(f"  {metric}: MISSING")

        # Erwartete Größenordnung für Apple (ca. Werte):
        # Revenue: ~380-400 Milliarden
        # Assets: ~350-370 Milliarden
        # Net Income: ~90-100 Milliarden
        expected = {
            'revt': (300_000, 500_000),  # in Millionen
            'at': (300_000, 400_000),
            'ni': (80_000, 120_000),
        }

        logger.info("\nPlausibility assessment:")
        for field, (min_val, max_val) in expected.items():
            if field in aapl_latest.columns and len(aapl_latest) > 0:
                try:
                    value = float(aapl_latest[field].values[0])
                    if pd.notna(value) and min_val <= value <= max_val:
                        logger.info(f"  ✓ {field} is within expected range")
                    elif pd.notna(value):
                        logger.warning(f"  ⚠ {field} outside expected range: {value:,.0f} (expected {min_val:,.0f}-{max_val:,.0f})")
                    else:
                        logger.warning(f"  ⚠ {field} is missing")
                except (TypeError, ValueError, IndexError):
                    logger.warning(f"  ⚠ {field} is missing or invalid")


def test_integration_readiness(df):
    """Test 4: Prüfe ob DataFrame ready für Pipeline ist"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Integration Readiness Check")
    logger.info("="*80)

    # Felder die Calculator erwarten (aus ratio_calculators.py)
    calculator_required_fields = {
        'Profitability': ['ebit', 'at', 'ib', 'seq', 'revt', 'oibdp', 'cogs'],
        'Liquidity': ['act', 'lct', 'invt', 'che'],
        'Leverage': ['lt', 'seq', 'dltt', 'dlc'],
        'Cashflow': ['oancf', 'capx', 'ni', 'revt'],
    }

    logger.info("\nRequired fields for Calculators:")
    all_ready = True

    for category, fields in calculator_required_fields.items():
        available = [f for f in fields if f in df.columns]
        missing = [f for f in fields if f not in df.columns]

        logger.info(f"\n  {category}: {len(available)}/{len(fields)} available")
        if missing:
            logger.warning(f"    Missing: {missing}")
            all_ready = False
        else:
            logger.info(f"    ✓ All fields available")

    if all_ready:
        logger.info("\n✅ DataFrame is ready for integration with existing pipeline!")
    else:
        logger.warning("\n⚠ Some fields are missing - Calculators may not work completely")

    return all_ready


def main():
    """Hauptfunktion: Führe alle Tests durch"""
    logger.info("\n" + "="*80)
    logger.info("FMP DATA LOADER - PROOF OF CONCEPT TEST SUITE")
    logger.info("="*80)

    # 1. Check API Key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("\n❌ FMP_API_KEY environment variable not set!")
        logger.info("\nSet it with:")
        logger.info("  export FMP_API_KEY='your_key_here'")
        logger.info("\nYou can get a free API key at: https://site.financialmodelingprep.com/developer/docs")
        return 1

    logger.info(f"\n✓ API Key found (length: {len(api_key)})")

    try:
        # Test 1: Single Company
        df_single = test_single_company(api_key)

        # Test 2: Multiple Companies
        df_multi = test_multiple_companies(api_key)

        # Test 3: Data Quality
        test_data_quality(df_multi)

        # Test 4: Integration Readiness
        ready = test_integration_readiness(df_multi)

        # Final Summary
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ Single company test: SUCCESS")
        logger.info(f"✓ Multiple companies test: SUCCESS ({df_multi['gvkey'].nunique()} companies loaded)")
        logger.info(f"✓ Data quality check: COMPLETED")
        logger.info(f"{'✓' if ready else '⚠'} Integration readiness: {'READY' if ready else 'PARTIAL'}")

        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS - PHASE 2")
        logger.info("="*80)
        logger.info("1. Validate with your existing pipeline:")
        logger.info("   from src._02_preprocessing import data_cleaner")
        logger.info("   df_features = data_cleaner.run_preprocessing(")
        logger.info("       df_raw=df_multi,")
        logger.info("       market='fmp_test',")
        logger.info("       impute=True")
        logger.info("   )")
        logger.info("")
        logger.info("2. Check calculated ratios (ROA, ROE, etc.)")
        logger.info("3. Compare with expected values")
        logger.info("4. If successful → Proceed to Phase 3 (Scaling)")
        logger.info("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import pandas as pd  # Import needed for test_data_quality
    sys.exit(main())
