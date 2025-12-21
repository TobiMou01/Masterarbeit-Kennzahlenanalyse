#!/usr/bin/env python3
"""
Phase 2 Integration Test: FMP Data → Preprocessing Pipeline → Calculated Features

Testet ob:
1. FMP-Daten korrekt durch die Pipeline laufen
2. Alle Calculator funktionieren
3. Berechnete Kennzahlen plausibel sind
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src._02_preprocessing import fmp_data_loader, data_cleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_phase2_integration():
    """
    Haupttest: FMP API → Pipeline → Features
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: INTEGRATION TEST")
    logger.info("="*80)

    # 1. Get API Key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("FMP_API_KEY environment variable not set!")
        return False

    logger.info(f"✓ API Key found (length: {len(api_key)})")

    # 2. Load FMP Data (nur 2 Unternehmen für schnellen Test)
    logger.info("\n" + "-"*80)
    logger.info("STEP 1: Loading FMP Data")
    logger.info("-"*80)

    symbols = ['AAPL', 'MSFT']  # Apple & Microsoft
    years = 5  # Nur 5 Jahre für schnellen Test

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Years: {years}")

    try:
        df_raw = fmp_data_loader.load_fmp_market_data(
            market='phase2_test',
            symbols=symbols,
            api_key=api_key,
            years=years,
            save_csv=False  # Nicht speichern für schnellen Test
        )
    except Exception as e:
        logger.error(f"Failed to load FMP data: {e}")
        import traceback
        traceback.print_exc()
        return False

    if df_raw.empty:
        logger.error("No data loaded from FMP!")
        return False

    logger.info(f"\n✓ FMP Data loaded successfully:")
    logger.info(f"  Rows: {len(df_raw)}")
    logger.info(f"  Columns: {len(df_raw.columns)}")
    logger.info(f"  Companies: {df_raw['gvkey'].nunique()}")
    logger.info(f"  Years: {df_raw['fyear'].min()}-{df_raw['fyear'].max()}")

    # Zeige Sample-Spalten
    logger.info(f"\n  Sample WRDS columns available:")
    wrds_cols = ['gvkey', 'fyear', 'datadate', 'conm', 'at', 'revt', 'ebit', 'ni', 'seq', 'oancf', 'capx']
    available = [c for c in wrds_cols if c in df_raw.columns]
    logger.info(f"    {available}")

    # 3. Run through Preprocessing Pipeline
    logger.info("\n" + "-"*80)
    logger.info("STEP 2: Running through Preprocessing Pipeline")
    logger.info("-"*80)

    try:
        df_features = data_cleaner.run_preprocessing(
            input_dir='data/raw',  # Erforderlich, aber nicht genutzt da df_raw übergeben wird
            market='phase2_test',
            df_raw=df_raw,  # Pre-loaded FMP data
            impute=True,
            impute_method='median',
            impute_threshold=0.5
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    if df_features.empty:
        logger.error("Preprocessing returned empty DataFrame!")
        return False

    logger.info(f"\n✓ Preprocessing completed successfully:")
    logger.info(f"  Rows: {len(df_features)}")
    logger.info(f"  Columns: {len(df_features.columns)}")
    logger.info(f"  Companies: {df_features['gvkey'].nunique()}")

    # 4. Validate Calculated Features
    logger.info("\n" + "-"*80)
    logger.info("STEP 3: Validating Calculated Features")
    logger.info("-"*80)

    # Check welche Kennzahlen berechnet wurden
    expected_features = {
        'Profitability': ['roa', 'roe', 'ebit_margin', 'net_profit_margin', 'gross_margin'],
        'Liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
        'Leverage': ['debt_to_equity', 'equity_ratio', 'interest_coverage'],
        'Efficiency': ['asset_turnover', 'capital_intensity'],
        'Cashflow': ['fcf_margin', 'cash_conversion'],
    }

    logger.info("\nFeature availability by category:")
    total_available = 0
    total_expected = 0

    for category, features in expected_features.items():
        available = [f for f in features if f in df_features.columns]
        total_available += len(available)
        total_expected += len(features)

        status = "✓" if len(available) == len(features) else "⚠"
        logger.info(f"  {status} {category}: {len(available)}/{len(features)} available")

        if len(available) < len(features):
            missing = [f for f in features if f not in df_features.columns]
            logger.warning(f"      Missing: {missing}")

    coverage = (total_available / total_expected * 100) if total_expected > 0 else 0
    logger.info(f"\n  Overall Coverage: {total_available}/{total_expected} ({coverage:.1f}%)")

    # 5. Validate Apple's Ratios (Plausibility Check)
    logger.info("\n" + "-"*80)
    logger.info("STEP 4: Plausibility Check (Apple, latest year)")
    logger.info("-"*80)

    if 'AAPL' in df_features['gvkey'].values:
        aapl = df_features[df_features['gvkey'] == 'AAPL'].nlargest(1, 'fyear')

        if len(aapl) > 0:
            logger.info("\nApple Inc. - Calculated Ratios:")
            logger.info(f"  Company: {aapl['conm'].values[0] if 'conm' in aapl.columns else 'N/A'}")
            logger.info(f"  Year: {aapl['fyear'].values[0]}")

            # Expected ranges für Apple (basierend auf öffentlichen Daten)
            expected_ranges = {
                'roa': (25, 35, '%'),                    # Return on Assets
                'roe': (100, 180, '%'),                  # Return on Equity (sehr hoch bei Apple!)
                'ebit_margin': (28, 35, '%'),            # EBIT Margin
                'net_profit_margin': (20, 30, '%'),      # Net Profit Margin
                'current_ratio': (0.8, 1.2, ''),         # Current Ratio
                'debt_to_equity': (1.0, 2.5, ''),        # Debt-to-Equity
                'fcf_margin': (20, 30, '%'),             # Free Cash Flow Margin
            }

            logger.info("\n  Ratio Validation:")
            all_valid = True

            for metric, (min_val, max_val, unit) in expected_ranges.items():
                if metric in aapl.columns:
                    value = aapl[metric].values[0]

                    if pd.notna(value):
                        in_range = min_val <= value <= max_val
                        status = "✓" if in_range else "⚠"

                        logger.info(f"    {status} {metric}: {value:.2f}{unit} (expected: {min_val}-{max_val}{unit})")

                        if not in_range:
                            all_valid = False
                    else:
                        logger.warning(f"    ✗ {metric}: MISSING (NaN)")
                        all_valid = False
                else:
                    logger.warning(f"    ✗ {metric}: NOT CALCULATED")
                    all_valid = False

            if all_valid:
                logger.info("\n  ✓ All ratios within expected ranges!")
            else:
                logger.warning("\n  ⚠ Some ratios outside expected ranges (may be data issue)")
        else:
            logger.warning("No data for AAPL found!")
    else:
        logger.warning("AAPL not in dataset!")

    # 6. Show Sample Features DataFrame
    logger.info("\n" + "-"*80)
    logger.info("STEP 5: Sample Output")
    logger.info("-"*80)

    logger.info("\nSample features for latest year (both companies):")
    latest_year = df_features['fyear'].max()
    df_sample = df_features[df_features['fyear'] == latest_year].copy()

    display_cols = ['gvkey', 'conm', 'fyear', 'roa', 'roe', 'ebit_margin', 'current_ratio', 'debt_to_equity']
    available_display_cols = [c for c in display_cols if c in df_sample.columns]

    if available_display_cols:
        logger.info(f"\n{df_sample[available_display_cols].to_string(index=False)}")

    # 7. Final Summary
    logger.info("\n" + "="*80)
    logger.info("PHASE 2 TEST SUMMARY")
    logger.info("="*80)

    logger.info(f"✓ FMP Data Loading: SUCCESS")
    logger.info(f"✓ Preprocessing Pipeline: SUCCESS")
    logger.info(f"✓ Feature Calculation: {coverage:.1f}% coverage")
    logger.info(f"✓ Plausibility Check: {'PASS' if all_valid else 'PARTIAL'}")

    logger.info("\n" + "="*80)
    logger.info("CONCLUSION")
    logger.info("="*80)

    if coverage >= 80 and not df_features.empty:
        logger.info("✅ PHASE 2 SUCCESSFUL!")
        logger.info("\n   FMP data is compatible with your pipeline!")
        logger.info("   You can now use FMP API for your thesis data.")
        logger.info("\n   Next Steps:")
        logger.info("   - Phase 3: Test with German stocks (SAP.DE, BMW.DE, etc.)")
        logger.info("   - Phase 3: Implement index constituents loading")
        logger.info("   - Phase 3: Scale to larger datasets")
        return True
    else:
        logger.warning("⚠️ PHASE 2 PARTIAL SUCCESS")
        logger.warning(f"   Feature coverage: {coverage:.1f}% (expected >80%)")
        logger.warning("   Some calculators may not work correctly.")
        return False


if __name__ == "__main__":
    success = test_phase2_integration()
    sys.exit(0 if success else 1)
