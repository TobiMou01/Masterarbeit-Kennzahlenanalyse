"""
Cashflow Calculators Module
Berechnet Cashflow- und Struktur-Kennzahlen
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_cashflow_metrics(df):
    """
    Berechnet Cashflow- und Investitions-Kennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Cashflow-Kennzahlen
    """
    logger.info("Berechne Cashflow-Kennzahlen...")

    df = df.copy()

    # Capex to Revenue (%)
    if 'capx' in df.columns and 'revt' in df.columns:
        df['capex_to_revenue'] = np.where(
            df['revt'] > 0,
            (df['capx'].abs() / df['revt']) * 100,
            np.nan
        )
        logger.info("  ✓ Capex to Revenue berechnet")
    else:
        if 'capx' not in df.columns:
            logger.warning("  ⚠ Capex to Revenue nicht berechnet - capx Spalte fehlt")

    # Capex to Depreciation
    if 'capx' in df.columns and 'dp' in df.columns:
        df['capex_to_depreciation'] = np.where(
            (df['dp'].notna()) & (df['dp'] > 0),
            df['capx'].abs() / df['dp'],
            np.nan
        )
        logger.info("  ✓ Capex to Depreciation berechnet")
    else:
        if 'capx' not in df.columns:
            logger.warning("  ⚠ Capex to Depreciation nicht berechnet - capx Spalte fehlt")

    # Free Cash Flow (FCF)
    if 'oancf' in df.columns and 'capx' in df.columns:
        df['fcf'] = df['oancf'] - df['capx'].abs()
        logger.info("  ✓ Free Cash Flow berechnet")
    else:
        if 'capx' not in df.columns:
            logger.warning("  ⚠ Free Cash Flow nicht berechnet - capx Spalte fehlt")

    # FCF Margin (%)
    if 'fcf' in df.columns and 'revt' in df.columns:
        df['fcf_margin'] = np.where(
            df['revt'] > 0,
            (df['fcf'] / df['revt']) * 100,
            np.nan
        )
        logger.info("  ✓ FCF Margin berechnet")

    # Reinvestment Rate (%)
    if 'capx' in df.columns and 'oancf' in df.columns:
        df['reinvestment_rate'] = np.where(
            (df['oancf'].notna()) & (df['oancf'] > 0),
            (df['capx'].abs() / df['oancf']) * 100,
            np.nan
        )
        logger.info("  ✓ Reinvestment Rate berechnet")
    else:
        if 'capx' not in df.columns:
            logger.warning("  ⚠ Reinvestment Rate nicht berechnet - capx Spalte fehlt")

    # Cash Conversion (Operating CF / EBIT) (%)
    if 'oancf' in df.columns and 'ebit' in df.columns:
        df['cash_conversion'] = np.where(
            (df['ebit'].notna()) & (df['ebit'] > 0),
            (df['oancf'] / df['ebit']) * 100,
            np.nan
        )
        logger.info("  ✓ Cash Conversion berechnet")

    return df


def calculate_structure_metrics(df):
    """
    Berechnet Struktur- und Qualitätskennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Struktur-Kennzahlen
    """
    logger.info("Berechne Struktur-Kennzahlen...")

    df = df.copy()

    # Financial Leverage
    if 'at' in df.columns and 'seq' in df.columns:
        df['financial_leverage'] = np.where(
            (df['seq'].notna()) & (df['seq'] > 0),
            df['at'] / df['seq'],
            np.nan
        )
        logger.info("  ✓ Financial Leverage berechnet")

    # R&D Intensity (%) - falls verfügbar
    if 'xrd' in df.columns and 'revt' in df.columns:
        df['rnd_intensity'] = np.where(
            (df['xrd'].notna()) & (df['revt'] > 0),
            (df['xrd'] / df['revt']) * 100,
            np.nan
        )
        logger.info("  ✓ R&D Intensity berechnet")
    else:
        if 'xrd' not in df.columns:
            logger.warning("  ⚠ R&D Intensity nicht berechnet - xrd Spalte fehlt")

    # Dividend Payout Ratio (%) - verwende 'ib' als Proxy für 'ni'
    net_income_col = 'ni' if 'ni' in df.columns else 'ib'
    if 'dvt' in df.columns and net_income_col in df.columns:
        df['dividend_payout_ratio'] = np.where(
            (df[net_income_col].notna()) & (df[net_income_col] > 0),
            (df['dvt'].fillna(0) / df[net_income_col]) * 100,
            np.nan
        )
        logger.info(f"  ✓ Dividend Payout Ratio berechnet (using {net_income_col})")
    else:
        if 'dvt' not in df.columns:
            logger.warning("  ⚠ Dividend Payout Ratio nicht berechnet - dvt Spalte fehlt")

    # Retention Ratio (%)
    if 'dividend_payout_ratio' in df.columns:
        df['retention_ratio'] = np.where(
            df['dividend_payout_ratio'].notna(),
            100 - df['dividend_payout_ratio'],
            np.nan
        )
        logger.info("  ✓ Retention Ratio berechnet")

    return df
