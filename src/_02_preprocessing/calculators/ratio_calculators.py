"""
Ratio Calculators Module
Berechnet Profitabilitäts-, Liquiditäts-, Verschuldungs- und Effizienz-Kennzahlen
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_profitability_ratios(df):
    """
    Berechnet Profitabilitätskennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Profitabilitätskennzahlen
    """
    logger.info("Berechne Profitabilitätskennzahlen...")

    df = df.copy()

    # ROA - Return on Assets (%)
    if 'ebit' in df.columns and 'at' in df.columns:
        df['roa'] = (df['ebit'] / df['at']) * 100
        logger.info("  ✓ ROA berechnet")

    # ROE - Return on Equity (%)
    if 'ib' in df.columns and 'seq' in df.columns:
        df['roe'] = (df['ib'] / df['seq']) * 100
        logger.info("  ✓ ROE berechnet")

    # EBIT Margin (%)
    if 'ebit' in df.columns and 'revt' in df.columns:
        df['ebit_margin'] = (df['ebit'] / df['revt']) * 100
        logger.info("  ✓ EBIT Margin berechnet")

    # EBITDA Margin (%)
    if 'ebitda' in df.columns and 'revt' in df.columns:
        df['ebitda_margin'] = (df['ebitda'] / df['revt']) * 100
        logger.info("  ✓ EBITDA Margin berechnet")

    # Net Profit Margin (%)
    if 'ib' in df.columns and 'revt' in df.columns:
        df['net_profit_margin'] = (df['ib'] / df['revt']) * 100
        logger.info("  ✓ Net Profit Margin berechnet")

    # Operating Margin (%)
    if 'oibdp' in df.columns and 'revt' in df.columns:
        df['operating_margin'] = (df['oibdp'] / df['revt']) * 100
        logger.info("  ✓ Operating Margin berechnet")

    # Gross Margin (%)
    if 'revt' in df.columns and 'cogs' in df.columns:
        df['gross_margin'] = ((df['revt'] - df['cogs']) / df['revt']) * 100
        logger.info("  ✓ Gross Margin berechnet")

    # ROC - Return on Capital (%)
    if 'ebit' in df.columns and 'seq' in df.columns and 'dltt' in df.columns:
        capital = df['seq'] + df['dltt']
        df['roc'] = np.where(capital > 0, (df['ebit'] / capital) * 100, np.nan)
        logger.info("  ✓ ROC (Return on Capital) berechnet")

    return df


def calculate_liquidity_ratios(df):
    """
    Berechnet Liquiditätskennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Liquiditätskennzahlen
    """
    logger.info("Berechne Liquiditätskennzahlen...")

    df = df.copy()

    # Current Ratio
    if 'act' in df.columns and 'lct' in df.columns:
        df['current_ratio'] = df['act'] / df['lct']
        logger.info("  ✓ Current Ratio berechnet")

    # Quick Ratio (ohne Inventar)
    if 'act' in df.columns and 'invt' in df.columns and 'lct' in df.columns:
        df['quick_ratio'] = (df['act'] - df['invt']) / df['lct']
        logger.info("  ✓ Quick Ratio berechnet")

    # Cash Ratio
    if 'che' in df.columns and 'lct' in df.columns:
        df['cash_ratio'] = df['che'] / df['lct']
        logger.info("  ✓ Cash Ratio berechnet")

    # Cash Ratio Enhanced (mit kurzfristigen Investments)
    if 'che' in df.columns and 'lct' in df.columns:
        # Falls ivst (Short-term investments) verfügbar, sonst nur che
        if 'ivst' in df.columns:
            df['cash_ratio_enhanced'] = (df['che'] + df['ivst'].fillna(0)) / df['lct']
        else:
            df['cash_ratio_enhanced'] = df['che'] / df['lct']
        logger.info("  ✓ Cash Ratio Enhanced berechnet")

    # Working Capital Ratio (%)
    if 'act' in df.columns and 'lct' in df.columns and 'at' in df.columns:
        working_capital = df['act'] - df['lct']
        df['working_capital_ratio'] = (working_capital / df['at']) * 100
        logger.info("  ✓ Working Capital Ratio berechnet")

    return df


def calculate_leverage_ratios(df):
    """
    Berechnet Verschuldungskennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Verschuldungskennzahlen
    """
    logger.info("Berechne Verschuldungskennzahlen...")

    df = df.copy()

    # Debt-to-Equity Ratio
    if 'dltt' in df.columns and 'seq' in df.columns:
        df['debt_to_equity'] = df['dltt'] / df['seq']
        logger.info("  ✓ Debt-to-Equity berechnet")

    # Total Debt-to-Equity (inkl. kurzfristiger Schulden)
    if 'dlc' in df.columns and 'dltt' in df.columns and 'seq' in df.columns:
        df['total_debt_to_equity'] = (df['dlc'] + df['dltt']) / df['seq']
        logger.info("  ✓ Total Debt-to-Equity berechnet")

    # Equity Ratio (%)
    if 'seq' in df.columns and 'at' in df.columns:
        df['equity_ratio'] = (df['seq'] / df['at']) * 100
        logger.info("  ✓ Equity Ratio berechnet")

    # Debt Ratio (%)
    if 'lt' in df.columns and 'at' in df.columns:
        df['debt_ratio'] = (df['lt'] / df['at']) * 100
        logger.info("  ✓ Debt Ratio berechnet")

    # Interest Coverage (EBIT / Interest Expense)
    if 'ebit' in df.columns and 'xint' in df.columns:
        # Nur berechnen wenn Zinsen > 0, sonst np.nan
        df['interest_coverage'] = np.where(
            (df['xint'].notna()) & (df['xint'] > 0),
            df['ebit'] / df['xint'],
            np.nan
        )
        logger.info("  ✓ Interest Coverage berechnet")
    else:
        if 'xint' not in df.columns:
            logger.warning("  ⚠ Interest Coverage nicht berechnet - xint Spalte fehlt")

    # Net Debt to EBITDA
    if 'dlc' in df.columns and 'dltt' in df.columns and 'che' in df.columns and 'ebitda' in df.columns:
        total_debt = df['dlc'].fillna(0) + df['dltt'].fillna(0)
        net_debt = total_debt - df['che']
        df['net_debt_to_ebitda'] = np.where(df['ebitda'] > 0, net_debt / df['ebitda'], np.nan)
        logger.info("  ✓ Net Debt to EBITDA berechnet")

    # Debt to Assets (%)
    if 'dlc' in df.columns and 'dltt' in df.columns and 'at' in df.columns:
        total_debt = df['dlc'].fillna(0) + df['dltt'].fillna(0)
        df['debt_to_assets'] = (total_debt / df['at']) * 100
        logger.info("  ✓ Debt to Assets berechnet")

    return df


def calculate_efficiency_ratios(df):
    """
    Berechnet Effizienz- und Aktivitätskennzahlen.

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit zusätzlichen Effizienzkennzahlen
    """
    logger.info("Berechne Effizienzkennzahlen...")

    df = df.copy()

    # Asset Turnover
    if 'revt' in df.columns and 'at' in df.columns:
        df['asset_turnover'] = df['revt'] / df['at']
        logger.info("  ✓ Asset Turnover berechnet")

    # Revenue per Employee (in Tausend)
    if 'revt' in df.columns and 'emp' in df.columns:
        # emp ist oft schon in Hunderten, revt in Tausenden
        df['revenue_per_employee'] = df['revt'] / df['emp']
        logger.info("  ✓ Revenue per Employee berechnet")

    # Receivables Turnover
    if 'revt' in df.columns and 'rect' in df.columns:
        df['receivables_turnover'] = df['revt'] / df['rect']
        logger.info("  ✓ Receivables Turnover berechnet")

    # Days Sales Outstanding (DSO)
    if 'receivables_turnover' in df.columns:
        df['days_sales_outstanding'] = 365 / df['receivables_turnover']
        logger.info("  ✓ Days Sales Outstanding berechnet")

    # Capital Intensity (%)
    if 'ppent' in df.columns and 'revt' in df.columns:
        df['capital_intensity'] = (df['ppent'] / df['revt']) * 100
        logger.info("  ✓ Capital Intensity berechnet")

    # Working Capital Turnover
    if 'revt' in df.columns and 'act' in df.columns and 'lct' in df.columns:
        working_capital = df['act'] - df['lct']
        df['working_capital_turnover'] = np.where(working_capital > 0, df['revt'] / working_capital, np.nan)
        logger.info("  ✓ Working Capital Turnover berechnet")

    # Inventory Turnover
    if 'cogs' in df.columns and 'invt' in df.columns:
        df['inventory_turnover'] = np.where(df['invt'] > 0, df['cogs'] / df['invt'], np.nan)
        logger.info("  ✓ Inventory Turnover berechnet")

    # Asset Quality (%)
    if 'che' in df.columns and 'rect' in df.columns and 'at' in df.columns:
        liquid_assets = df['che'] + df['rect'].fillna(0)
        df['asset_quality'] = (liquid_assets / df['at']) * 100
        logger.info("  ✓ Asset Quality berechnet")

    return df
