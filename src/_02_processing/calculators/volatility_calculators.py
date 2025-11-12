"""
Volatility Calculators Module
Berechnet Volatilitäts-, Konsistenz- und Qualitäts-Kennzahlen
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_dynamic_volatility(df):
    """
    Berechnet Volatilitäts- und Qualitätskennzahlen über Zeit.
    Requires: gvkey für Gruppierung

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit Volatilitäts-Kennzahlen
    """
    logger.info("Berechne dynamische Volatilitäts-Kennzahlen...")

    df = df.copy()

    if 'gvkey' not in df.columns:
        logger.warning("  ⚠ gvkey fehlt - überspringe Volatilitäts-Kennzahlen")
        return df

    from scipy import stats

    def calculate_volatility(group, column):
        """Berechnet Standardabweichung"""
        if len(group) < 3 or column not in group.columns:
            return np.nan
        valid = group[column].dropna()
        if len(valid) < 3:
            return np.nan
        return valid.std()

    def calculate_consistency(group, column):
        """Berechnet R² der linearen Regression (0-1)"""
        if len(group) < 3 or column not in group.columns or 'fyear' not in group.columns:
            return np.nan
        valid = group[[column, 'fyear']].dropna()
        if len(valid) < 3:
            return np.nan
        try:
            _, _, r_value, _, _ = stats.linregress(valid['fyear'], valid[column])
            return r_value ** 2
        except:
            return np.nan

    def calculate_correlation(group, col1, col2):
        """Berechnet Korrelation zwischen zwei Kennzahlen"""
        if len(group) < 3 or col1 not in group.columns or col2 not in group.columns:
            return np.nan
        valid = group[[col1, col2]].dropna()
        if len(valid) < 3:
            return np.nan
        try:
            return valid[col1].corr(valid[col2])
        except:
            return np.nan

    # Margin Volatility
    if 'ebit_margin' in df.columns:
        df['margin_volatility'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_volatility(g, 'ebit_margin')] * len(g), index=g.index)
        )
        logger.info("  ✓ Margin Volatility berechnet")

    # Leverage Volatility
    if 'total_debt_to_equity' in df.columns:
        df['leverage_volatility'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_volatility(g, 'total_debt_to_equity')] * len(g), index=g.index)
        )
        logger.info("  ✓ Leverage Volatility berechnet")

    # Cashflow Volatility (Operating CF / Assets)
    if 'oancf' in df.columns and 'at' in df.columns:
        df['cf_to_assets'] = (df['oancf'] / df['at']) * 100
        df['cashflow_volatility'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_volatility(g, 'cf_to_assets')] * len(g), index=g.index)
        )
        logger.info("  ✓ Cashflow Volatility berechnet")

    # Margin Consistency (R² of margin trend)
    if 'ebit_margin' in df.columns and 'fyear' in df.columns:
        df['margin_consistency'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_consistency(g, 'ebit_margin')] * len(g), index=g.index)
        )
        logger.info("  ✓ Margin Consistency berechnet")

    # Growth Quality (Korrelation zwischen Revenue Growth und FCF Growth)
    if 'revenue_growth' in df.columns and 'fcf_growth' in df.columns:
        df['growth_quality'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_correlation(g, 'revenue_growth', 'fcf_growth')] * len(g), index=g.index)
        )
        logger.info("  ✓ Growth Quality berechnet")
    else:
        if 'fcf_growth' not in df.columns:
            logger.warning("  ⚠ Growth Quality nicht berechnet - fcf_growth muss zuerst berechnet werden")

    return df


def handle_outliers(df, method='iqr', threshold=3):
    """
    Identifiziert und markiert Ausreißer.

    Args:
        df: DataFrame
        method: 'iqr' oder 'zscore'
        threshold: IQR-Multiplikator oder Z-Score Schwellenwert

    Returns:
        DataFrame mit Ausreißer-Flags
    """
    logger.info(f"Identifiziere Ausreißer mit Methode: {method}...")

    df = df.copy()

    # Nur numerische Spalten die Kennzahlen sind
    ratio_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'roc', 'growth', 'equity',
         'coverage', 'leverage', 'intensity', 'quality', 'trend', 'volatility',
         'consistency', 'fcf', 'capex', 'reinvestment', 'conversion', 'payout',
         'retention', 'rnd', 'debt', 'per_employee', 'days_', 'ebit', 'ebitda']
    )]

    for col in ratio_columns:
        if df[col].dtype in ['float64', 'int64']:

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                df[f'{col}_outlier'] = outliers

                n_outliers = outliers.sum()
                if n_outliers > 0:
                    logger.info(f"  {col}: {n_outliers} Ausreißer gefunden")

    return df


def clean_calculated_features(df):
    """
    Bereinigt berechnete Kennzahlen von Inf und extremen Werten.

    Args:
        df: DataFrame mit berechneten Kennzahlen

    Returns:
        Bereinigter DataFrame
    """
    logger.info("Bereinige berechnete Kennzahlen...")

    df = df.copy()

    # Ersetze Inf mit NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Log extreme Werte
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and 'outlier' not in col:
            if df[col].max() > 1e6 or df[col].min() < -1e6:
                logger.warning(f"  ⚠ Extreme Werte in {col}: min={df[col].min():.0f}, max={df[col].max():.0f}")

    return df
