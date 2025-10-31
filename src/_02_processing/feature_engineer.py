"""
Modul 2: Feature Engineering
Berechnet Finanzkennzahlen aus bereinigten Daten
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        df['interest_coverage'] = np.where(df['xint'] > 0, df['ebit'] / df['xint'], np.nan)
        logger.info("  ✓ Interest Coverage berechnet")

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
        df['capex_to_revenue'] = (df['capx'].abs() / df['revt']) * 100
        logger.info("  ✓ Capex to Revenue berechnet")

    # Capex to Depreciation
    if 'capx' in df.columns and 'dp' in df.columns:
        df['capex_to_depreciation'] = np.where(df['dp'] > 0, df['capx'].abs() / df['dp'], np.nan)
        logger.info("  ✓ Capex to Depreciation berechnet")

    # Free Cash Flow (FCF)
    if 'oancf' in df.columns and 'capx' in df.columns:
        df['fcf'] = df['oancf'] - df['capx'].abs()
        logger.info("  ✓ Free Cash Flow berechnet")

    # FCF Margin (%)
    if 'fcf' in df.columns and 'revt' in df.columns:
        df['fcf_margin'] = (df['fcf'] / df['revt']) * 100
        logger.info("  ✓ FCF Margin berechnet")

    # Reinvestment Rate (%)
    if 'capx' in df.columns and 'oancf' in df.columns:
        df['reinvestment_rate'] = np.where(df['oancf'] > 0, (df['capx'].abs() / df['oancf']) * 100, np.nan)
        logger.info("  ✓ Reinvestment Rate berechnet")

    # Cash Conversion (Operating CF / EBIT) (%)
    if 'oancf' in df.columns and 'ebit' in df.columns:
        df['cash_conversion'] = np.where(df['ebit'] > 0, (df['oancf'] / df['ebit']) * 100, np.nan)
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
        df['financial_leverage'] = np.where(df['seq'] > 0, df['at'] / df['seq'], np.nan)
        logger.info("  ✓ Financial Leverage berechnet")

    # R&D Intensity (%) - falls verfügbar
    if 'xrd' in df.columns and 'revt' in df.columns:
        df['rnd_intensity'] = (df['xrd'] / df['revt']) * 100
        logger.info("  ✓ R&D Intensity berechnet")

    # Dividend Payout Ratio (%)
    if 'dvt' in df.columns and 'ni' in df.columns:
        df['dividend_payout_ratio'] = np.where(df['ni'] > 0, (df['dvt'] / df['ni']) * 100, np.nan)
        logger.info("  ✓ Dividend Payout Ratio berechnet")

    # Retention Ratio (%)
    if 'dividend_payout_ratio' in df.columns:
        df['retention_ratio'] = 100 - df['dividend_payout_ratio']
        logger.info("  ✓ Retention Ratio berechnet")

    return df


def calculate_growth_metrics(df):
    """
    Berechnet Wachstumskennzahlen (Jahr-zu-Jahr).
    Requires: datadate und gvkey für Sortierung
    
    Args:
        df: DataFrame mit Finanzdaten
        
    Returns:
        DataFrame mit Wachstumskennzahlen
    """
    logger.info("Berechne Wachstumskennzahlen...")
    
    df = df.copy()
    
    if 'gvkey' not in df.columns or 'datadate' not in df.columns:
        logger.warning("  ⚠ gvkey oder datadate fehlt - überspringe Wachstumskennzahlen")
        return df
    
    # Nach Unternehmen und Datum sortieren
    df = df.sort_values(['gvkey', 'datadate'])
    
    # Revenue Growth (%)
    if 'revt' in df.columns:
        df['revenue_growth'] = df.groupby('gvkey')['revt'].pct_change() * 100
        logger.info("  ✓ Revenue Growth berechnet")
    
    # Asset Growth (%)
    if 'at' in df.columns:
        df['asset_growth'] = df.groupby('gvkey')['at'].pct_change() * 100
        logger.info("  ✓ Asset Growth berechnet")
    
    # Employee Growth (%)
    if 'emp' in df.columns:
        df['employee_growth'] = df.groupby('gvkey')['emp'].pct_change() * 100
        logger.info("  ✓ Employee Growth berechnet")

    # FCF Growth (%)
    if 'fcf' in df.columns:
        df['fcf_growth'] = df.groupby('gvkey')['fcf'].pct_change() * 100
        logger.info("  ✓ FCF Growth berechnet")

    # Capex Growth (%)
    if 'capx' in df.columns:
        df['capex_abs'] = df['capx'].abs()
        df['capex_growth'] = df.groupby('gvkey')['capex_abs'].pct_change() * 100
        df = df.drop('capex_abs', axis=1)
        logger.info("  ✓ Capex Growth berechnet")

    return df


def calculate_dynamic_trends(df):
    """
    Berechnet Trend-Kennzahlen über Zeit (lineare Regression).
    Requires: gvkey und fyear für Zeitreihen

    Args:
        df: DataFrame mit Finanzdaten

    Returns:
        DataFrame mit Trend-Kennzahlen
    """
    logger.info("Berechne dynamische Trend-Kennzahlen...")

    df = df.copy()

    if 'gvkey' not in df.columns or 'fyear' not in df.columns:
        logger.warning("  ⚠ gvkey oder fyear fehlt - überspringe Trend-Kennzahlen")
        return df

    from scipy import stats

    def calculate_trend(group, column):
        """Berechnet Slope der linearen Regression"""
        if len(group) < 3 or column not in group.columns:
            return np.nan
        valid = group[[column, 'fyear']].dropna()
        if len(valid) < 3:
            return np.nan
        try:
            slope, _, _, _, _ = stats.linregress(valid['fyear'], valid[column])
            return slope
        except:
            return np.nan

    # Margin Trend (EBIT Margin)
    if 'ebit_margin' in df.columns:
        df['margin_trend'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_trend(g, 'ebit_margin')] * len(g), index=g.index)
        )
        logger.info("  ✓ Margin Trend berechnet")

    # Leverage Trend (Debt-to-Equity)
    if 'total_debt_to_equity' in df.columns:
        df['leverage_trend'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_trend(g, 'total_debt_to_equity')] * len(g), index=g.index)
        )
        logger.info("  ✓ Leverage Trend berechnet")

    # Capex Trend (Capex/Revenue)
    if 'capex_to_revenue' in df.columns:
        df['capex_trend'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_trend(g, 'capex_to_revenue')] * len(g), index=g.index)
        )
        logger.info("  ✓ Capex Trend berechnet")

    # FCF Trend (FCF Margin)
    if 'fcf_margin' in df.columns:
        df['fcf_trend'] = df.groupby('gvkey', group_keys=False).apply(
            lambda g: pd.Series([calculate_trend(g, 'fcf_margin')] * len(g), index=g.index)
        )
        logger.info("  ✓ FCF Trend berechnet")

    return df


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


def create_all_features(df):
    """
    Führt alle Feature Engineering Schritte aus.

    Args:
        df: Bereinigter Input DataFrame

    Returns:
        DataFrame mit allen berechneten Kennzahlen
    """
    logger.info("\n" + "="*50)
    logger.info("STARTE FEATURE ENGINEERING")
    logger.info("="*50 + "\n")

    # Statische Kennzahlen berechnen
    df = calculate_profitability_ratios(df)
    df = calculate_liquidity_ratios(df)
    df = calculate_leverage_ratios(df)
    df = calculate_efficiency_ratios(df)
    df = calculate_cashflow_metrics(df)
    df = calculate_structure_metrics(df)

    # Dynamische Kennzahlen berechnen (benötigen gvkey/fyear)
    df = calculate_growth_metrics(df)
    df = calculate_dynamic_trends(df)
    df = calculate_dynamic_volatility(df)

    # Bereinigen
    df = clean_calculated_features(df)

    # Optional: Ausreißer identifizieren
    df = handle_outliers(df, method='iqr', threshold=3)

    logger.info("\n✅ Feature Engineering abgeschlossen!")
    logger.info(f"Finale Spaltenanzahl: {len(df.columns)}")

    return df


def summary_statistics(df):
    """
    Erstellt deskriptive Statistiken für berechnete Kennzahlen.

    Args:
        df: DataFrame mit Kennzahlen

    Returns:
        DataFrame mit Summary Statistics
    """
    logger.info("Erstelle Summary Statistics...")

    # Nur Kennzahl-Spalten
    ratio_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'roc', 'growth', 'equity',
         'coverage', 'leverage', 'intensity', 'quality', 'trend', 'volatility',
         'consistency', 'fcf', 'capex', 'reinvestment', 'conversion', 'payout',
         'retention', 'rnd', 'debt', 'per_employee', 'days_', 'ebit', 'ebitda']
    ) and 'outlier' not in col]
    
    summary = df[ratio_columns].describe(percentiles=[.25, .5, .75, .90, .95])
    summary.loc['missing'] = df[ratio_columns].isna().sum()
    summary.loc['missing_pct'] = (df[ratio_columns].isna().sum() / len(df)) * 100
    
    return summary


def save_features(df, summary, output_dir='data/processed'):
    """
    Speichert Features und Summary Statistics.
    
    Args:
        df: DataFrame mit Features
        summary: Summary Statistics DataFrame
        output_dir: Ausgabeverzeichnis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Features speichern
    features_path = output_path / 'features.csv'
    df.to_csv(features_path, index=False)
    logger.info(f"✓ Features gespeichert: {features_path}")
    
    # Summary speichern
    summary_path = output_path / 'feature_summary.csv'
    summary.to_csv(summary_path)
    logger.info(f"✓ Summary gespeichert: {summary_path}")


def main():
    """Beispiel-Verwendung des Moduls."""
    
    # Bereinigte Daten laden
    input_path = 'data/processed/cleaned_data.csv'
    logger.info(f"Lade bereinigte Daten aus: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
    
    # Features berechnen
    df_features = create_all_features(df)
    
    # Summary Statistics
    summary = summary_statistics(df_features)
    
    # Speichern
    save_features(df_features, summary)
    
    logger.info("\n" + "="*50)
    logger.info("FEATURE ENGINEERING ABGESCHLOSSEN")
    logger.info("="*50)
    
    return df_features, summary


if __name__ == "__main__":
    df_features, summary = main()
    
    print("\n" + "="*50)
    print("ÜBERSICHT DER BERECHNETEN KENNZAHLEN")
    print("="*50)
    print(f"\nShape: {df_features.shape}")
    print(f"\nNeu berechnete Spalten:")

    new_cols = [col for col in df_features.columns if any(
        keyword in col.lower() for keyword in
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'roc', 'growth', 'equity',
         'coverage', 'leverage', 'intensity', 'quality', 'trend', 'volatility',
         'consistency', 'fcf', 'capex', 'reinvestment', 'conversion', 'payout',
         'retention', 'rnd', 'debt', 'per_employee', 'days_', 'ebit', 'ebitda']
    ) and 'outlier' not in col]

    # Gruppiere die Kennzahlen
    static_metrics = [col for col in new_cols if not any(
        kw in col.lower() for kw in ['growth', 'trend', 'volatility', 'consistency', 'quality']
    )]
    dynamic_metrics = [col for col in new_cols if any(
        kw in col.lower() for kw in ['growth', 'trend', 'volatility', 'consistency', 'quality']
    )]

    print(f"\n  STATISCHE KENNZAHLEN ({len(static_metrics)}):")
    for col in sorted(static_metrics):
        print(f"    - {col}")

    print(f"\n  DYNAMISCHE KENNZAHLEN ({len(dynamic_metrics)}):")
    for col in sorted(dynamic_metrics):
        print(f"    - {col}")
    
    print(f"\n\nSummary Statistics:")
    print(summary)