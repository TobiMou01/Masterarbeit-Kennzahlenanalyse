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
    
    # Interest Coverage (EBIT / Zinsen) - wenn Zinsdaten verfügbar
    # Note: Oft nicht direkt in Daten, würde 'xint' brauchen
    
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
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'growth', 'equity']
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
    
    # Alle Kennzahlen berechnen
    df = calculate_profitability_ratios(df)
    df = calculate_liquidity_ratios(df)
    df = calculate_leverage_ratios(df)
    df = calculate_efficiency_ratios(df)
    df = calculate_growth_metrics(df)
    
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
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'growth', 'equity']
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
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'growth', 'equity']
    ) and 'outlier' not in col]
    
    for col in new_cols:
        print(f"  - {col}")
    
    print(f"\n\nSummary Statistics:")
    print(summary)