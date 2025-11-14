"""
Feature Coordinator Module
Orchestrates all feature engineering calculations and provides high-level API
"""

import pandas as pd
import logging
from pathlib import Path

from .ratio_calculators import (
    calculate_profitability_ratios,
    calculate_liquidity_ratios,
    calculate_leverage_ratios,
    calculate_efficiency_ratios
)
from .cashflow_calculators import (
    calculate_cashflow_metrics,
    calculate_structure_metrics
)
from .trend_calculators import (
    calculate_growth_metrics,
    calculate_dynamic_trends,
    smooth_with_cagr
)
from .volatility_calculators import (
    calculate_dynamic_volatility,
    handle_outliers,
    clean_calculated_features
)

logger = logging.getLogger(__name__)


def create_all_features(df, smooth_static=False, cagr_years=3):
    """
    Führt alle Feature Engineering Schritte aus.

    Args:
        df: Bereinigter Input DataFrame
        smooth_static: Führe CAGR-Glättung für statische Daten durch (default: False)
        cagr_years: Anzahl Jahre für CAGR-Glättung (default: 3)

    Returns:
        DataFrame mit allen berechneten Kennzahlen
    """
    logger.info("\n" + "="*50)
    logger.info("STARTE FEATURE ENGINEERING")
    logger.info("="*50 + "\n")

    # Optional: CAGR-Glättung für Basis-Daten (VOR Kennzahlen-Berechnung)
    if smooth_static:
        df = smooth_with_cagr(df, years=cagr_years)

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
