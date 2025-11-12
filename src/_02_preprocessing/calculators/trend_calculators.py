"""
Trend Calculators Module
Berechnet Wachstums-, Trend- und Glättungs-Kennzahlen
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        df['employee_growth'] = df.groupby('gvkey')['emp'].pct_change(fill_method=None) * 100
        logger.info("  ✓ Employee Growth berechnet")

    # FCF Growth (%)
    if 'fcf' in df.columns:
        df['fcf_growth'] = df.groupby('gvkey')['fcf'].pct_change(fill_method=None) * 100
        logger.info("  ✓ FCF Growth berechnet")
    else:
        logger.warning("  ⚠ FCF Growth nicht berechnet - fcf muss zuerst berechnet werden")

    # Capex Growth (%)
    if 'capx' in df.columns:
        df['capex_abs'] = df['capx'].abs()
        df['capex_growth'] = df.groupby('gvkey')['capex_abs'].pct_change(fill_method=None) * 100
        df = df.drop('capex_abs', axis=1)
        logger.info("  ✓ Capex Growth berechnet")
    else:
        logger.warning("  ⚠ Capex Growth nicht berechnet - capx Spalte fehlt")

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
    else:
        logger.warning("  ⚠ FCF Trend nicht berechnet - fcf_margin muss zuerst berechnet werden")

    return df


def smooth_with_cagr(df, years=3, fields=None):
    """
    Glättet statische Finanzdaten mit N-Jahres-CAGR.

    Anwendung: Reduziert Volatilität in Basisdaten (Revenue, Assets, etc.)
    für stabilere Clustering-Ergebnisse.

    Args:
        df: DataFrame mit Finanzdaten
        years: Anzahl Jahre für CAGR (default: 3)
        fields: Liste der zu glättenden Felder (default: wichtige Basis-Felder)

    Returns:
        DataFrame mit geglätteten Werten
    """
    logger.info(f"Glättung mit {years}-Jahres-CAGR...")

    df = df.copy()

    if 'gvkey' not in df.columns or 'fyear' not in df.columns:
        logger.warning("  ⚠ gvkey/fyear fehlt - überspringe CAGR-Glättung")
        return df

    # Standard-Felder für Glättung (wichtige Basisdaten)
    if fields is None:
        fields = ['revt', 'at', 'ebit', 'ebitda', 'oancf', 'capx', 'ib']

    # Nur verfügbare Felder
    available_fields = [f for f in fields if f in df.columns]

    if not available_fields:
        logger.warning("  ⚠ Keine Felder zum Glätten gefunden")
        return df

    # Nach Unternehmen und Jahr sortieren
    df = df.sort_values(['gvkey', 'fyear'])

    for field in available_fields:
        # Berechne rolling CAGR
        # CAGR = (End_Value / Start_Value)^(1/years) - 1

        def calc_cagr(group):
            """Berechnet CAGR für eine Gruppe"""
            if len(group) < years:
                return group  # Zu wenig Jahre

            # Rolling window
            group_sorted = group.sort_values('fyear')
            smoothed = group_sorted[field].copy()

            for i in range(years, len(group_sorted)):
                start_val = group_sorted[field].iloc[i - years]
                end_val = group_sorted[field].iloc[i]

                if pd.notna(start_val) and pd.notna(end_val) and start_val > 0 and end_val > 0:
                    cagr = (end_val / start_val) ** (1 / years) - 1
                    # Geglätteter Wert: Start * (1 + CAGR)^years
                    smoothed.iloc[i] = start_val * ((1 + cagr) ** years)

            group_sorted[f'{field}_smoothed'] = smoothed
            return group_sorted

        # Gruppiere und glätte
        df = df.groupby('gvkey', group_keys=False).apply(lambda g: calc_cagr(g) if len(g) >= years else g)

        # Verwende geglättete Werte (falls vorhanden)
        if f'{field}_smoothed' in df.columns:
            # Ersetze Originalwerte durch geglättete (optional: behalte Original als backup)
            df[f'{field}_original'] = df[field]
            df[field] = df[f'{field}_smoothed'].fillna(df[field])
            df = df.drop(f'{field}_smoothed', axis=1)
            logger.info(f"  ✓ {field} geglättet ({years}-Jahres-CAGR)")

    return df
