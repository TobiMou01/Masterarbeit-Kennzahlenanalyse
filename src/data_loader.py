"""
Modul 1: Data Loader
Lädt WRDS-Export CSV und bereinigt die Daten
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_german_number(value):
    """
    Konvertiert deutsche Zahlenformate zu Float.
    '73.340.000' -> 73340000.0
    '1.234,56' -> 1234.56
    """
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # String bereinigen
    value = str(value).strip()
    
    # Entferne Punkte (Tausendertrennzeichen)
    value = value.replace('.', '')
    # Ersetze Komma durch Punkt (Dezimaltrennzeichen)
    value = value.replace(',', '.')
    
    try:
        return float(value)
    except ValueError:
        return np.nan


def convert_german_date(date_str):
    """
    Konvertiert deutsches Datum zu datetime.
    '31.12.14' -> datetime(2014, 12, 31)
    """
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        # Format: DD.MM.YY
        return pd.to_datetime(date_str, format='%d.%m.%y')
    except:
        try:
            # Alternative Formate
            return pd.to_datetime(date_str)
        except:
            return pd.NaT


def load_data(filepath):
    """
    Lädt CSV-Datei mit korrektem Delimiter und Encoding.
    Erkennt automatisch ob Semikolon (;) oder Komma (,) als Delimiter verwendet wird.

    Args:
        filepath: Pfad zur CSV-Datei

    Returns:
        DataFrame mit geladenen Daten
    """
    logger.info(f"Lade Daten aus: {filepath}")

    try:
        # Versuche zunächst mit Semikolon (deutsches Format)
        try:
            df = pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)
            # Prüfe ob erfolgreich (mehr als 1 Spalte)
            if len(df.columns) > 1:
                logger.info(f"✓ {len(df)} Zeilen und {len(df.columns)} Spalten geladen (Delimiter: ';')")
                return df
        except:
            pass

        # Falls Semikolon nicht funktioniert, versuche Komma
        df = pd.read_csv(filepath, sep=',', encoding='utf-8', low_memory=False)
        logger.info(f"✓ {len(df)} Zeilen und {len(df.columns)} Spalten geladen (Delimiter: ',')")
        return df

    except Exception as e:
        logger.error(f"Fehler beim Laden der Datei: {e}")
        raise


def load_all_csv_from_directory(directory_path):
    """
    Lädt alle CSV-Dateien aus einem Verzeichnis und kombiniert sie.

    Args:
        directory_path: Pfad zum Verzeichnis mit CSV-Dateien

    Returns:
        Kombinierter DataFrame mit allen geladenen Daten
    """
    directory = Path(directory_path)

    if not directory.exists():
        logger.error(f"Verzeichnis existiert nicht: {directory}")
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")

    # Alle CSV-Dateien finden
    csv_files = list(directory.glob('*.csv'))

    if not csv_files:
        logger.warning(f"⚠ Keine CSV-Dateien gefunden in: {directory}")
        raise FileNotFoundError(f"Keine CSV-Dateien im Verzeichnis: {directory}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Gefundene CSV-Dateien in {directory}:")
    for i, file in enumerate(csv_files, 1):
        logger.info(f"  {i}. {file.name}")
    logger.info(f"{'='*60}\n")

    # Alle Dateien laden und kombinieren
    dataframes = []

    for csv_file in csv_files:
        logger.info(f"\n--- Lade Datei: {csv_file.name} ---")
        try:
            df = load_data(csv_file)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Fehler beim Laden von {csv_file.name}: {e}")
            logger.warning(f"⚠ Überspringe Datei: {csv_file.name}")
            continue

    if not dataframes:
        logger.error("Keine Dateien erfolgreich geladen!")
        raise ValueError("Alle CSV-Dateien konnten nicht geladen werden")

    # DataFrames kombinieren
    logger.info(f"\n{'='*60}")
    logger.info(f"Kombiniere {len(dataframes)} DataFrames...")

    combined_df = pd.concat(dataframes, ignore_index=True)

    logger.info(f"✓ Kombinierter DataFrame erstellt:")
    logger.info(f"  Gesamt Zeilen: {len(combined_df)}")
    logger.info(f"  Gesamt Spalten: {len(combined_df.columns)}")
    logger.info(f"{'='*60}\n")

    return combined_df


def clean_numeric_columns(df):
    """
    Bereinigt numerische Spalten mit deutschem Format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame mit bereinigten numerischen Spalten
    """
    logger.info("Bereinige numerische Spalten...")
    
    # Liste der Spalten die sicher Text sind
    text_columns = ['fic', 'costat', 'datafmt', 'indfmt', 'consol', 'conm', 
                    'isin', 'sedol', 'add1', 'add2', 'add3', 'add4', 'busdesc',
                    'city', 'conml', 'county', 'incorp', 'loc', 'state', 
                    'weburl', 'acctstd', 'bspr', 'curcd']
    
    # Datum-Spalten
    date_columns = ['datadate', 'dldte', 'ipodate', 'fdate', 'pdate']
    
    # Datum-Spalten konvertieren
    for col in date_columns:
        if col in df.columns:
            logger.info(f"  Konvertiere Datum: {col}")
            df[col] = df[col].apply(convert_german_date)
    
    # Alle anderen Spalten (außer Text) als numerisch behandeln
    for col in df.columns:
        if col not in text_columns and col not in date_columns:
            # Versuche numerische Konvertierung
            if df[col].dtype == 'object':
                logger.debug(f"  Konvertiere zu numerisch: {col}")
                df[col] = df[col].apply(convert_german_number)
    
    logger.info("✓ Numerische Spalten bereinigt")
    return df


def clean_data(df):
    """
    Bereinigt DataFrame: Duplikate, leere Spalten, etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        tuple: (Bereinigter DataFrame, Data Quality Report)
    """
    logger.info("Starte Datenbereinigung...")
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # Numerische Spalten bereinigen
    df = clean_numeric_columns(df)
    
    # 1. Entferne komplett leere Spalten
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        logger.info(f"Entferne {len(empty_cols)} leere Spalten")
        df = df.drop(columns=empty_cols)
    
    # 2. Entferne Duplikate basierend auf gvkey + datadate
    if 'gvkey' in df.columns and 'datadate' in df.columns:
        duplicates = df.duplicated(subset=['gvkey', 'datadate'], keep='first')
        n_duplicates = duplicates.sum()
        if n_duplicates > 0:
            logger.info(f"Entferne {n_duplicates} Duplikate (gvkey + datadate)")
            df = df[~duplicates]
    
    # 3. Data Quality Report erstellen
    report = {
        'initial_rows': initial_rows,
        'initial_columns': initial_cols,
        'final_rows': len(df),
        'final_columns': len(df.columns),
        'rows_removed': initial_rows - len(df),
        'columns_removed': initial_cols - len(df.columns),
        'missing_values_per_column': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict()
    }
    
    logger.info(f"✓ Bereinigung abgeschlossen:")
    logger.info(f"  Zeilen: {initial_rows} → {len(df)} ({report['rows_removed']} entfernt)")
    logger.info(f"  Spalten: {initial_cols} → {len(df.columns)} ({report['columns_removed']} entfernt)")
    
    return df, report


def filter_relevant_columns(df):
    """
    Behält nur für Finanzkennzahlen relevante Spalten.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame mit gefilterten Spalten
    """
    logger.info("Filtere relevante Spalten...")
    
    # Relevante Spalten definieren
    relevant_columns = {
        # Identifikation
        'gvkey', 'datadate', 'conm', 'isin', 'loc', 'fyear', 'sic', 'naics',

        # GICS Classification (für Vergleichsanalysen)
        'gsector', 'ggroup', 'gind', 'gsubind',
        
        # Bilanz - Assets
        'at',      # Total Assets
        'act',     # Current Assets
        'che',     # Cash and Equivalents
        'rect',    # Receivables
        'invt',    # Inventories
        'ppent',   # Property, Plant & Equipment Net
        
        # Bilanz - Liabilities & Equity
        'lt',      # Total Liabilities
        'lct',     # Current Liabilities
        'dlc',     # Debt in Current Liabilities
        'dltt',    # Long-term Debt
        'seq',     # Stockholders Equity
        'ceq',     # Common Equity
        
        # GuV
        'revt',    # Revenue Total
        'sale',    # Sales/Revenue
        'cogs',    # Cost of Goods Sold
        'xsga',    # Selling, General & Administrative Expense
        'ebit',    # EBIT
        'ebitda',  # EBITDA
        'ib',      # Income Before Extraordinary Items (Net Income)
        'ni',      # Net Income
        'oibdp',   # Operating Income Before Depreciation
        
        # Cashflow
        'oancf',   # Operating Activities Net Cash Flow
        
        # Sonstiges
        'emp',     # Employees
        'dp',      # Depreciation and Amortization
    }
    
    # Nur Spalten behalten die existieren
    available_columns = [col for col in relevant_columns if col in df.columns]
    missing_columns = relevant_columns - set(available_columns)
    
    if missing_columns:
        logger.warning(f"Fehlende Spalten: {missing_columns}")
    
    df_filtered = df[available_columns].copy()
    logger.info(f"✓ {len(available_columns)} relevante Spalten behalten")
    
    return df_filtered


def save_cleaned_data(df, report, output_dir='data/processed'):
    """
    Speichert bereinigte Daten und Report.
    
    Args:
        df: Bereinigter DataFrame
        report: Data Quality Report
        output_dir: Ausgabe-Verzeichnis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Bereinigte Daten speichern
    csv_path = output_path / 'cleaned_data.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Bereinigte Daten gespeichert: {csv_path}")
    
    # Report speichern
    report_path = output_path / 'data_quality_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== DATA QUALITY REPORT ===\n\n")
        f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Initial Zeilen: {report['initial_rows']}\n")
        f.write(f"Final Zeilen: {report['final_rows']}\n")
        f.write(f"Entfernte Zeilen: {report['rows_removed']}\n\n")
        f.write(f"Initial Spalten: {report['initial_columns']}\n")
        f.write(f"Final Spalten: {report['final_columns']}\n")
        f.write(f"Entfernte Spalten: {report['columns_removed']}\n\n")
        f.write("=== FEHLENDE WERTE PRO SPALTE (Top 10) ===\n")
        
        # Top 10 Spalten mit meisten fehlenden Werten
        missing_sorted = sorted(
            report['missing_percentage'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        for col, pct in missing_sorted:
            f.write(f"{col}: {pct:.1f}%\n")
    
    logger.info(f"✓ Report gespeichert: {report_path}")


def main():
    """Beispiel-Verwendung des Moduls."""

    # Pfad zum Verzeichnis mit CSV-Dateien
    data_directory = 'data/raw'

    # 1. Alle CSV-Dateien aus dem Verzeichnis laden
    df = load_all_csv_from_directory(data_directory)

    # 2. Daten bereinigen
    df_cleaned, report = clean_data(df)

    # 3. Relevante Spalten filtern
    df_final = filter_relevant_columns(df_cleaned)

    # 4. Speichern
    save_cleaned_data(df_final, report)

    logger.info("\n✅ Datenbereinigung abgeschlossen!")
    logger.info(f"Finale Daten: {len(df_final)} Zeilen, {len(df_final.columns)} Spalten")

    return df_final


if __name__ == "__main__":
    df = main()
    print("\n" + "="*50)
    print("ÜBERSICHT DER BEREINIGTEN DATEN")
    print("="*50)
    print(f"\nShape: {df.shape}")
    print(f"\nKolonnen:\n{df.columns.tolist()}")
    print(f"\nErste 3 Zeilen:")
    print(df.head(3))
    print(f"\nFehlende Werte pro Spalte:")
    print(df.isna().sum().sort_values(ascending=False).head(10))