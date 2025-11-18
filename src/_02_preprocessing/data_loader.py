"""
Modul 1: Data Loader
Lädt WRDS-Export CSV und bereinigt die Daten

Multi-Market Support:
====================
Dieses Modul unterstützt zwei Lade-Modi:

1. LEGACY MODE: Einzelner Markt aus separaten CSV-Dateien
   - Nutzt load_market_data() Funktion
   - Lädt aus data/raw/{market}/ Ordner

2. MULTI-MARKET MODE: Flexible Länderauswahl aus großer CSV
   - Nutzt load_multi_market_data() Funktion
   - Lädt aus großer internationaler CSV (z.B. WRDS Global)
   - Filtert nach 'fic' (Foreign Incorporation Code) Spalte
   - Unterstützt Index-Proxies (Top N Unternehmen pro Land)
   - Sektor-Ausschluss (z.B. Financials)

PRESET CONFIGURATIONS:
----------------------
Vordefinierte Markt-Kombinationen für häufige Use-Cases:
- 'germany_dax_family': Top 160 deutsche Unternehmen
- 'europe_large_cap': Top 50 aus 5 EU-Ländern
- 'germany_vs_france': Vergleich DEU vs FRA
- 'all_europe': Alle EU-Unternehmen
- 'usa_large_cap': S&P 500 Proxy
- 'global_giants': Globale Top-Unternehmen

USAGE EXAMPLE:
--------------
# Mit Preset:
config = get_preset_config('germany_dax_family')
df = load_multi_market_data('international_data.csv', config)

# Manuell:
config = {
    'mode': 'multi',
    'countries': ['DEU', 'FRA'],
    'use_index_proxy': True,
    'index_proxies': {
        'DEU': {'type': 'top_n', 'n': 40},
        'FRA': {'type': 'top_n', 'n': 40}
    },
    'exclude_sectors': [40]  # Financials
}
df = load_multi_market_data('international_data.csv', config)

# Verfügbare Länder entdecken:
countries_df = discover_available_countries('international_data.csv')
print(countries_df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import re

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


def impute_missing_values(df, method='median', threshold=0.5):
    """
    Systematische Imputation fehlender Werte.

    Strategie:
    - Spalten mit >threshold missing → nicht imputieren (zu wenig Daten)
    - Restliche Spalten: Median (robust gegen Outlier)
    - Optional: Branchen-spezifische Imputation (falls GICS verfügbar)

    Args:
        df: DataFrame mit fehlenden Werten
        method: Imputation-Methode ('median', 'mean')
        threshold: Max. Anteil fehlender Werte für Imputation (0-1)

    Returns:
        DataFrame mit imputierten Werten
    """
    logger.info(f"Starte Imputation (method={method}, threshold={threshold*100:.0f}%)...")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    imputed_cols = []
    skipped_cols = []

    for col in numeric_cols:
        missing_pct = df[col].isna().sum() / len(df)

        if missing_pct == 0:
            continue  # Keine fehlenden Werte

        if missing_pct > threshold:
            skipped_cols.append((col, missing_pct))
            continue  # Zu viele fehlende Werte

        # Imputation
        if method == 'median':
            fill_value = df[col].median()
        elif method == 'mean':
            fill_value = df[col].mean()
        else:
            fill_value = df[col].median()

        df[col] = df[col].fillna(fill_value)
        imputed_cols.append((col, missing_pct, fill_value))

    if imputed_cols:
        logger.info(f"  ✓ {len(imputed_cols)} Spalten imputiert")
        for col, pct, val in imputed_cols[:5]:  # Top 5 loggen
            logger.debug(f"    {col}: {pct*100:.1f}% missing → filled with {val:.2f}")

    if skipped_cols:
        logger.info(f"  ⚠ {len(skipped_cols)} Spalten übersprungen (>threshold)")
        for col, pct in skipped_cols[:3]:
            logger.debug(f"    {col}: {pct*100:.1f}% missing")

    return df


def clean_data(df, impute=True, impute_method='median', impute_threshold=0.5):
    """
    Bereinigt DataFrame: Duplikate, leere Spalten, Imputation.

    Args:
        df: Input DataFrame
        impute: Führe Imputation durch (default: True)
        impute_method: Imputation-Methode ('median', 'mean')
        impute_threshold: Max. Anteil fehlender Werte für Imputation

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

    # 3. Imputation fehlender Werte (NEU)
    if impute:
        df = impute_missing_values(df, method=impute_method, threshold=impute_threshold)

    # 4. Data Quality Report erstellen
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
        'ivst',    # Short-term Investments (optional)

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
        'xint',    # Interest Expense (for Interest Coverage)
        'xrd',     # R&D Expense (for R&D Intensity)

        # Cashflow
        'oancf',   # Operating Activities Net Cash Flow
        'capx',    # Capital Expenditures (for FCF and investment metrics)
        'ivch',    # Investing CF (optional)
        'fincf',   # Financing CF (optional)

        # Sonstiges
        'emp',     # Employees
        'dp',      # Depreciation and Amortization
        'dvt',     # Dividends Total (for payout ratio)
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


def load_market_data(
    market: str,
    data_dir: str = 'data/raw',
    file_selection: list = None,
    filter_country: str = None
) -> pd.DataFrame:
    """
    Lädt Marktdaten mit flexibler Dateiauswahl.

    Unterstützt drei Modi:
    1. Verzeichnis-Modus: Lädt alle CSVs aus data_dir/market/
    2. Selektions-Modus: Lädt nur spezifische Dateien (via file_selection)
    3. Filter-Modus: Lädt und filtert nach country-Spalte

    Args:
        market: Marktname (z.B. 'germany', 'france', 'international')
        data_dir: Basis-Verzeichnis für Daten (default: 'data/raw')
        file_selection: Optional - Liste spezifischer Dateien zum Laden
                       Beispiel: ['dax40_proxy.csv', 'mdax_proxy.csv']
        filter_country: Optional - Filtert nach country-Spalte
                       Nur relevant wenn eine große internationale CSV existiert

    Returns:
        DataFrame mit geladenen Daten

    Examples:
        # Alle Dateien aus germany/ laden
        df = load_market_data('germany')

        # Nur DAX und MDAX laden
        df = load_market_data('germany', file_selection=['dax40_proxy.csv', 'mdax_proxy.csv'])

        # International laden und nach France filtern
        df = load_market_data('international', filter_country='France')
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"LADE MARKTDATEN: {market}")
    logger.info(f"{'='*80}")

    market_dir = Path(data_dir) / market

    # Modus 1: Verzeichnis existiert → Lade CSVs aus Verzeichnis
    if market_dir.is_dir():
        if file_selection:
            # Selektions-Modus: Nur bestimmte Dateien laden
            logger.info(f"📂 Modus: Selektive Dateiauswahl ({len(file_selection)} Dateien)")
            dataframes = []

            for filename in file_selection:
                filepath = market_dir / filename
                if not filepath.exists():
                    logger.warning(f"⚠️ Datei nicht gefunden: {filename} (überspringe)")
                    continue

                logger.info(f"  ✓ Lade: {filename}")
                df = load_data(filepath)
                dataframes.append(df)

            if not dataframes:
                raise FileNotFoundError(f"Keine der angegebenen Dateien gefunden: {file_selection}")

            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"✅ {len(dataframes)} Dateien kombiniert: {len(df)} Zeilen")

        else:
            # Verzeichnis-Modus: Alle CSVs laden
            logger.info(f"📂 Modus: Alle CSVs aus Verzeichnis laden")
            df = load_all_csv_from_directory(str(market_dir))

    # Modus 2: Einzelne Datei (z.B. international.csv)
    else:
        # Suche nach market.csv oder international.csv
        possible_files = [
            Path(data_dir) / f"{market}.csv",
            Path(data_dir) / "international.csv"
        ]

        csv_file = None
        for f in possible_files:
            if f.exists():
                csv_file = f
                break

        if csv_file is None:
            raise FileNotFoundError(
                f"Weder Verzeichnis '{market_dir}' noch Datei '{market}.csv' gefunden"
            )

        logger.info(f"📄 Modus: Einzelne CSV-Datei: {csv_file.name}")
        df = load_data(csv_file)

        # Filter nach country wenn angegeben
        if filter_country:
            if 'country' in df.columns:
                logger.info(f"🔍 Filtere nach country='{filter_country}'")
                df = df[df['country'] == filter_country].copy()
                logger.info(f"  ✓ {len(df)} Zeilen nach Filterung")
            else:
                logger.warning(f"⚠️ Keine 'country' Spalte gefunden - filter_country ignoriert")

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Marktdaten geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
    logger.info(f"{'='*80}\n")

    return df


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


# ============================================================================
# MULTI-MARKET LOADING FUNCTIONS (NEW)
# ============================================================================

def discover_available_countries(csv_path: str, fic_column: str = 'fic') -> pd.DataFrame:
    """
    Analysiert CSV und zeigt verfügbare Länder mit Statistiken.

    Args:
        csv_path: Pfad zur großen internationalen CSV
        fic_column: Name der Spalte mit Ländercodes (default: 'fic')

    Returns:
        DataFrame mit Länderstatistiken (Country Code, Count, % of Total)
    """
    logger.info(f"🔍 Analysiere verfügbare Länder in: {csv_path}")

    # Lade nur fic Spalte für schnelle Analyse
    df = pd.read_csv(csv_path, usecols=[fic_column] if fic_column else None, low_memory=False)

    # Count pro Land
    country_counts = df[fic_column].value_counts().reset_index()
    country_counts.columns = ['country_code', 'n_rows']
    country_counts['percentage'] = (country_counts['n_rows'] / len(df) * 100).round(2)

    # Sortiere nach Anzahl
    country_counts = country_counts.sort_values('n_rows', ascending=False)

    logger.info(f"✅ Gefunden: {len(country_counts)} verschiedene Länder")
    logger.info(f"   Total Zeilen: {len(df):,}")

    return country_counts


def create_index_proxy(
    df: pd.DataFrame,
    country_code: str,
    proxy_config: dict,
    size_column: str = 'mkvalt',
    fallback_column: str = 'at'
) -> pd.DataFrame:
    """
    Erstellt Index-Proxy durch Auswahl der größten Unternehmen.

    Args:
        df: DataFrame mit Unternehmen eines Landes
        country_code: Ländercode (für Logging)
        proxy_config: Config dict mit:
            - type: 'top_n', 'range', 'all', 'percentile'
            - n: Anzahl Unternehmen (für top_n)
            - start, end: Range (für range)
            - percentile: Top X% (für percentile)
        size_column: Spalte für Größenmessung (default: 'mkvalt' = Market Value)
        fallback_column: Fallback wenn size_column fehlt (default: 'at' = Total Assets)

    Returns:
        Gefilterter DataFrame mit Index-Proxy Unternehmen
    """
    if proxy_config.get('type') == 'all':
        logger.info(f"  {country_code}: Alle {len(df)} Unternehmen ausgewählt")
        return df

    # Bestimme Größen-Spalte
    size_col = size_column if size_column in df.columns else fallback_column

    if size_col not in df.columns:
        logger.warning(f"  {country_code}: Keine Größen-Spalte gefunden - nutze alle Unternehmen")
        return df

    # Sortiere nach Größe (absteigend)
    df_sorted = df.sort_values(size_col, ascending=False, na_position='last')

    proxy_type = proxy_config.get('type', 'top_n')

    if proxy_type == 'top_n':
        n = proxy_config.get('n', 50)
        df_selected = df_sorted.head(n)
        logger.info(f"  {country_code}: Top {n} Unternehmen ausgewählt (nach {size_col})")

    elif proxy_type == 'range':
        start = proxy_config.get('start', 1) - 1  # 0-indexed
        end = proxy_config.get('end', 100)
        df_selected = df_sorted.iloc[start:end]
        logger.info(f"  {country_code}: Rang {start+1}-{end} ausgewählt ({len(df_selected)} Unternehmen)")

    elif proxy_type == 'percentile':
        percentile = proxy_config.get('percentile', 90)
        threshold = df_sorted[size_col].quantile(percentile / 100)
        df_selected = df_sorted[df_sorted[size_col] >= threshold]
        logger.info(f"  {country_code}: Top {100-percentile}% ausgewählt ({len(df_selected)} Unternehmen)")

    else:
        logger.warning(f"  {country_code}: Unbekannter proxy_type '{proxy_type}' - nutze top_n")
        df_selected = df_sorted.head(50)

    return df_selected


def load_multi_market_data(
    csv_path: str,
    market_config: dict,
    fic_column: str = 'fic',
    data_dir: str = 'data/raw'
) -> pd.DataFrame:
    """
    Lädt Daten aus großer internationaler CSV mit flexibler Länderauswahl.

    Args:
        csv_path: Pfad zur CSV (absolute oder relativ zu data_dir)
        market_config: Konfiguration mit:
            {
                'mode': 'single' oder 'multi',
                'countries': ['DEU', 'FRA'] oder 'ALL',
                'use_index_proxy': True/False,
                'index_proxies': {
                    'DEU': {'type': 'top_n', 'n': 160},
                    'FRA': {'type': 'top_n', 'n': 40}
                },
                'exclude_sectors': [40, 60],  # Optional
                'min_company_size': None  # Optional
            }
        fic_column: Spaltenname für Ländercode
        data_dir: Basis-Verzeichnis (falls csv_path relativ)

    Returns:
        DataFrame mit geladenen und gefilterten Daten + 'source_country' Spalte
    """
    logger.info(f"\n{'='*80}")
    logger.info("MULTI-MARKET DATA LOADING")
    logger.info(f"{'='*80}")

    # Pfad vorbereiten
    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = Path(data_dir) / csv_path

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_file}")

    logger.info(f"📂 Lade Daten aus: {csv_file.name}")

    # Gesamte CSV laden
    df_full = load_data(csv_file)
    logger.info(f"   Total Zeilen: {len(df_full):,}")

    # Prüfe ob fic Spalte existiert
    if fic_column not in df_full.columns:
        raise ValueError(f"Spalte '{fic_column}' nicht gefunden in CSV. Verfügbare: {df_full.columns.tolist()[:10]}")

    # Länderauswahl
    countries = market_config.get('countries', 'ALL')

    if countries == 'ALL':
        logger.info("🌍 Modus: Alle Länder")
        df_selected = df_full.copy()
    else:
        logger.info(f"🌍 Modus: Ausgewählte Länder - {countries}")
        df_selected = df_full[df_full[fic_column].isin(countries)].copy()
        logger.info(f"   Nach Länderfilter: {len(df_selected):,} Zeilen")

        if len(df_selected) == 0:
            raise ValueError(f"Keine Daten gefunden für Länder: {countries}")

    # Index-Proxy Filterung
    use_proxy = market_config.get('use_index_proxy', False)

    if use_proxy and countries != 'ALL':
        logger.info("\n📊 Erstelle Index-Proxies pro Land...")

        index_proxies = market_config.get('index_proxies', {})
        dataframes_per_country = []

        for country in countries:
            df_country = df_selected[df_selected[fic_column] == country].copy()

            if len(df_country) == 0:
                logger.warning(f"  ⚠️  {country}: Keine Daten gefunden")
                continue

            # Proxy Config für dieses Land
            if isinstance(index_proxies, dict):
                proxy_config = index_proxies.get(country, {'type': 'all'})
            else:
                proxy_config = {'type': 'all'}

            # Erstelle Proxy
            df_proxy = create_index_proxy(df_country, country, proxy_config)
            df_proxy['source_country'] = country  # Markiere Herkunft
            dataframes_per_country.append(df_proxy)

        # Kombiniere alle Länder
        df_selected = pd.concat(dataframes_per_country, ignore_index=True)
        logger.info(f"\n   ✅ Index-Proxy erstellt: {len(df_selected)} Unternehmen gesamt")
    else:
        # Keine Proxy-Filterung, nur source_country hinzufügen
        df_selected['source_country'] = df_selected[fic_column]

    # Sektor-Ausschluss
    exclude_sectors = market_config.get('exclude_sectors', [])
    if exclude_sectors and 'gsector' in df_selected.columns:
        logger.info(f"\n🚫 Schließe Sektoren aus: {exclude_sectors}")
        initial_len = len(df_selected)
        df_selected = df_selected[~df_selected['gsector'].isin(exclude_sectors)]
        logger.info(f"   {initial_len - len(df_selected)} Zeilen entfernt → {len(df_selected)} verbleiben")

    # Größenfilter
    min_size = market_config.get('min_company_size')
    if min_size:
        size_col = 'mkvalt' if 'mkvalt' in df_selected.columns else 'at'
        if size_col in df_selected.columns:
            logger.info(f"\n📏 Filtere nach Mindestgröße: {size_col} >= {min_size}")
            initial_len = len(df_selected)
            df_selected = df_selected[df_selected[size_col] >= min_size]
            logger.info(f"   {initial_len - len(df_selected)} Zeilen entfernt → {len(df_selected)} verbleiben")

    # Zusammenfassung
    logger.info(f"\n{'='*80}")
    logger.info("✅ MULTI-MARKET LOADING ABGESCHLOSSEN")
    logger.info(f"{'='*80}")
    logger.info(f"   Final Zeilen: {len(df_selected):,}")
    logger.info(f"   Final Spalten: {len(df_selected.columns)}")

    if 'source_country' in df_selected.columns:
        logger.info("\n   Verteilung pro Land:")
        for country, count in df_selected['source_country'].value_counts().items():
            logger.info(f"     {country}: {count:,} Zeilen")

    logger.info(f"{'='*80}\n")

    return df_selected


# Preset Konfigurationen
MARKET_PRESETS = {
    'germany_dax_family': {
        'mode': 'single',
        'countries': ['DEU'],
        'use_index_proxy': True,
        'index_proxies': {
            'DEU': {'type': 'top_n', 'n': 160}  # DAX (40) + MDAX (60) + SDAX (60)
        },
        'exclude_sectors': [40],  # Financials
        'description': 'Deutsche DAX-Familie (Top 160 Unternehmen)'
    },

    'europe_large_cap': {
        'mode': 'multi',
        'countries': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP'],
        'use_index_proxy': True,
        'index_proxies': {
            'DEU': {'type': 'top_n', 'n': 50},
            'FRA': {'type': 'top_n', 'n': 50},
            'GBR': {'type': 'top_n', 'n': 50},
            'ITA': {'type': 'top_n', 'n': 50},
            'ESP': {'type': 'top_n', 'n': 50}
        },
        'exclude_sectors': [40, 60],  # Financials & Real Estate
        'description': 'Top 50 Unternehmen aus 5 großen EU-Ländern'
    },

    'germany_vs_france': {
        'mode': 'multi',
        'countries': ['DEU', 'FRA'],
        'use_index_proxy': True,
        'index_proxies': {
            'DEU': {'type': 'top_n', 'n': 40},
            'FRA': {'type': 'top_n', 'n': 40}
        },
        'exclude_sectors': [],
        'description': 'Vergleich Deutschland vs Frankreich (je Top 40)'
    },

    'all_europe': {
        'mode': 'multi',
        'countries': ['DEU', 'GBR', 'FRA', 'ITA', 'ESP', 'NLD', 'CHE', 'BEL', 'AUT', 'SWE'],
        'use_index_proxy': False,  # Alle Unternehmen
        'exclude_sectors': [40],
        'description': 'Alle Unternehmen aus 10 europäischen Ländern'
    },

    'usa_large_cap': {
        'mode': 'single',
        'countries': ['USA'],
        'use_index_proxy': True,
        'index_proxies': {
            'USA': {'type': 'top_n', 'n': 500}  # S&P 500 Proxy
        },
        'exclude_sectors': [40, 60],
        'description': 'US Large Caps (Top 500 Unternehmen)'
    },

    'global_giants': {
        'mode': 'multi',
        'countries': ['USA', 'DEU', 'GBR', 'FRA', 'JPN', 'CHN'],
        'use_index_proxy': True,
        'index_proxies': {
            'USA': {'type': 'top_n', 'n': 100},
            'DEU': {'type': 'top_n', 'n': 30},
            'GBR': {'type': 'top_n', 'n': 30},
            'FRA': {'type': 'top_n', 'n': 30},
            'JPN': {'type': 'top_n', 'n': 30},
            'CHN': {'type': 'top_n', 'n': 30}
        },
        'exclude_sectors': [40],
        'description': 'Größte Unternehmen aus 6 wichtigsten Wirtschaftsräumen'
    }
}


def get_preset_config(preset_name: str) -> dict:
    """
    Gibt Preset-Konfiguration zurück.

    Args:
        preset_name: Name des Presets

    Returns:
        Konfiguration als dict

    Raises:
        ValueError: Wenn Preset nicht existiert
    """
    if preset_name not in MARKET_PRESETS:
        available = list(MARKET_PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' nicht gefunden. Verfügbar: {available}")

    return MARKET_PRESETS[preset_name].copy()


def list_available_presets() -> pd.DataFrame:
    """
    Zeigt alle verfügbaren Presets mit Beschreibungen.

    Returns:
        DataFrame mit Preset-Übersicht
    """
    presets_info = []

    for name, config in MARKET_PRESETS.items():
        n_countries = len(config['countries']) if config['countries'] != 'ALL' else 'ALL'

        presets_info.append({
            'preset_name': name,
            'description': config.get('description', ''),
            'countries': n_countries,
            'mode': config['mode'],
            'use_proxy': config.get('use_index_proxy', False)
        })

    return pd.DataFrame(presets_info)


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