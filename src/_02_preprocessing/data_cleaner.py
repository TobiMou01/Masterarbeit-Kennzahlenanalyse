"""
Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import logging
from pathlib import Path
from src._02_preprocessing import data_loader
from src._02_preprocessing.calculators import feature_coordinator

logger = logging.getLogger(__name__)


def run_preprocessing(
    input_dir: str,
    market: str,
    impute=True,
    impute_method='median',
    impute_threshold=0.5,
    smooth_static=False,
    cagr_years=3
) -> pd.DataFrame:
    """
    Complete preprocessing: Load → Clean → Impute → Engineer Features → (Optional: CAGR Smoothing)

    Args:
        input_dir: Input directory (e.g., 'data/raw')
        market: Market name (e.g., 'germany')
        impute: Führe Imputation durch (default: True)
        impute_method: Imputation-Methode ('median', 'mean')
        impute_threshold: Max. Anteil fehlender Werte für Imputation (0-1)
        smooth_static: Führe CAGR-Glättung für statische Daten durch (default: False)
        cagr_years: Anzahl Jahre für CAGR-Glättung (default: 3)

    Returns:
        DataFrame with all features
    """
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING")
    logger.info("=" * 80 + "\n")

    # 1. Load data
    input_path = Path(input_dir) / market
    if input_path.is_dir():
        df = data_loader.load_all_csv_from_directory(str(input_path))
    else:
        df = data_loader.load_data(str(input_path))

    logger.info(f"  Loaded: {len(df)} rows")

    # 2. Clean data (inkl. Imputation)
    df_cleaned, _ = data_loader.clean_data(
        df,
        impute=impute,
        impute_method=impute_method,
        impute_threshold=impute_threshold
    )
    df_final = data_loader.filter_relevant_columns(df_cleaned)

    # 3. Feature engineering (inkl. optionale CAGR-Glättung)
    df_features = feature_coordinator.create_all_features(
        df_final,
        smooth_static=smooth_static,
        cagr_years=cagr_years
    )

    # 4. Save processed data
    output_dir = Path(f'data/processed/{market}')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_dir / 'features.csv', index=False)

    logger.info(f"  ✓ Saved: {len(df_features)} rows, {len(df_features.columns)} columns\n")

    return df_features


def prepare_time_data(df: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare time-series data structure

    Args:
        df: DataFrame with features
        market: Market name

    Returns:
        (df_all, df_latest)
    """
    logger.info("Preparing time structure...")

    # Add time metadata
    df['years_available'] = df.groupby('gvkey')['fyear'].transform('count')
    df['latest_year'] = df.groupby('gvkey')['fyear'].transform('max') == df['fyear']

    # Filter latest year
    df_latest = df[df['latest_year'] == True].copy()
    logger.info(f"  Latest year: {len(df_latest)} companies")

    # Save
    output_dir = Path(f'data/processed/{market}')
    df_latest.to_csv(output_dir / 'features_latest.csv', index=False)

    return df, df_latest


def load_processed_data(market: str) -> pd.DataFrame:
    """
    Load previously processed features

    Args:
        market: Market name

    Returns:
        DataFrame with features
    """
    path = Path(f'data/processed/{market}/features.csv')

    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            f"Run without --skip-prep first"
        )

    return pd.read_csv(path)
