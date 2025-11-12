"""
Validation Runner Module
Orchestrates validation steps for clustering results
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


def perform_validation(
    df: pd.DataFrame,
    features: List[str],
    analysis_type: str,
    config: dict,
    validation_enabled: bool = True,
    algorithm_comparison: Optional[object] = None,
    external_validation: Optional[object] = None
) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Perform validation on clustering results

    Runs:
    1. Algorithm comparison (alternative clustering methods)
    2. External validation (GICS sectors, size categories)

    Args:
        df: DataFrame with cluster assignments
        features: List of features used for clustering
        analysis_type: Type of analysis ('static', 'dynamic', 'combined')
        config: Configuration dictionary
        validation_enabled: Whether validation is enabled
        algorithm_comparison: AlgorithmComparison instance
        external_validation: ExternalValidation instance

    Returns:
        Tuple of (validation_results, validation_df)
    """
    if not validation_enabled:
        logger.info(f"  Validation disabled, skipping...")
        return {}, None

    logger.info(f"\n  🔍 Running Validation ({analysis_type})...")

    validation_results = {}
    df_validation = None

    try:
        # Import validation modules if not provided
        if algorithm_comparison is None:
            from src._06_validation.algorithm_comparison import AlgorithmComparison
            algorithm_comparison = AlgorithmComparison()

        if external_validation is None:
            from src._06_validation.external_validation import ExternalValidation
            external_validation = ExternalValidation()

        # 1. Algorithm Comparison (K-Means vs Hierarchical vs DBSCAN)
        logger.info(f"     → Comparing alternative algorithms...")

        algo_results, df_algo = algorithm_comparison.compare_multiple_algorithms(
            df=df,
            features=features,
            original_labels=df['cluster'].values,
            n_clusters=len(df['cluster'].unique())
        )

        validation_results['algorithm_comparison'] = algo_results
        df_validation = df_algo

        # Log ARI scores
        if 'ari_scores' in algo_results:
            ari = algo_results['ari_scores']
            logger.info(f"       ✓ ARI scores:")
            for alg, score in ari.items():
                logger.info(f"         {alg}: {score:.3f}")

        # 2. External Validation (if external labels available)
        external_label_cols = config.get('validation', {}).get('external_labels', [])
        available_labels = [col for col in external_label_cols if col in df.columns]

        if available_labels:
            logger.info(f"     → Validating against external labels: {available_labels}")

            ext_results = external_validation.validate_against_external(
                df=df,
                cluster_column='cluster',
                external_label_columns=available_labels
            )

            validation_results['external_validation'] = ext_results

            # Log Cramér's V scores
            if 'cramers_v' in ext_results:
                cv = ext_results['cramers_v']
                logger.info(f"       ✓ Cramér's V scores:")
                for label, score in cv.items():
                    logger.info(f"         {label}: {score:.3f}")
        else:
            logger.info(f"     ⚠️  No external labels available for validation")

        logger.info(f"     ✓ Validation complete")

        return validation_results, df_validation

    except Exception as e:
        logger.error(f"     ❌ Validation failed: {e}", exc_info=True)
        return validation_results, df_validation


def add_external_labels(
    df: pd.DataFrame,
    label_columns: List[str]
) -> pd.DataFrame:
    """
    Add external labels to DataFrame (GICS sectors, size categories, etc.)

    Args:
        df: DataFrame
        label_columns: List of label column names to add

    Returns:
        DataFrame with external labels added
    """
    logger.info(f"\n  🏷️  Adding External Labels...")

    try:
        # GICS Sector mapping
        if 'gsector' in df.columns and 'gics_sector' in label_columns:
            gics_mapping = {
                10: 'Energy',
                15: 'Materials',
                20: 'Industrials',
                25: 'Consumer Discretionary',
                30: 'Consumer Staples',
                35: 'Health Care',
                40: 'Financials',
                45: 'Information Technology',
                50: 'Communication Services',
                55: 'Utilities',
                60: 'Real Estate'
            }
            df['gics_sector'] = df['gsector'].map(gics_mapping).fillna('Unknown')
            logger.info(f"     ✓ GICS sectors mapped: {df['gics_sector'].nunique()} unique sectors")

        # Size categories based on revenue or assets
        if ('revt' in df.columns or 'at' in df.columns) and 'size_category' in label_columns:
            # Use revenue if available, otherwise assets
            size_metric = 'revt' if 'revt' in df.columns else 'at'

            # Create size categories (quartiles)
            df['size_category'] = pd.qcut(
                df[size_metric],
                q=4,
                labels=['Small', 'Medium', 'Large', 'Very Large'],
                duplicates='drop'
            )
            logger.info(f"     ✓ Size categories created based on {size_metric}")

        # Add more label types as needed
        logger.info(f"     ✓ External labels added")

        return df

    except Exception as e:
        logger.error(f"     ❌ Adding external labels failed: {e}", exc_info=True)
        return df


def run_pca_validation(
    df: pd.DataFrame,
    features: List[str],
    cluster_column: str = 'cluster',
    n_components: float = 0.85
) -> Dict:
    """
    Validate clustering in PCA space

    Compares clustering quality in original space vs PCA space

    Args:
        df: DataFrame with features and cluster assignments
        features: List of feature names
        cluster_column: Name of cluster column
        n_components: Number of PCA components (or variance to preserve)

    Returns:
        Dictionary with PCA validation results
    """
    logger.info(f"\n  🔬 PCA Space Validation...")

    try:
        from src._02_preprocessing.pca_transformer import PCATransformer
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        # Prepare data
        X = df[features].copy()
        X = X.fillna(X.median())
        labels = df[cluster_column].values

        # Original space metrics
        sil_original = silhouette_score(X, labels)
        dbi_original = davies_bouldin_score(X, labels)

        logger.info(f"     Original space - Silhouette: {sil_original:.3f}, DBI: {dbi_original:.3f}")

        # PCA transformation
        pca_transformer = PCATransformer(n_components=n_components)
        X_pca, _ = pca_transformer.fit_transform(df, features)

        # PCA space metrics
        sil_pca = silhouette_score(X_pca, labels)
        dbi_pca = davies_bouldin_score(X_pca, labels)

        logger.info(f"     PCA space    - Silhouette: {sil_pca:.3f}, DBI: {dbi_pca:.3f}")
        logger.info(f"     PCA components: {pca_transformer.pca.n_components_}")

        results = {
            'original_space': {
                'silhouette': sil_original,
                'davies_bouldin': dbi_original
            },
            'pca_space': {
                'silhouette': sil_pca,
                'davies_bouldin': dbi_pca,
                'n_components': pca_transformer.pca.n_components_,
                'variance_explained': pca_transformer.pca.explained_variance_ratio_.sum()
            }
        }

        logger.info(f"     ✓ PCA validation complete")

        return results

    except Exception as e:
        logger.error(f"     ❌ PCA validation failed: {e}", exc_info=True)
        return {}
