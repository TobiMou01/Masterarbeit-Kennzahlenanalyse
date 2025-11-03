"""
PCA Transformer - Principal Component Analysis for Dimensionality Reduction

Provides PCA transformation for extended feature sets (25+ features):
- Dimensionality reduction while preserving variance
- Component interpretation (feature loadings)
- Inverse transformation for profile interpretation
- Validation and diagnostic tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class PCATransformer:
    """
    Performs PCA transformation for clustering with extended feature sets

    Key Features:
    - Automatic standardization before PCA
    - Configurable variance threshold (e.g., 85%)
    - Component interpretation via feature loadings
    - Inverse transformation for interpretability
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.85,
        random_state: int = 42
    ):
        """
        Initialize PCA Transformer

        Args:
            n_components: Number of components or variance threshold
                - int (e.g., 6): Exact number of components
                - float (e.g., 0.85): Preserve this much variance (85%)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state

        # Will be set during fit
        self.pca = None
        self.scaler = None
        self.feature_names = None

        logger.info(f"✓ PCATransformer initialized (n_components={n_components})")

    # =========================================================================
    # TRANSFORMATION METHODS
    # =========================================================================

    def fit_transform(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fits PCA and transforms data

        Workflow:
        1. Validate features
        2. Standardize data (StandardScaler)
        3. Fit PCA
        4. Transform data

        Args:
            df: DataFrame with features
            features: List of feature column names

        Returns:
            Tuple of:
            - np.ndarray: Transformed data (n_samples, n_components)
            - pd.DataFrame: Metadata (variance explained, etc.)
        """
        logger.info(f"\n{'='*80}")
        logger.info("PCA TRANSFORMATION")
        logger.info(f"{'='*80}")
        logger.info(f"  Original Features: {len(features)}")
        logger.info(f"  Samples: {len(df)}")

        # Validate features
        available_features = [f for f in features if f in df.columns]
        missing_features = set(features) - set(available_features)

        if missing_features:
            logger.warning(f"  ⚠️  Missing features: {missing_features}")

        if len(available_features) < 3:
            raise ValueError(f"PCA requires at least 3 features, only {len(available_features)} available")

        self.feature_names = available_features

        # Extract and clean data
        X = df[available_features].copy()

        # Handle missing values (fill with median)
        for col in available_features:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.debug(f"  Filled NaN in {col} with median={median_val:.2f}")

        X = X.values

        # 1. Standardize (critical for PCA!)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"  ✓ Data standardized (mean=0, std=1)")

        # 2. Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)

        logger.info(f"  ✓ PCA fitted")
        logger.info(f"    Components: {self.pca.n_components_}")
        logger.info(f"    Explained Variance: {self.pca.explained_variance_ratio_.sum():.1%}")
        logger.info(f"{'='*80}\n")

        # 3. Create metadata
        metadata = self.get_explained_variance_summary()

        return X_pca, metadata

    def transform(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Transforms new data using fitted PCA

        Args:
            df: DataFrame with features
            features: List of feature column names (must match fit)

        Returns:
            np.ndarray: Transformed data
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA not fitted yet. Call fit_transform() first.")

        # Extract data
        X = df[features].copy()

        # Handle missing values
        for col in features:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

        X = X.values

        # Standardize and transform
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return X_pca

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Transforms PCA space back to original feature space

        Useful for interpreting PCA cluster profiles.

        Args:
            X_pca: Data in PCA space (n_samples, n_components)

        Returns:
            np.ndarray: Approximation in original feature space (n_samples, n_features)
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA not fitted yet. Call fit_transform() first.")

        # PCA inverse transform
        X_scaled = self.pca.inverse_transform(X_pca)

        # Scaler inverse transform
        X_original = self.scaler.inverse_transform(X_scaled)

        return X_original

    # =========================================================================
    # COMPONENT INTERPRETATION
    # =========================================================================

    def get_component_loadings(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Gets feature loadings for each principal component

        Loadings indicate how much each original feature contributes to each PC.

        Args:
            feature_names: Optional list of feature names (uses self.feature_names if None)

        Returns:
            DataFrame with:
                - Rows: Features
                - Columns: PC1, PC2, ..., PCn
                - Values: Loading values (-1 to +1)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet. Call fit_transform() first.")

        if feature_names is None:
            feature_names = self.feature_names

        if len(feature_names) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} feature names, got {len(feature_names)}")

        # Get components (rows=components, cols=features)
        components = self.pca.components_

        # Create DataFrame (transpose so rows=features, cols=components)
        loadings_df = pd.DataFrame(
            components.T,
            index=feature_names,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )

        return loadings_df

    def interpret_components(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Interprets components by identifying top contributing features

        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to return per component

        Returns:
            Dict mapping PC name to list of (feature, loading) tuples
            Example:
                {
                    'PC1': [('roa', 0.92), ('roe', 0.88), ...],
                    'PC2': [('debt_to_equity', 0.87), ...]
                }
        """
        loadings_df = self.get_component_loadings(feature_names)

        interpretation = {}

        for pc in loadings_df.columns:
            # Get loadings for this PC
            pc_loadings = loadings_df[pc]

            # Sort by absolute value (strongest contributions)
            top_features = pc_loadings.abs().nlargest(top_n)

            # Get actual values (with sign)
            top_features_with_values = [
                (feature, pc_loadings[feature])
                for feature in top_features.index
            ]

            interpretation[pc] = top_features_with_values

        return interpretation

    def get_explained_variance_summary(self) -> pd.DataFrame:
        """
        Gets summary of explained variance per component

        Returns:
            DataFrame with:
                - Component (PC1, PC2, ...)
                - Explained Variance Ratio (%)
                - Cumulative Variance (%)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet. Call fit_transform() first.")

        summary_data = []

        cumulative_variance = 0

        for i in range(self.pca.n_components_):
            variance_ratio = self.pca.explained_variance_ratio_[i]
            cumulative_variance += variance_ratio

            summary_data.append({
                'Component': f'PC{i+1}',
                'Explained_Variance_Ratio': variance_ratio * 100,  # Convert to %
                'Cumulative_Variance': cumulative_variance * 100
            })

        summary_df = pd.DataFrame(summary_data)

        return summary_df

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_transformation(
        self,
        df: pd.DataFrame,
        features: List[str],
        min_variance: float = 0.85
    ) -> Dict:
        """
        Validates PCA transformation

        Checks:
        - Is min_variance threshold met?
        - How many components were needed?
        - Are features properly scaled?
        - Is there multicollinearity (good for PCA)?

        Args:
            df: DataFrame with features
            features: List of feature names
            min_variance: Minimum variance threshold to validate

        Returns:
            Dict with validation results
        """
        logger.info(f"\nValidating PCA Transformation...")

        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        # Check 1: PCA fitted?
        if self.pca is None:
            validation['errors'].append("PCA not fitted yet")
            validation['valid'] = False
            return validation

        # Check 2: Variance threshold met?
        total_variance = self.pca.explained_variance_ratio_.sum()

        if total_variance < min_variance:
            validation['warnings'].append(
                f"Total variance ({total_variance:.1%}) < min_variance ({min_variance:.1%})"
            )
            validation['valid'] = False
        else:
            logger.info(f"  ✓ Variance threshold met: {total_variance:.1%} >= {min_variance:.1%}")

        # Check 3: Number of components reasonable?
        n_features = len(features)
        n_components = self.pca.n_components_

        reduction_ratio = n_components / n_features

        if reduction_ratio > 0.9:
            validation['warnings'].append(
                f"Low dimensionality reduction: {n_components}/{n_features} components ({reduction_ratio:.1%})"
            )
        else:
            logger.info(f"  ✓ Dimensionality reduction: {n_features} → {n_components} features ({reduction_ratio:.1%})")

        # Check 4: Features available?
        available_features = [f for f in features if f in df.columns]

        if len(available_features) < len(features):
            validation['warnings'].append(
                f"Missing features: {set(features) - set(available_features)}"
            )

        # Check 5: Multicollinearity (correlation matrix)
        # High correlation = good for PCA
        X = df[available_features].fillna(df[available_features].median())
        corr_matrix = X.corr()

        # Count high correlations (|r| > 0.7)
        high_corr_count = ((corr_matrix.abs() > 0.7) & (corr_matrix.abs() < 1.0)).sum().sum() // 2

        if high_corr_count > 0:
            logger.info(f"  ✓ Multicollinearity detected: {high_corr_count} feature pairs with |r| > 0.7 (good for PCA)")
        else:
            validation['warnings'].append(
                "Low multicollinearity - PCA might not provide much benefit"
            )

        # Summary
        validation['n_components'] = n_components
        validation['total_variance'] = total_variance
        validation['reduction_ratio'] = reduction_ratio
        validation['high_correlations'] = high_corr_count

        logger.info(f"  Validation: {'✓ PASSED' if validation['valid'] else '⚠️  WARNINGS'}")

        return validation

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_pca_profiles(self, df_pca: pd.DataFrame, cluster_column: str = 'cluster') -> pd.DataFrame:
        """
        Calculates cluster profiles in PCA space

        Args:
            df_pca: DataFrame with PCA coordinates and cluster assignments
            cluster_column: Name of cluster column

        Returns:
            DataFrame with average PCA coordinates per cluster
        """
        # Identify PCA coordinate columns (PC1, PC2, ...)
        pca_cols = [col for col in df_pca.columns if col.startswith('PC')]

        if len(pca_cols) == 0:
            raise ValueError("No PCA coordinate columns found in DataFrame")

        # Calculate profiles
        profiles = df_pca.groupby(cluster_column)[pca_cols].mean()

        return profiles

    def map_pca_profiles_to_original(self, profiles_pca: pd.DataFrame) -> pd.DataFrame:
        """
        Maps PCA cluster profiles back to original feature space

        Useful for interpreting what a cluster "means" in original features.

        Args:
            profiles_pca: Cluster profiles in PCA space (index=cluster_id, cols=PC1,PC2,...)

        Returns:
            DataFrame with cluster profiles in original feature space
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA not fitted yet. Call fit_transform() first.")

        # Get PCA coordinates (ensure correct order)
        pca_cols = [col for col in profiles_pca.columns if col.startswith('PC')]
        X_pca = profiles_pca[pca_cols].values

        # Inverse transform
        X_original = self.inverse_transform(X_pca)

        # Create DataFrame
        profiles_original = pd.DataFrame(
            X_original,
            index=profiles_pca.index,
            columns=self.feature_names
        )

        return profiles_original

    def __repr__(self):
        if self.pca is None:
            return "PCATransformer(not fitted)"
        else:
            variance = self.pca.explained_variance_ratio_.sum()
            return f"PCATransformer(n_components={self.pca.n_components_}, variance={variance:.1%})"


if __name__ == "__main__":
    # Test PCATransformer
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("PCA TRANSFORMER TEST")
    print("="*80)

    # Mock data (25 correlated features)
    np.random.seed(42)
    n_samples = 100
    n_features = 25

    # Create correlated features
    # First 5 are highly correlated (profitability)
    base = np.random.randn(n_samples, 1)
    feat_1_5 = base + np.random.randn(n_samples, 5) * 0.3

    # Next 5 are correlated (leverage)
    base2 = np.random.randn(n_samples, 1)
    feat_6_10 = base2 + np.random.randn(n_samples, 5) * 0.3

    # Rest are more independent
    feat_11_25 = np.random.randn(n_samples, 15)

    X = np.hstack([feat_1_5, feat_6_10, feat_11_25])

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    # Test 1: Fit Transform
    print("\n1. FIT TRANSFORM:")
    transformer = PCATransformer(n_components=0.85, random_state=42)
    X_pca, metadata = transformer.fit_transform(df, feature_names)

    print(f"\nOriginal: {n_features} features")
    print(f"PCA: {X_pca.shape[1]} components")
    print(f"Explained Variance: {metadata['Cumulative_Variance'].iloc[-1]:.2f}%")

    # Test 2: Component Interpretation
    print("\n2. COMPONENT INTERPRETATION:")
    interpretation = transformer.interpret_components(feature_names, top_n=3)
    for pc, features in list(interpretation.items())[:3]:
        print(f"\n{pc}:")
        for feat, loading in features:
            print(f"  {feat}: {loading:+.3f}")

    # Test 3: Inverse Transform
    print("\n3. INVERSE TRANSFORM:")
    X_reconstructed = transformer.inverse_transform(X_pca)
    print(f"Reconstructed shape: {X_reconstructed.shape}")

    # Reconstruction error
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"MSE (reconstruction error): {reconstruction_error:.4f}")

    # Test 4: Validation
    print("\n4. VALIDATION:")
    validation = transformer.validate_transformation(df, feature_names, min_variance=0.85)
    print(f"Valid: {validation['valid']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Components: {validation['n_components']}")
    print(f"Total Variance: {validation['total_variance']:.1%}")

    print("\n✓ PCATransformer Test erfolgreich!")
    print("="*80 + "\n")
