"""
Score Calculator - Comprehensive Scoring Mechanisms

Provides 4 types of scores:
1. Proximity Score - Distance to cluster center (0-100, higher = closer)
2. Dimensional Scores - Per category (Profitability, Leverage, Efficiency, Growth)
3. Relative Score - Z-Score based performance vs. cluster average
4. Overall Score - Weighted combination of all scores
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import logging

from src._01_setup.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class ScoreCalculator:
    """
    Calculates various scores for clustering quality and company performance

    Scores are always in range 0-100:
    - 100: Perfect/Best
    - 50: Average
    - 0: Worst
    """

    def __init__(self, feature_selector: Optional[FeatureSelector] = None):
        """
        Initialize Score Calculator

        Args:
            feature_selector: FeatureSelector instance (creates new one if None)
        """
        self.feature_selector = feature_selector or FeatureSelector()
        logger.info("✓ ScoreCalculator initialized")

    # =========================================================================
    # PROXIMITY SCORE - Distance to Cluster Center
    # =========================================================================

    def calculate_proximity_score(
        self,
        df: pd.DataFrame,
        features: List[str],
        cluster_column: str = 'cluster',
        profiles: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculates proximity score based on distance to cluster center

        Proximity Score measures how close a company is to its cluster center.
        Higher scores indicate companies are more typical/representative of their cluster.

        Formula:
            distance = euclidean_distance(company, cluster_center)
            score = 100 * exp(-distance / 2)

        Args:
            df: DataFrame with cluster assignments and features
            features: List of features used for distance calculation
            cluster_column: Name of cluster column (default: 'cluster')
            profiles: Cluster profiles (optional, will be computed if None)

        Returns:
            pd.Series with proximity scores (0-100)
        """
        logger.info(f"\nCalculating Proximity Scores...")
        logger.info(f"  Features: {len(features)}")
        logger.info(f"  Companies: {len(df)}")

        # Validate features exist in df
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"  ⚠️  Missing features in df: {missing_features}")
            features = [f for f in features if f in df.columns]

        # If profiles provided, also check they exist in profiles
        if profiles is not None:
            missing_in_profiles = [f for f in features if f not in profiles.columns]
            if missing_in_profiles:
                logger.warning(f"  ⚠️  Missing features in profiles: {missing_in_profiles}")
                features = [f for f in features if f in profiles.columns]

        if len(features) == 0:
            logger.error("  ❌ No valid features found!")
            return pd.Series(0, index=df.index)

        # Prepare data
        df_clean = df.copy()

        # Handle missing values and infinities
        for feature in features:
            # Replace inf with NaN first
            df_clean[feature] = df_clean[feature].replace([np.inf, -np.inf], np.nan)

            # Fill NaN with median
            if df_clean[feature].isna().any():
                median_val = df_clean[feature].median()
                if np.isnan(median_val):  # If median is also NaN, use 0
                    median_val = 0
                df_clean[feature] = df_clean[feature].fillna(median_val)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[features])

        # Final check for NaN/inf in scaled data
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute cluster centers if not provided
        if profiles is None:
            cluster_centers = {}
            for cluster_id in df_clean[cluster_column].unique():
                if cluster_id >= 0:  # Exclude noise points (-1)
                    cluster_mask = df_clean[cluster_column] == cluster_id
                    cluster_centers[cluster_id] = X_scaled[cluster_mask].mean(axis=0)
        else:
            # Use provided profiles
            cluster_centers = {}
            for cluster_id in profiles.index:
                if cluster_id >= 0:
                    # Standardize profile using same scaler
                    profile_features = profiles.loc[cluster_id, features].values.reshape(1, -1)
                    cluster_centers[cluster_id] = scaler.transform(profile_features)[0]

        # Calculate distances for each company
        distances = []
        for idx, row in df_clean.iterrows():
            cluster_id = row[cluster_column]

            if cluster_id < 0:  # Noise point
                distances.append(np.inf)
                continue

            if cluster_id not in cluster_centers:
                distances.append(np.inf)
                continue

            # Get company features
            company_vector = X_scaled[df_clean.index.get_loc(idx)].reshape(1, -1)

            # Get cluster center
            center_vector = cluster_centers[cluster_id].reshape(1, -1)

            # Calculate euclidean distance
            distance = euclidean_distances(company_vector, center_vector)[0, 0]
            distances.append(distance)

        distances = np.array(distances)

        # Transform distance to score: score = 100 * exp(-distance / 2)
        # This gives:
        # - distance = 0 → score = 100
        # - distance = 1 → score = 60.7
        # - distance = 2 → score = 36.8
        # - distance = 3 → score = 22.3
        scores = 100 * np.exp(-distances / 2)

        # Handle infinite distances (noise points)
        scores = np.where(np.isinf(distances), 0, scores)

        # Ensure scores are in [0, 100]
        scores = np.clip(scores, 0, 100)

        proximity_series = pd.Series(scores, index=df.index, name='proximity_score')

        logger.info(f"  ✓ Proximity Scores: Mean={proximity_series.mean():.1f}, "
                   f"Std={proximity_series.std():.1f}, "
                   f"Min={proximity_series.min():.1f}, "
                   f"Max={proximity_series.max():.1f}")

        return proximity_series

    # =========================================================================
    # DIMENSIONAL SCORES - Per Category
    # =========================================================================

    def calculate_dimensional_scores(
        self,
        df: pd.DataFrame,
        cluster_column: str = 'cluster',
        profiles: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculates separate proximity scores for each feature category

        Categories:
        - Profitability (roa, roe, ebit_margin, ...)
        - Leverage (debt_to_equity, interest_coverage, ...)
        - Efficiency (asset_turnover, inventory_turnover, ...)
        - Growth (revenue_growth, earnings_growth, ...)

        Args:
            df: DataFrame with cluster assignments and features
            cluster_column: Name of cluster column
            profiles: Cluster profiles (optional)

        Returns:
            pd.DataFrame with dimensional scores (4 columns)
        """
        logger.info(f"\nCalculating Dimensional Scores...")

        # Get features by category
        categories = self.feature_selector.get_features_by_category()

        dimensional_scores = pd.DataFrame(index=df.index)

        # Calculate score for each category
        for category_name, feature_list in categories.items():
            # Only use features that exist in df
            available_features = [f for f in feature_list if f in df.columns]

            # If profiles provided, also check they exist in profiles
            if profiles is not None:
                available_features = [f for f in available_features if f in profiles.columns]

            if len(available_features) == 0:
                logger.warning(f"  ⚠️  {category_name}: No available features, skipping")
                continue

            logger.info(f"  {category_name}: {len(available_features)} features")

            # Calculate proximity score for this dimension
            score = self.calculate_proximity_score(
                df=df,
                features=available_features,
                cluster_column=cluster_column,
                profiles=profiles
            )

            # Add to results
            score_column_name = f"{category_name.lower()}_score"
            dimensional_scores[score_column_name] = score

        logger.info(f"  ✓ Calculated {len(dimensional_scores.columns)} dimensional scores")

        return dimensional_scores

    # =========================================================================
    # RELATIVE SCORE - Z-Score based
    # =========================================================================

    def calculate_relative_score(
        self,
        df: pd.DataFrame,
        features: List[str],
        cluster_column: str = 'cluster'
    ) -> pd.Series:
        """
        Calculates relative score based on Z-scores within cluster

        Relative Score measures how well a company performs compared to its
        cluster peers. Uses Z-scores (standardized values) within each cluster.

        Formula:
            z_score = (value - cluster_mean) / cluster_std
            avg_z_score = mean(z_scores across all features)
            score = avg_z_score * 10 + 50

        This transforms Z-scores to ~0-100 scale:
        - z = 0 (average) → score = 50
        - z = 1 (1 std above) → score = 60
        - z = 2 (2 std above) → score = 70

        Args:
            df: DataFrame with cluster assignments and features
            features: List of features for comparison
            cluster_column: Name of cluster column

        Returns:
            pd.Series with relative scores (0-100)
        """
        logger.info(f"\nCalculating Relative Scores...")
        logger.info(f"  Features: {len(features)}")

        # Validate features
        available_features = [f for f in features if f in df.columns]
        if len(available_features) == 0:
            logger.error("  ❌ No valid features!")
            return pd.Series(50, index=df.index)  # Return neutral score

        df_clean = df.copy()

        # Calculate Z-scores within each cluster
        z_scores = pd.DataFrame(index=df.index)

        for cluster_id in df_clean[cluster_column].unique():
            if cluster_id < 0:  # Skip noise points
                continue

            cluster_mask = df_clean[cluster_column] == cluster_id

            for feature in available_features:
                cluster_data = df_clean.loc[cluster_mask, feature]

                # Skip if all NaN
                if cluster_data.isna().all():
                    continue

                # Calculate cluster statistics
                cluster_mean = cluster_data.mean()
                cluster_std = cluster_data.std()

                # Handle edge case: std = 0 (all values identical)
                if cluster_std == 0 or pd.isna(cluster_std):
                    z_scores.loc[cluster_mask, feature] = 0
                else:
                    # Calculate Z-score
                    z_scores.loc[cluster_mask, feature] = (
                        (df_clean.loc[cluster_mask, feature] - cluster_mean) / cluster_std
                    )

        # Average Z-score across all features
        avg_z_score = z_scores.mean(axis=1)

        # Transform to 0-100 scale: score = z_score * 10 + 50
        scores = avg_z_score * 10 + 50

        # Clip to [0, 100]
        scores = np.clip(scores, 0, 100)

        relative_series = pd.Series(scores, index=df.index, name='relative_score')

        logger.info(f"  ✓ Relative Scores: Mean={relative_series.mean():.1f}, "
                   f"Std={relative_series.std():.1f}, "
                   f"Min={relative_series.min():.1f}, "
                   f"Max={relative_series.max():.1f}")

        return relative_series

    # =========================================================================
    # OVERALL SCORE - Weighted Combination
    # =========================================================================

    def calculate_overall_score(
        self,
        proximity_score: pd.Series,
        dimensional_scores: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculates overall score as weighted average of all scores

        Args:
            proximity_score: Proximity score series
            dimensional_scores: DataFrame with dimensional scores
            weights: Custom weights (optional)
                Default: {'proximity': 0.4, 'profitability': 0.2,
                         'leverage': 0.1, 'efficiency': 0.15, 'growth': 0.15}

        Returns:
            pd.Series with overall scores (0-100)
        """
        logger.info(f"\nCalculating Overall Scores...")

        # Default weights
        default_weights = {
            'proximity': 0.4,
            'profitability': 0.2,
            'leverage': 0.1,
            'efficiency': 0.15,
            'growth': 0.15
        }

        weights = weights or default_weights

        # Combine all scores
        all_scores = pd.DataFrame(index=proximity_score.index)
        all_scores['proximity'] = proximity_score

        # Add dimensional scores
        for col in dimensional_scores.columns:
            # Extract category name from column (e.g., 'profitability_score' -> 'profitability')
            category = col.replace('_score', '')
            all_scores[category] = dimensional_scores[col]

        # Calculate weighted average
        overall_scores = pd.Series(0.0, index=all_scores.index)

        for score_name, weight in weights.items():
            if score_name in all_scores.columns:
                overall_scores += all_scores[score_name] * weight
                logger.info(f"  {score_name}: weight={weight:.2f}")
            else:
                logger.warning(f"  ⚠️  {score_name}: not found, skipping")

        # Ensure in [0, 100]
        overall_scores = np.clip(overall_scores, 0, 100)

        overall_series = pd.Series(overall_scores, index=all_scores.index, name='overall_score')

        logger.info(f"  ✓ Overall Scores: Mean={overall_series.mean():.1f}, "
                   f"Std={overall_series.std():.1f}, "
                   f"Min={overall_series.min():.1f}, "
                   f"Max={overall_series.max():.1f}")

        return overall_series

    # =========================================================================
    # CONVENIENCE METHOD - Calculate All Scores
    # =========================================================================

    def calculate_all_scores(
        self,
        df: pd.DataFrame,
        features: List[str],
        cluster_column: str = 'cluster',
        profiles: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculates all scores at once and returns combined DataFrame

        Args:
            df: DataFrame with cluster assignments and features
            features: List of features
            cluster_column: Name of cluster column
            profiles: Cluster profiles (optional)
            weights: Custom weights for overall score (optional)

        Returns:
            DataFrame with all score columns added
        """
        logger.info("\n" + "="*80)
        logger.info("CALCULATING ALL SCORES")
        logger.info("="*80)

        result_df = df.copy()

        # 1. Proximity Score
        proximity = self.calculate_proximity_score(
            df=result_df,
            features=features,
            cluster_column=cluster_column,
            profiles=profiles
        )
        result_df['proximity_score'] = proximity

        # 2. Dimensional Scores
        dimensional = self.calculate_dimensional_scores(
            df=result_df,
            cluster_column=cluster_column,
            profiles=profiles
        )
        for col in dimensional.columns:
            result_df[col] = dimensional[col]

        # 3. Relative Score
        relative = self.calculate_relative_score(
            df=result_df,
            features=features,
            cluster_column=cluster_column
        )
        result_df['relative_score'] = relative

        # 4. Overall Score
        overall = self.calculate_overall_score(
            proximity_score=proximity,
            dimensional_scores=dimensional,
            weights=weights
        )
        result_df['overall_score'] = overall

        logger.info("\n" + "="*80)
        logger.info(f"✓ All scores calculated successfully")
        logger.info(f"  Added {1 + len(dimensional.columns) + 2} score columns")
        logger.info("="*80 + "\n")

        return result_df


if __name__ == "__main__":
    # Test ScoreCalculator
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("SCORE CALCULATOR TEST")
    print("="*80)

    # Create mock data
    np.random.seed(42)
    n_companies = 100

    mock_df = pd.DataFrame({
        'gvkey': [f'{i:03d}' for i in range(n_companies)],
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'cluster': np.random.choice([0, 1, 2, 3], n_companies),
        'roa': np.random.normal(0.1, 0.05, n_companies),
        'roe': np.random.normal(0.15, 0.08, n_companies),
        'ebit_margin': np.random.normal(0.12, 0.06, n_companies),
        'debt_to_equity': np.random.normal(1.0, 0.5, n_companies),
        'current_ratio': np.random.normal(1.5, 0.3, n_companies),
        'asset_turnover': np.random.normal(1.2, 0.4, n_companies)
    })

    features = ['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'current_ratio', 'asset_turnover']

    # Test Calculator
    calculator = ScoreCalculator()

    # Test all scores
    result_df = calculator.calculate_all_scores(
        df=mock_df,
        features=features,
        cluster_column='cluster'
    )

    print("\nScore Columns Added:")
    score_cols = [col for col in result_df.columns if 'score' in col]
    for col in score_cols:
        print(f"  • {col}: Mean={result_df[col].mean():.1f}, Std={result_df[col].std():.1f}")

    print("\n✓ ScoreCalculator Test erfolgreich!")
    print("="*80 + "\n")
