"""
Cluster Naming - Automatic Feature-Based Cluster Naming

Generates descriptive cluster names based on dominant financial characteristics:
- Technical Style: "High ROA, Low Debt"
- Business Style: "Cash Generators", "Growth Stars"
- Hybrid Style: "High Performers (Cash Generators)"

Uses Z-Score analysis to identify dominant features per cluster.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from src._01_setup.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class ClusterNamer:
    """
    Generates descriptive names for clusters based on their feature profiles

    Three naming styles:
    1. Technical: Feature-based (e.g., "High ROA, Low Debt")
    2. Business: Archetype-based (e.g., "Cash Generators")
    3. Hybrid: Combined (e.g., "High Performers (Cash Generators)")
    """

    def __init__(self, feature_selector: Optional[FeatureSelector] = None):
        """
        Initialize Cluster Namer

        Args:
            feature_selector: FeatureSelector instance (creates new one if None)
        """
        self.feature_selector = feature_selector or FeatureSelector()
        logger.info("✓ ClusterNamer initialized")

    # =========================================================================
    # MAIN METHOD - Generate Names
    # =========================================================================

    def generate_names(
        self,
        profiles: pd.DataFrame,
        features: List[str],
        style: str = 'hybrid',
        top_n: int = 2
    ) -> Dict[int, str]:
        """
        Generates descriptive names for all clusters

        Args:
            profiles: Cluster profiles DataFrame (index = cluster_id, columns = features)
            features: List of features used for clustering
            style: 'technical', 'business', or 'hybrid'
            top_n: Number of dominant features to consider (default: 2)

        Returns:
            Dict mapping cluster_id to cluster_name
        """
        logger.info(f"\nGenerating Cluster Names (style={style})...")
        logger.info(f"  Clusters: {len(profiles)}")
        logger.info(f"  Features: {len(features)}")

        if style not in ['technical', 'business', 'hybrid']:
            logger.warning(f"  ⚠️  Invalid style '{style}', using 'hybrid'")
            style = 'hybrid'

        # Identify dominant features for each cluster
        dominant_features_df = self._identify_dominant_features(profiles, features, top_n)

        # Generate names based on style
        cluster_names = {}

        for cluster_id in profiles.index:
            if style == 'technical':
                name = self._generate_technical_name(cluster_id, dominant_features_df, profiles)
            elif style == 'business':
                archetype = self._classify_archetype(profiles.loc[cluster_id], features)
                name = archetype
            else:  # hybrid
                archetype = self._classify_archetype(profiles.loc[cluster_id], features)
                name = self._generate_hybrid_name(cluster_id, archetype, dominant_features_df, profiles)

            cluster_names[cluster_id] = name
            logger.info(f"  Cluster {cluster_id}: {name}")

        logger.info(f"✓ Generated {len(cluster_names)} cluster names")

        return cluster_names

    # =========================================================================
    # FEATURE DOMINANCE ANALYSIS
    # =========================================================================

    def _identify_dominant_features(
        self,
        profiles: pd.DataFrame,
        features: List[str],
        top_n: int = 2
    ) -> pd.DataFrame:
        """
        Identifies dominant features for each cluster using Z-scores

        Z-scores are calculated across clusters (not companies), identifying
        which features make each cluster distinctive.

        Args:
            profiles: Cluster profiles DataFrame
            features: List of features to analyze
            top_n: Number of top dominant features to return

        Returns:
            DataFrame with dominant features per cluster:
                cluster_id, rank, feature, z_score, direction, value
        """
        logger.info(f"\n  Analyzing Feature Dominance (top {top_n})...")

        # Filter to available features
        available_features = [f for f in features if f in profiles.columns]

        if len(available_features) == 0:
            logger.warning("  ⚠️  No features available for analysis")
            return pd.DataFrame()

        # Calculate Z-scores across clusters
        z_scores = pd.DataFrame(index=profiles.index, columns=available_features)

        for feature in available_features:
            feature_values = profiles[feature]
            mean = feature_values.mean()
            std = feature_values.std()

            if std == 0 or pd.isna(std):
                # All clusters have same value - no dominance
                z_scores[feature] = 0
            else:
                z_scores[feature] = (feature_values - mean) / std

        # Identify top N features per cluster
        dominant_features = []

        for cluster_id in profiles.index:
            cluster_z_scores = z_scores.loc[cluster_id]

            # Get features sorted by absolute Z-score
            sorted_features = cluster_z_scores.abs().sort_values(ascending=False)

            # Take top N features with |z| > 1.0 (1 std deviation)
            count = 0
            for rank, (feature, abs_z) in enumerate(sorted_features.items(), 1):
                z_score = cluster_z_scores[feature]

                # Only include features with significant Z-score
                if abs(z_score) >= 1.0 and count < top_n:
                    direction = 'High' if z_score > 0 else 'Low'
                    value = profiles.loc[cluster_id, feature]

                    dominant_features.append({
                        'cluster': cluster_id,
                        'rank': rank,
                        'feature': feature,
                        'z_score': z_score,
                        'direction': direction,
                        'value': value
                    })
                    count += 1

            # If no dominant features (all |z| < 1.0), take top 1
            if count == 0:
                top_feature = sorted_features.index[0]
                z_score = cluster_z_scores[top_feature]
                direction = 'Moderate' if abs(z_score) < 0.5 else ('High' if z_score > 0 else 'Low')
                value = profiles.loc[cluster_id, top_feature]

                dominant_features.append({
                    'cluster': cluster_id,
                    'rank': 1,
                    'feature': top_feature,
                    'z_score': z_score,
                    'direction': direction,
                    'value': value
                })

        dominant_df = pd.DataFrame(dominant_features)

        logger.info(f"  ✓ Identified dominant features for {len(profiles)} clusters")

        return dominant_df

    # =========================================================================
    # TECHNICAL NAME GENERATION
    # =========================================================================

    def _generate_technical_name(
        self,
        cluster_id: int,
        dominant_features: pd.DataFrame,
        profiles: pd.DataFrame
    ) -> str:
        """
        Generates technical name based on dominant features

        Format: "Direction Feature1, Direction Feature2"
        Example: "High ROA, Low Debt/Equity"

        Args:
            cluster_id: Cluster ID
            dominant_features: DataFrame with dominant features
            profiles: Cluster profiles

        Returns:
            Technical cluster name
        """
        cluster_dominant = dominant_features[dominant_features['cluster'] == cluster_id]

        if len(cluster_dominant) == 0:
            return f"Cluster {cluster_id}"

        # Build name from dominant features
        name_parts = []

        for _, row in cluster_dominant.iterrows():
            feature = row['feature']
            direction = row['direction']
            value = row['value']

            # Get display name
            display_name = self._get_feature_display_name(feature)

            # Format with value (optional)
            formatted_value = self._format_feature_value(feature, value)

            if formatted_value:
                part = f"{direction} {display_name} ({formatted_value})"
            else:
                part = f"{direction} {display_name}"

            name_parts.append(part)

        # Combine (max 2 to avoid too long names)
        name = ", ".join(name_parts[:2])

        # Ensure max length
        if len(name) > 60:
            # Fallback: shorter version without values
            name_parts_short = []
            for _, row in cluster_dominant.iterrows():
                feature = row['feature']
                direction = row['direction']
                display_name = self._get_feature_display_name(feature)
                name_parts_short.append(f"{direction} {display_name}")

            name = ", ".join(name_parts_short[:2])

        return name

    # =========================================================================
    # BUSINESS ARCHETYPE CLASSIFICATION
    # =========================================================================

    def _classify_archetype(
        self,
        profile: pd.Series,
        features: List[str]
    ) -> str:
        """
        Classifies cluster into business archetype

        Archetypes:
        - Cash Generators: High ROA + Low Debt + Positive Cash Flow
        - Growth Stars: High Revenue Growth + Moderate Profitability
        - Value Players: Moderate ROA + Low Debt + Stable
        - Distressed: Low ROA + High Debt
        - Efficient Operators: High Asset Turnover + Moderate ROA
        - Leveraged Growth: High Growth + High Debt
        - Quality Defensive: High ROE + Low Debt + Low Growth

        Args:
            profile: Cluster profile (Series with feature values)
            features: List of available features

        Returns:
            Archetype name
        """
        # Extract key metrics (with defaults if missing)
        roa = profile.get('roa', 0)
        roe = profile.get('roe', 0)
        debt_to_equity = profile.get('debt_to_equity', 1.0)
        revenue_growth = profile.get('revenue_growth', 0)
        asset_turnover = profile.get('asset_turnover', 1.0)
        fcf_margin = profile.get('fcf_margin', 0)
        ebit_margin = profile.get('ebit_margin', 0)

        # Define dynamic thresholds (based on "typical" values)
        # These are reasonable defaults for financial ratios
        high_roa = roa > 10  # >10% ROA is strong
        moderate_roa = 5 <= roa <= 10
        low_roa = roa < 5

        high_roe = roe > 15  # >15% ROE is strong
        moderate_roe = 10 <= roe <= 15

        low_debt = debt_to_equity < 0.5
        moderate_debt = 0.5 <= debt_to_equity <= 1.5
        high_debt = debt_to_equity > 1.5

        high_growth = revenue_growth > 10  # >10% growth is high
        moderate_growth = 5 <= revenue_growth <= 10
        low_growth = revenue_growth < 5
        negative_growth = revenue_growth < 0

        high_turnover = asset_turnover > 1.5
        high_fcf = fcf_margin > 5

        # Classification logic (Decision Tree)

        # 1. Cash Generators - Strong profitability, low debt, good cash flow
        if high_roa and low_debt and high_fcf:
            return "Cash Generators"

        # 2. Growth Stars - High growth with decent profitability
        if high_growth and (moderate_roa or high_roa):
            return "Growth Stars"

        # 3. Quality Defensive - High quality, low debt, but low growth
        if high_roe and low_debt and low_growth:
            return "Quality Defensive"

        # 4. Leveraged Growth - High growth but also high debt
        if high_growth and high_debt:
            return "Leveraged Growth"

        # 5. Efficient Operators - High asset turnover, moderate profitability
        if high_turnover and moderate_roa:
            return "Efficient Operators"

        # 6. Distressed - Low profitability + high debt
        if low_roa and high_debt:
            return "Distressed"

        # 7. Turnaround Candidates - Negative growth but not terrible debt
        if negative_growth and not high_debt:
            return "Turnaround Candidates"

        # 8. Value Players - Moderate everything, stable
        if moderate_roa and moderate_debt and moderate_growth:
            return "Value Players"

        # 9. Mature Stable - Low growth but solid profitability
        if low_growth and (moderate_roa or high_roa) and low_debt:
            return "Mature Stable"

        # Default: Balanced
        return "Balanced"

    # =========================================================================
    # HYBRID NAME GENERATION
    # =========================================================================

    def _generate_hybrid_name(
        self,
        cluster_id: int,
        archetype: str,
        dominant_features: pd.DataFrame,
        profiles: pd.DataFrame
    ) -> str:
        """
        Generates hybrid name combining performance level and archetype

        Format: "Performance Level (Archetype)"
        Example: "High Performers (Cash Generators)"

        Args:
            cluster_id: Cluster ID
            archetype: Business archetype
            dominant_features: DataFrame with dominant features
            profiles: Cluster profiles

        Returns:
            Hybrid cluster name
        """
        profile = profiles.loc[cluster_id]

        # Determine performance level
        performance_level = self._determine_performance_level(profile, profiles)

        # Build hybrid name
        hybrid_name = f"{performance_level} ({archetype})"

        # Optionally add top feature for context
        cluster_dominant = dominant_features[dominant_features['cluster'] == cluster_id]

        if len(cluster_dominant) > 0:
            top_feature = cluster_dominant.iloc[0]
            feature = top_feature['feature']
            value = top_feature['value']

            formatted_value = self._format_feature_value(feature, value)
            if formatted_value:
                hybrid_name += f" - {self._get_feature_display_name(feature)}: {formatted_value}"

        # Ensure max length
        if len(hybrid_name) > 60:
            hybrid_name = f"{performance_level} ({archetype})"

        return hybrid_name

    def _determine_performance_level(
        self,
        profile: pd.Series,
        all_profiles: pd.DataFrame
    ) -> str:
        """
        Determines performance level based on ROA and ROE

        Levels:
        - "High Performers": Avg profitability > Mean + 1 Std
        - "Mid-Tier": Within ±1 Std
        - "Challenged": Avg profitability < Mean - 1 Std

        Args:
            profile: Single cluster profile
            all_profiles: All cluster profiles (for statistics)

        Returns:
            Performance level string
        """
        # Calculate average profitability (ROA + ROE) / 2
        roa = profile.get('roa', 0)
        roe = profile.get('roe', 0)
        avg_profitability = (roa + roe) / 2

        # Calculate statistics across all clusters
        all_roa = all_profiles.get('roa', pd.Series([0]))
        all_roe = all_profiles.get('roe', pd.Series([0]))
        all_avg_profitability = (all_roa + all_roe) / 2

        mean_prof = all_avg_profitability.mean()
        std_prof = all_avg_profitability.std()

        # Classify
        if std_prof == 0:
            return "Mid-Tier"

        if avg_profitability > mean_prof + std_prof:
            return "High Performers"
        elif avg_profitability < mean_prof - std_prof:
            return "Challenged"
        else:
            return "Mid-Tier"

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_feature_display_name(self, feature: str) -> str:
        """
        Gets human-readable display name for feature

        Uses FeatureSelector metadata if available, otherwise formats feature name

        Args:
            feature: Feature name (e.g., 'roa')

        Returns:
            Display name (e.g., 'Return on Assets' or 'ROA')
        """
        metadata = self.feature_selector.get_feature_metadata(feature)
        display_name = metadata.get('name', feature)

        # Shorten common names for brevity
        name_shortcuts = {
            'Return on Assets': 'ROA',
            'Return on Equity': 'ROE',
            'Debt-to-Equity': 'Debt/Equity',
            'Free Cash Flow Margin': 'FCF Margin',
            'Revenue Growth': 'Revenue Growth',
            'Asset Turnover': 'Asset Turnover'
        }

        return name_shortcuts.get(display_name, display_name)

    def _format_feature_value(self, feature: str, value: float) -> str:
        """
        Formats feature value for display

        Args:
            feature: Feature name
            value: Feature value

        Returns:
            Formatted string (e.g., "15.2%", "1.25x", "+12.1%")
        """
        if pd.isna(value):
            return ""

        # Percentage features (margins, returns, growth)
        percent_features = [
            'roa', 'roe', 'ebit_margin', 'ebitda_margin', 'net_profit_margin',
            'operating_margin', 'gross_margin', 'fcf_margin',
            'revenue_growth', 'asset_growth', 'earnings_growth'
        ]

        # Ratio features
        ratio_features = [
            'debt_to_equity', 'current_ratio', 'quick_ratio', 'cash_ratio',
            'asset_turnover', 'inventory_turnover', 'receivables_turnover',
            'interest_coverage'
        ]

        if any(feat in feature for feat in percent_features):
            # Percentage format
            if 'growth' in feature:
                # Growth with +/- sign
                return f"{value:+.1f}%"
            else:
                # Regular percentage
                return f"{value:.1f}%"

        elif any(feat in feature for feat in ratio_features):
            # Ratio format
            return f"{value:.2f}x"

        else:
            # Default: round to 1 decimal
            return f"{value:.1f}"

    # =========================================================================
    # DATAFRAME INTEGRATION
    # =========================================================================

    def rename_clusters(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        names: Dict[int, str],
        new_column: str = 'cluster_name'
    ) -> pd.DataFrame:
        """
        Adds cluster names to DataFrame

        Args:
            df: DataFrame with cluster assignments
            cluster_column: Name of cluster column (e.g., 'cluster')
            names: Dict mapping cluster_id to cluster_name
            new_column: Name for new column with cluster names

        Returns:
            DataFrame with new cluster_name column
        """
        logger.info(f"\nAdding cluster names to DataFrame...")
        logger.info(f"  Cluster Column: {cluster_column}")
        logger.info(f"  New Column: {new_column}")

        df_result = df.copy()

        # Map cluster IDs to names
        df_result[new_column] = df_result[cluster_column].map(names)

        # Handle unmapped clusters (e.g., noise points)
        unmapped = df_result[new_column].isna()
        if unmapped.any():
            df_result.loc[unmapped, new_column] = df_result.loc[unmapped, cluster_column].apply(
                lambda x: f"Cluster {x}" if x >= 0 else "Noise"
            )

        logger.info(f"  ✓ Added {new_column} column")
        logger.info(f"  Unique Names: {df_result[new_column].nunique()}")

        return df_result


if __name__ == "__main__":
    # Test ClusterNamer
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("CLUSTER NAMER TEST")
    print("="*80)

    # Mock Cluster-Profile
    profiles = pd.DataFrame({
        'roa': [15.2, 8.3, 3.1],
        'roe': [22.1, 12.5, 5.2],
        'ebit_margin': [18.5, 10.2, 4.3],
        'debt_to_equity': [0.3, 0.8, 1.5],
        'revenue_growth': [8.5, 12.1, -2.3],
        'current_ratio': [2.1, 1.5, 0.9],
        'asset_turnover': [1.8, 2.2, 1.1],
        'fcf_margin': [12.3, 6.5, -1.2]
    }, index=[0, 1, 2])

    features = ['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'revenue_growth',
                'current_ratio', 'asset_turnover', 'fcf_margin']

    # Initialize Namer
    namer = ClusterNamer()

    # Test 1: Technical Names
    print("\n1. TECHNICAL NAMES:")
    technical_names = namer.generate_names(profiles, features, style='technical')
    for cluster_id, name in technical_names.items():
        print(f"   Cluster {cluster_id}: {name}")

    # Test 2: Business Names
    print("\n2. BUSINESS NAMES:")
    business_names = namer.generate_names(profiles, features, style='business')
    for cluster_id, name in business_names.items():
        print(f"   Cluster {cluster_id}: {name}")

    # Test 3: Hybrid Names
    print("\n3. HYBRID NAMES:")
    hybrid_names = namer.generate_names(profiles, features, style='hybrid')
    for cluster_id, name in hybrid_names.items():
        print(f"   Cluster {cluster_id}: {name}")

    # Test 4: DataFrame Integration
    print("\n4. DATAFRAME INTEGRATION:")
    df = pd.DataFrame({
        'gvkey': ['001', '002', '003', '004', '005', '006'],
        'company_name': ['CompanyA', 'CompanyB', 'CompanyC', 'CompanyD', 'CompanyE', 'CompanyF'],
        'cluster': [0, 0, 1, 1, 2, 2]
    })

    df_renamed = namer.rename_clusters(df, 'cluster', hybrid_names)
    print(df_renamed[['company_name', 'cluster', 'cluster_name']])

    print("\n✓ ClusterNamer Test erfolgreich!")
    print("="*80 + "\n")
