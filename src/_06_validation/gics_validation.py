"""
GICS Validation - Validate clusters against GICS sector classifications

Provides specialized tools for:
- GICS sector-based validation
- Sector classification comparison
- GICS-specific metrics and interpretations
- Industry-cluster association analysis
- Sector outlier detection

GICS (Global Industry Classification Standard) is the standard for categorizing
companies into sectors, industry groups, industries, and sub-industries.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .base_validation import BaseValidation

logger = logging.getLogger(__name__)


class GICSValidation(BaseValidation):
    """
    Validate clustering results against GICS sector classifications

    GICS provides hierarchical industry classification:
    - Sectors (11 categories)
    - Industry Groups (25 categories)
    - Industries (74 categories)
    - Sub-Industries (163 categories)

    This class validates clusters against GICS codes at any level.
    """

    # GICS Sector codes and names (Level 1 - 11 sectors)
    GICS_SECTORS = {
        10: "Energy",
        15: "Materials",
        20: "Industrials",
        25: "Consumer Discretionary",
        30: "Consumer Staples",
        35: "Health Care",
        40: "Financials",
        45: "Information Technology",
        50: "Communication Services",
        55: "Utilities",
        60: "Real Estate"
    }

    # Expected cluster-sector relationships for interpretation
    # These can be customized based on specific clustering objectives
    EXPECTED_ASSOCIATIONS = {
        'high': ['Financials', 'Information Technology', 'Health Care'],
        'medium': ['Consumer Discretionary', 'Industrials', 'Communication Services'],
        'low': ['Utilities', 'Energy', 'Materials', 'Consumer Staples', 'Real Estate']
    }

    def __init__(self):
        """Initialize GICS Validation"""
        super().__init__(validation_type="GICS")
        logger.info("  GICS sector validation enabled")

    # =========================================================================
    # GICS-SPECIFIC METHODS
    # =========================================================================

    def get_sector_name(self, gics_code: int) -> str:
        """
        Get sector name from GICS code

        Args:
            gics_code: GICS sector code (2-digit)

        Returns:
            Sector name or 'Unknown'
        """
        # Extract first 2 digits for sector code
        sector_code = int(str(gics_code)[:2])
        return self.GICS_SECTORS.get(sector_code, f"Unknown ({sector_code})")

    def add_sector_names(self, df: pd.DataFrame, gics_column: str = 'gics_code') -> pd.DataFrame:
        """
        Add sector names to DataFrame based on GICS codes

        Args:
            df: DataFrame with GICS codes
            gics_column: Name of GICS code column

        Returns:
            DataFrame with added 'sector_name' column
        """
        df = df.copy()

        if gics_column not in df.columns:
            logger.warning(f"  ⚠️  GICS column '{gics_column}' not found")
            return df

        df['sector_name'] = df[gics_column].apply(
            lambda x: self.get_sector_name(x) if pd.notna(x) else np.nan
        )

        return df

    def validate_gics_sectors(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        gics_column: str = 'gics_code',
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Perform comprehensive GICS sector validation

        This is a convenience method that:
        1. Adds sector names to DataFrame
        2. Validates clusters against GICS sectors
        3. Analyzes sector distribution
        4. Identifies sector-specific outliers
        5. Provides sector-specific interpretations

        Args:
            df: DataFrame with cluster and GICS columns
            cluster_column: Name of cluster column
            gics_column: Name of GICS code column
            save_path: Optional path to save report

        Returns:
            Dictionary with GICS validation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🏭 GICS SECTOR VALIDATION")
        logger.info("=" * 80)

        # Add sector names
        df_with_sectors = self.add_sector_names(df, gics_column)

        # Generate validation report
        report = self.generate_validation_report(
            df_with_sectors,
            cluster_column,
            ['sector_name'],
            save_path
        )

        # Add GICS-specific analysis
        report['gics_specific'] = self._analyze_sector_patterns(
            df_with_sectors,
            cluster_column
        )

        return report

    def _analyze_sector_patterns(
        self,
        df: pd.DataFrame,
        cluster_column: str
    ) -> Dict:
        """
        Analyze cluster-sector patterns

        Args:
            df: DataFrame with cluster and sector_name columns
            cluster_column: Name of cluster column

        Returns:
            Dictionary with sector pattern analysis
        """
        if 'sector_name' not in df.columns:
            return {'error': 'sector_name column not found'}

        analysis = {
            'sector_counts': {},
            'cluster_dominant_sectors': {},
            'sector_dominant_clusters': {},
            'cross_sector_clusters': []
        }

        # Count companies per sector
        sector_counts = df['sector_name'].value_counts().to_dict()
        analysis['sector_counts'] = sector_counts

        # Find dominant sector for each cluster
        for cluster in df[cluster_column].unique():
            cluster_df = df[df[cluster_column] == cluster]
            dominant_sector = cluster_df['sector_name'].mode()
            if len(dominant_sector) > 0:
                dominant_sector = dominant_sector.iloc[0]
                pct = (cluster_df['sector_name'] == dominant_sector).sum() / len(cluster_df) * 100
                analysis['cluster_dominant_sectors'][cluster] = {
                    'sector': dominant_sector,
                    'percentage': pct
                }

        # Find dominant cluster for each sector
        for sector in df['sector_name'].dropna().unique():
            sector_df = df[df['sector_name'] == sector]
            dominant_cluster = sector_df[cluster_column].mode()
            if len(dominant_cluster) > 0:
                dominant_cluster = dominant_cluster.iloc[0]
                pct = (sector_df[cluster_column] == dominant_cluster).sum() / len(sector_df) * 100
                analysis['sector_dominant_clusters'][sector] = {
                    'cluster': dominant_cluster,
                    'percentage': pct
                }

        # Identify clusters that span multiple sectors (cross-sector clusters)
        for cluster in df[cluster_column].unique():
            cluster_df = df[df[cluster_column] == cluster]
            n_sectors = cluster_df['sector_name'].nunique()
            if n_sectors >= 5:  # Cluster spans 5+ sectors
                sector_dist = cluster_df['sector_name'].value_counts()
                analysis['cross_sector_clusters'].append({
                    'cluster': cluster,
                    'n_sectors': n_sectors,
                    'sectors': sector_dist.to_dict()
                })

        return analysis

    def compare_sector_levels(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        gics_columns: List[str]
    ) -> pd.DataFrame:
        """
        Compare validation across different GICS hierarchy levels

        Compare association strength at different GICS levels:
        - Sector (2-digit)
        - Industry Group (4-digit)
        - Industry (6-digit)
        - Sub-Industry (8-digit)

        Args:
            df: DataFrame with cluster and GICS columns at different levels
            cluster_column: Name of cluster column
            gics_columns: List of GICS column names at different levels

        Returns:
            DataFrame comparing Cramér's V across levels
        """
        logger.info("\n📊 Comparing GICS Hierarchy Levels")

        results = self.compare_with_multiple_externals(
            df,
            cluster_column,
            gics_columns
        )

        # Add level interpretation
        results['hierarchy_level'] = results['external_label'].apply(
            self._interpret_gics_level
        )

        return results

    def _interpret_gics_level(self, column_name: str) -> str:
        """
        Interpret GICS hierarchy level from column name

        Args:
            column_name: Name of GICS column

        Returns:
            Hierarchy level description
        """
        name_lower = column_name.lower()
        if 'sector' in name_lower:
            return "Level 1 - Sector"
        elif 'group' in name_lower:
            return "Level 2 - Industry Group"
        elif 'industry' in name_lower and 'sub' not in name_lower:
            return "Level 3 - Industry"
        elif 'sub' in name_lower:
            return "Level 4 - Sub-Industry"
        else:
            return "Unknown Level"

    def identify_sector_mismatches(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        expected_cluster_sectors: Dict[int, List[str]]
    ) -> pd.DataFrame:
        """
        Identify companies in unexpected sector-cluster combinations

        Args:
            df: DataFrame with cluster and sector_name columns
            cluster_column: Name of cluster column
            expected_cluster_sectors: Dict mapping cluster ID to expected sector names

        Returns:
            DataFrame with mismatched companies
        """
        if 'sector_name' not in df.columns:
            logger.warning("  ⚠️  sector_name column not found")
            return pd.DataFrame()

        mismatches = []

        for cluster, expected_sectors in expected_cluster_sectors.items():
            cluster_df = df[df[cluster_column] == cluster]

            # Find companies NOT in expected sectors
            mismatched = cluster_df[~cluster_df['sector_name'].isin(expected_sectors)]

            for _, row in mismatched.iterrows():
                mismatches.append({
                    'gvkey': row.get('gvkey', np.nan),
                    'company_name': row.get('company_name', 'Unknown'),
                    'cluster': cluster,
                    'actual_sector': row['sector_name'],
                    'expected_sectors': ', '.join(expected_sectors)
                })

        return pd.DataFrame(mismatches)


if __name__ == "__main__":
    # Test GICSValidation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("GICS VALIDATION TEST")
    print("=" * 80)

    # Mock data with realistic GICS codes
    np.random.seed(42)
    n = 300

    # Create GICS sector codes (2-digit)
    gics_sectors = list(GICSValidation.GICS_SECTORS.keys())

    # Create correlated data (clusters somewhat related to sectors)
    df = pd.DataFrame({
        'gvkey': range(n),
        'company_name': [f'Company {i}' for i in range(n)],
        'cluster': np.random.randint(0, 4, n)
    })

    # Assign GICS codes with correlation to clusters
    cluster_sector_mapping = {
        0: [10, 15, 55],  # Cluster 0: Energy, Materials, Utilities
        1: [40, 45, 50],  # Cluster 1: Financials, IT, Communication
        2: [25, 30, 35],  # Cluster 2: Consumer Discretionary, Staples, Health
        3: [20, 60]       # Cluster 3: Industrials, Real Estate
    }

    gics_codes = []
    for _, row in df.iterrows():
        cluster = row['cluster']
        # 70% correlation: assign sector from cluster's preferred sectors
        if np.random.random() < 0.7:
            gics_code = np.random.choice(cluster_sector_mapping[cluster])
        else:
            # 30% noise: assign random sector
            gics_code = np.random.choice(gics_sectors)
        gics_codes.append(gics_code)

    df['gics_code'] = gics_codes

    print(f"\nMock data created:")
    print(f"  Companies: {n}")
    print(f"  Clusters: {df['cluster'].nunique()}")
    print(f"  GICS Sectors: {df['gics_code'].nunique()}")

    # Test GICSValidation
    print("\n" + "-" * 80)
    print("Testing GICSValidation")
    print("-" * 80)

    try:
        validator = GICSValidation()

        # 1. Test sector name mapping
        print("\n1. Testing sector name mapping:")
        df_with_sectors = validator.add_sector_names(df, 'gics_code')
        print(f"   Sectors found: {df_with_sectors['sector_name'].nunique()}")
        print(f"   Sample sectors: {df_with_sectors['sector_name'].value_counts().head()}")

        # 2. Test GICS validation
        print("\n2. Testing GICS sector validation:")
        report = validator.validate_gics_sectors(
            df,
            cluster_column='cluster',
            gics_column='gics_code'
        )

        # 3. Test sector pattern analysis
        print("\n3. Sector Pattern Analysis:")
        if 'gics_specific' in report:
            gics_analysis = report['gics_specific']
            print(f"   Total sectors: {len(gics_analysis['sector_counts'])}")
            print(f"   Dominant sectors per cluster:")
            for cluster, info in gics_analysis['cluster_dominant_sectors'].items():
                print(f"     Cluster {cluster}: {info['sector']} ({info['percentage']:.1f}%)")

        # 4. Test sector mismatch identification
        print("\n4. Testing sector mismatch identification:")
        expected_sectors = {
            0: ["Energy", "Materials", "Utilities"],
            1: ["Financials", "Information Technology", "Communication Services"],
            2: ["Consumer Discretionary", "Consumer Staples", "Health Care"],
            3: ["Industrials", "Real Estate"]
        }

        mismatches = validator.identify_sector_mismatches(
            df_with_sectors,
            'cluster',
            expected_sectors
        )
        print(f"   Mismatches found: {len(mismatches)}")
        if len(mismatches) > 0:
            print(f"   Sample mismatches:")
            print(mismatches.head().to_string(index=False))

        # 5. Test contingency table
        print("\n5. Testing GICS contingency table:")
        contingency = validator.create_contingency_table(
            df_with_sectors,
            'cluster',
            'sector_name'
        )
        print(contingency)

        print("\n✓ GICSValidation test successful!")

    except Exception as e:
        print(f"\n❌ GICSValidation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
