"""
Size Validation - Validate clusters against company size classifications

Provides specialized tools for:
- Company size-based validation
- Size classification comparison
- Size-specific metrics and interpretations
- Size-cluster association analysis
- Size category outlier detection

Size categories are typically based on:
- Revenue (Umsatz)
- Total Assets (Bilanzsumme)
- Number of Employees (Mitarbeiterzahl)
- Market Capitalization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .base_validation import BaseValidation

logger = logging.getLogger(__name__)


class SizeValidation(BaseValidation):
    """
    Validate clustering results against company size classifications

    Size categories are used to validate whether clusters align with
    company size (Small, Medium, Large enterprises).

    Common size classification criteria:
    - EU Definition: Based on employees, revenue, and balance sheet total
    - Custom thresholds: Based on industry-specific criteria
    """

    # EU size classification thresholds
    EU_SIZE_THRESHOLDS = {
        'micro': {
            'employees': 10,
            'revenue_mio': 2,
            'assets_mio': 2
        },
        'small': {
            'employees': 50,
            'revenue_mio': 10,
            'assets_mio': 10
        },
        'medium': {
            'employees': 250,
            'revenue_mio': 50,
            'assets_mio': 43
        }
        # 'large' is everything above medium thresholds
    }

    # Standard size categories
    SIZE_CATEGORIES = ['Small', 'Medium', 'Large']

    # Expected cluster-size relationships for interpretation
    # These can be customized based on specific clustering objectives
    EXPECTED_ASSOCIATIONS = {
        'high_performers': 'Large',     # High performers often large companies
        'medium_performers': 'Medium',  # Medium performers medium-sized
        'low_performers': 'Small',      # Low performers often smaller
        'stable': 'Large',              # Stable companies often large
        'growth': 'Small'               # Growth companies often smaller
    }

    def __init__(self):
        """Initialize Size Validation"""
        super().__init__(validation_type="Size")
        logger.info("  Company size validation enabled")

    # =========================================================================
    # SIZE-SPECIFIC METHODS
    # =========================================================================

    def classify_by_revenue(
        self,
        df: pd.DataFrame,
        revenue_column: str = 'revenue',
        thresholds: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Classify companies by revenue into size categories

        Args:
            df: DataFrame with revenue data
            revenue_column: Name of revenue column (in millions)
            thresholds: Custom thresholds dict with 'small' and 'medium' keys
                       Default: {'small': 10, 'medium': 50}

        Returns:
            DataFrame with added 'size_category' column
        """
        df = df.copy()

        if revenue_column not in df.columns:
            logger.warning(f"  ⚠️  Revenue column '{revenue_column}' not found")
            return df

        # Use custom or default thresholds
        if thresholds is None:
            thresholds = {
                'small': self.EU_SIZE_THRESHOLDS['small']['revenue_mio'],
                'medium': self.EU_SIZE_THRESHOLDS['medium']['revenue_mio']
            }

        # Classify
        df['size_category'] = pd.cut(
            df[revenue_column],
            bins=[0, thresholds['small'], thresholds['medium'], float('inf')],
            labels=['Small', 'Medium', 'Large'],
            include_lowest=True
        )

        return df

    def classify_by_assets(
        self,
        df: pd.DataFrame,
        assets_column: str = 'total_assets',
        thresholds: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Classify companies by total assets into size categories

        Args:
            df: DataFrame with assets data
            assets_column: Name of assets column (in millions)
            thresholds: Custom thresholds dict with 'small' and 'medium' keys
                       Default: {'small': 10, 'medium': 43}

        Returns:
            DataFrame with added 'size_category' column
        """
        df = df.copy()

        if assets_column not in df.columns:
            logger.warning(f"  ⚠️  Assets column '{assets_column}' not found")
            return df

        # Use custom or default thresholds
        if thresholds is None:
            thresholds = {
                'small': self.EU_SIZE_THRESHOLDS['small']['assets_mio'],
                'medium': self.EU_SIZE_THRESHOLDS['medium']['assets_mio']
            }

        # Classify
        df['size_category'] = pd.cut(
            df[assets_column],
            bins=[0, thresholds['small'], thresholds['medium'], float('inf')],
            labels=['Small', 'Medium', 'Large'],
            include_lowest=True
        )

        return df

    def classify_by_employees(
        self,
        df: pd.DataFrame,
        employees_column: str = 'employees',
        thresholds: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Classify companies by number of employees into size categories

        Args:
            df: DataFrame with employee data
            employees_column: Name of employees column
            thresholds: Custom thresholds dict with 'small' and 'medium' keys
                       Default: {'small': 50, 'medium': 250}

        Returns:
            DataFrame with added 'size_category' column
        """
        df = df.copy()

        if employees_column not in df.columns:
            logger.warning(f"  ⚠️  Employees column '{employees_column}' not found")
            return df

        # Use custom or default thresholds
        if thresholds is None:
            thresholds = {
                'small': self.EU_SIZE_THRESHOLDS['small']['employees'],
                'medium': self.EU_SIZE_THRESHOLDS['medium']['employees']
            }

        # Classify
        df['size_category'] = pd.cut(
            df[employees_column],
            bins=[0, thresholds['small'], thresholds['medium'], float('inf')],
            labels=['Small', 'Medium', 'Large'],
            include_lowest=True
        )

        return df

    def validate_size_categories(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        size_column: str = 'size_category',
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Perform comprehensive size category validation

        This is a convenience method that:
        1. Validates clusters against size categories
        2. Analyzes size distribution
        3. Identifies size-specific outliers
        4. Provides size-specific interpretations

        Args:
            df: DataFrame with cluster and size columns
            cluster_column: Name of cluster column
            size_column: Name of size category column
            save_path: Optional path to save report

        Returns:
            Dictionary with size validation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("📏 SIZE CATEGORY VALIDATION")
        logger.info("=" * 80)

        # Ensure size_column exists
        if size_column not in df.columns:
            logger.error(f"  ❌ Size column '{size_column}' not found")
            return {'error': f"Column '{size_column}' not found"}

        # Generate validation report
        report = self.generate_validation_report(
            df,
            cluster_column,
            [size_column],
            save_path
        )

        # Add size-specific analysis
        report['size_specific'] = self._analyze_size_patterns(
            df,
            cluster_column,
            size_column
        )

        return report

    def _analyze_size_patterns(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        size_column: str = 'size_category'
    ) -> Dict:
        """
        Analyze cluster-size patterns

        Args:
            df: DataFrame with cluster and size columns
            cluster_column: Name of cluster column
            size_column: Name of size category column

        Returns:
            Dictionary with size pattern analysis
        """
        if size_column not in df.columns:
            return {'error': f'{size_column} column not found'}

        analysis = {
            'size_counts': {},
            'cluster_dominant_sizes': {},
            'size_dominant_clusters': {},
            'size_progression': {}
        }

        # Count companies per size category
        size_counts = df[size_column].value_counts().to_dict()
        analysis['size_counts'] = size_counts

        # Find dominant size for each cluster
        for cluster in df[cluster_column].unique():
            cluster_df = df[df[cluster_column] == cluster]
            dominant_size = cluster_df[size_column].mode()
            if len(dominant_size) > 0:
                dominant_size = dominant_size.iloc[0]
                pct = (cluster_df[size_column] == dominant_size).sum() / len(cluster_df) * 100
                analysis['cluster_dominant_sizes'][cluster] = {
                    'size': dominant_size,
                    'percentage': pct
                }

        # Find dominant cluster for each size
        for size_cat in df[size_column].dropna().unique():
            size_df = df[df[size_column] == size_cat]
            dominant_cluster = size_df[cluster_column].mode()
            if len(dominant_cluster) > 0:
                dominant_cluster = dominant_cluster.iloc[0]
                pct = (size_df[cluster_column] == dominant_cluster).sum() / len(size_df) * 100
                analysis['size_dominant_clusters'][size_cat] = {
                    'cluster': dominant_cluster,
                    'percentage': pct
                }

        # Analyze size progression across clusters (ordered by cluster ID)
        for size_cat in ['Small', 'Medium', 'Large']:
            if size_cat in df[size_column].values:
                size_dist = df[df[size_column] == size_cat][cluster_column].value_counts()
                analysis['size_progression'][size_cat] = size_dist.to_dict()

        return analysis

    def compare_size_metrics(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        size_columns: List[str]
    ) -> pd.DataFrame:
        """
        Compare validation across different size metrics

        Compare association strength across different size measures:
        - Size by revenue
        - Size by assets
        - Size by employees
        - Size by market cap

        Args:
            df: DataFrame with cluster and multiple size columns
            cluster_column: Name of cluster column
            size_columns: List of size category column names

        Returns:
            DataFrame comparing Cramér's V across size metrics
        """
        logger.info("\n📊 Comparing Size Metrics")

        results = self.compare_with_multiple_externals(
            df,
            cluster_column,
            size_columns
        )

        # Add metric type interpretation
        results['metric_type'] = results['external_label'].apply(
            self._interpret_size_metric
        )

        return results

    def _interpret_size_metric(self, column_name: str) -> str:
        """
        Interpret size metric type from column name

        Args:
            column_name: Name of size column

        Returns:
            Metric type description
        """
        name_lower = column_name.lower()
        if 'revenue' in name_lower or 'umsatz' in name_lower:
            return "Revenue-based"
        elif 'asset' in name_lower or 'bilanz' in name_lower:
            return "Assets-based"
        elif 'employee' in name_lower or 'mitarbeiter' in name_lower:
            return "Employees-based"
        elif 'market' in name_lower or 'cap' in name_lower:
            return "Market Cap-based"
        else:
            return "Unknown Metric"

    def identify_size_mismatches(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        size_column: str,
        expected_cluster_sizes: Dict[int, str]
    ) -> pd.DataFrame:
        """
        Identify companies in unexpected size-cluster combinations

        Args:
            df: DataFrame with cluster and size columns
            cluster_column: Name of cluster column
            size_column: Name of size category column
            expected_cluster_sizes: Dict mapping cluster ID to expected size category

        Returns:
            DataFrame with mismatched companies
        """
        if size_column not in df.columns:
            logger.warning(f"  ⚠️  {size_column} column not found")
            return pd.DataFrame()

        mismatches = []

        for cluster, expected_size in expected_cluster_sizes.items():
            cluster_df = df[df[cluster_column] == cluster]

            # Find companies NOT in expected size category
            mismatched = cluster_df[cluster_df[size_column] != expected_size]

            for _, row in mismatched.iterrows():
                mismatches.append({
                    'gvkey': row.get('gvkey', np.nan),
                    'company_name': row.get('company_name', 'Unknown'),
                    'cluster': cluster,
                    'actual_size': row[size_column],
                    'expected_size': expected_size
                })

        return pd.DataFrame(mismatches)

    def analyze_size_transitions(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        size_column: str = 'size_category'
    ) -> Dict:
        """
        Analyze how size categories are distributed across clusters

        Useful for understanding if clustering creates size-based separations

        Args:
            df: DataFrame with cluster and size columns
            cluster_column: Name of cluster column
            size_column: Name of size category column

        Returns:
            Dictionary with size transition analysis
        """
        if size_column not in df.columns:
            return {'error': f'{size_column} column not found'}

        # Calculate size distribution per cluster
        distribution = pd.crosstab(
            df[cluster_column],
            df[size_column],
            normalize='index'
        ) * 100  # Convert to percentages

        # Identify size-homogeneous clusters (>70% one size)
        homogeneous_clusters = []
        for cluster in distribution.index:
            max_pct = distribution.loc[cluster].max()
            if max_pct > 70:
                dominant_size = distribution.loc[cluster].idxmax()
                homogeneous_clusters.append({
                    'cluster': cluster,
                    'dominant_size': dominant_size,
                    'percentage': max_pct
                })

        # Identify size-diverse clusters (<50% largest size)
        diverse_clusters = []
        for cluster in distribution.index:
            max_pct = distribution.loc[cluster].max()
            if max_pct < 50:
                diverse_clusters.append({
                    'cluster': cluster,
                    'largest_size_percentage': max_pct
                })

        return {
            'distribution_matrix': distribution,
            'homogeneous_clusters': homogeneous_clusters,
            'diverse_clusters': diverse_clusters
        }


if __name__ == "__main__":
    # Test SizeValidation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("SIZE VALIDATION TEST")
    print("=" * 80)

    # Mock data with realistic size metrics
    np.random.seed(42)
    n = 300

    # Create correlated data (clusters somewhat related to size)
    df = pd.DataFrame({
        'gvkey': range(n),
        'company_name': [f'Company {i}' for i in range(n)],
        'cluster': np.random.randint(0, 4, n)
    })

    # Create size-related metrics with correlation to clusters
    # Cluster 0: Mostly large companies
    # Cluster 1: Mostly medium companies
    # Cluster 2: Mostly small companies
    # Cluster 3: Mixed sizes

    revenues = []
    assets = []
    employees = []

    for _, row in df.iterrows():
        cluster = row['cluster']

        # Generate revenue based on cluster (with noise)
        if cluster == 0:  # Large companies
            revenue = np.random.lognormal(4.5, 0.8)  # Mean ~90, high variance
        elif cluster == 1:  # Medium companies
            revenue = np.random.lognormal(3.5, 0.6)  # Mean ~33
        elif cluster == 2:  # Small companies
            revenue = np.random.lognormal(2.0, 0.7)  # Mean ~7.4
        else:  # Mixed
            revenue = np.random.lognormal(3.0, 1.2)  # High variance

        revenues.append(revenue)

        # Assets correlate with revenue
        assets.append(revenue * np.random.uniform(0.8, 1.5))

        # Employees correlate with revenue (log scale)
        employees.append(int(revenue * np.random.uniform(20, 50)))

    df['revenue'] = revenues
    df['total_assets'] = assets
    df['employees'] = employees

    print(f"\nMock data created:")
    print(f"  Companies: {n}")
    print(f"  Clusters: {df['cluster'].nunique()}")
    print(f"  Revenue range: {df['revenue'].min():.1f} - {df['revenue'].max():.1f}")
    print(f"  Assets range: {df['total_assets'].min():.1f} - {df['total_assets'].max():.1f}")

    # Test SizeValidation
    print("\n" + "-" * 80)
    print("Testing SizeValidation")
    print("-" * 80)

    try:
        validator = SizeValidation()

        # 1. Test size classification by revenue
        print("\n1. Testing size classification by revenue:")
        df_with_size = validator.classify_by_revenue(df, 'revenue')
        print(f"   Size distribution: {df_with_size['size_category'].value_counts().to_dict()}")

        # 2. Test size validation
        print("\n2. Testing size category validation:")
        report = validator.validate_size_categories(
            df_with_size,
            cluster_column='cluster',
            size_column='size_category'
        )

        # 3. Test size pattern analysis
        print("\n3. Size Pattern Analysis:")
        if 'size_specific' in report:
            size_analysis = report['size_specific']
            print(f"   Total sizes: {len(size_analysis['size_counts'])}")
            print(f"   Dominant sizes per cluster:")
            for cluster, info in size_analysis['cluster_dominant_sizes'].items():
                print(f"     Cluster {cluster}: {info['size']} ({info['percentage']:.1f}%)")

        # 4. Test size transitions
        print("\n4. Testing size transition analysis:")
        transitions = validator.analyze_size_transitions(
            df_with_size,
            'cluster',
            'size_category'
        )
        print(f"   Size distribution matrix:")
        print(transitions['distribution_matrix'].round(1))
        print(f"\n   Homogeneous clusters: {len(transitions['homogeneous_clusters'])}")
        print(f"   Diverse clusters: {len(transitions['diverse_clusters'])}")

        # 5. Test size mismatch identification
        print("\n5. Testing size mismatch identification:")
        expected_sizes = {
            0: "Large",
            1: "Medium",
            2: "Small",
            3: "Medium"
        }

        mismatches = validator.identify_size_mismatches(
            df_with_size,
            'cluster',
            'size_category',
            expected_sizes
        )
        print(f"   Mismatches found: {len(mismatches)}")
        if len(mismatches) > 0:
            print(f"   Sample mismatches:")
            print(mismatches.head().to_string(index=False))

        # 6. Test contingency table
        print("\n6. Testing size contingency table:")
        contingency = validator.create_contingency_table(
            df_with_size,
            'cluster',
            'size_category'
        )
        print(contingency)

        print("\n✓ SizeValidation test successful!")

    except Exception as e:
        print(f"\n❌ SizeValidation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
