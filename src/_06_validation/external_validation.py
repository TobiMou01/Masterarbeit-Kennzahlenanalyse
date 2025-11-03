"""
External Validation - Validate clusters against external labels

Provides tools for:
- Calculating Cramér's V (association measure)
- Performing Chi²-Tests for independence
- Creating contingency tables
- Analyzing cluster distribution in external labels
- Identifying external outliers (unusual combinations)
- Generating comprehensive validation reports
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy.stats import chi2_contingency
import warnings

logger = logging.getLogger(__name__)


class ExternalValidation:
    """
    Validate clustering results against external categorical labels

    External labels can be:
    - Industry codes (GICS, NAICS, etc.)
    - Size categories (Small, Medium, Large)
    - Countries
    - Any categorical variable

    Workflow:
    1. Calculate Cramér's V (association strength)
    2. Perform Chi²-Test (statistical significance)
    3. Create contingency tables (overlap patterns)
    4. Analyze cluster distributions
    5. Identify unusual combinations (outliers)
    """

    def __init__(self):
        """Initialize External Validation"""
        logger.info("✓ ExternalValidation initialized")

    # =========================================================================
    # CRAMÉR'S V
    # =========================================================================

    def calculate_cramers_v(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str
    ) -> float:
        """
        Calculate Cramér's V between clusters and external labels

        Cramér's V measures association between categorical variables:
        - V = 0: No association (independent)
        - V = 1: Perfect association

        Formula: V = sqrt(chi2 / (n * (min(r,c) - 1)))

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_column: Name of external label column

        Returns:
            Cramér's V (0-1)
        """
        # Remove missing values
        df_clean = df[[cluster_column, external_column]].dropna()

        if len(df_clean) == 0:
            logger.warning(f"  ⚠️  No valid data for {external_column}")
            return 0.0

        # Create contingency table
        contingency = pd.crosstab(df_clean[cluster_column], df_clean[external_column])

        # Chi²-Test
        chi2, p, dof, expected = chi2_contingency(contingency)

        # Cramér's V
        n = contingency.sum().sum()
        r, c = contingency.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1)))

        return cramers_v

    def compare_with_multiple_externals(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_columns: List[str]
    ) -> pd.DataFrame:
        """
        Calculate Cramér's V for multiple external labels

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_columns: List of external label column names

        Returns:
            DataFrame with Cramér's V and interpretation for each external
        """
        results = []

        for ext_col in external_columns:
            if ext_col not in df.columns:
                logger.warning(f"  ⚠️  External column '{ext_col}' not found")
                continue

            # Calculate Cramér's V
            cramers_v = self.calculate_cramers_v(df, cluster_column, ext_col)

            # Interpret
            interpretation = self._interpret_cramers_v(cramers_v)

            results.append({
                'external_label': ext_col,
                'cramers_v': cramers_v,
                'interpretation': interpretation
            })

        results_df = pd.DataFrame(results)

        # Sort by Cramér's V (descending)
        results_df = results_df.sort_values('cramers_v', ascending=False)

        return results_df

    def _interpret_cramers_v(self, v: float) -> str:
        """
        Interpret Cramér's V value

        Based on Cohen (1988) for social sciences

        Args:
            v: Cramér's V value

        Returns:
            Interpretation string
        """
        if v < 0.1:
            return "Very Low - Independent"
        elif v < 0.2:
            return "Low - Weak Association"
        elif v < 0.4:
            return "Moderate - Orthogonal"
        elif v < 0.6:
            return "Strong - Partial Overlap"
        elif v < 0.8:
            return "Very Strong - High Association"
        else:
            return "Extremely Strong - Near-Perfect"

    # =========================================================================
    # CHI²-TEST
    # =========================================================================

    def perform_chi_square_test(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform Chi²-Test for independence

        H0: Clusters and external labels are independent
        H1: There is an association between clusters and external labels

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_column: Name of external label column
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        # Remove missing values
        df_clean = df[[cluster_column, external_column]].dropna()

        if len(df_clean) == 0:
            return {
                'chi2_statistic': np.nan,
                'p_value': np.nan,
                'dof': np.nan,
                'significant': False,
                'interpretation': 'No valid data',
                'warning': 'All values are missing'
            }

        # Create contingency table
        contingency = pd.crosstab(df_clean[cluster_column], df_clean[external_column])

        # Chi²-Test
        chi2, p, dof, expected = chi2_contingency(contingency)

        # Check assumptions (expected frequency ≥ 5)
        min_expected = expected.min()
        pct_below_5 = (expected < 5).sum() / expected.size

        warning = None
        if min_expected < 5:
            if pct_below_5 > 0.2:
                warning = f"Chi²-Test unreliable: {pct_below_5:.1%} of cells have expected < 5"
            else:
                warning = f"Warning: Minimum expected frequency = {min_expected:.1f} (should be ≥ 5)"

        # Interpret
        significant = p < alpha

        if significant:
            interpretation = f"Clusters and {external_column} are significantly associated (p={p:.4f})"
        else:
            interpretation = f"No significant association between clusters and {external_column} (p={p:.4f})"

        return {
            'chi2_statistic': chi2,
            'p_value': p,
            'dof': dof,
            'significant': significant,
            'alpha': alpha,
            'interpretation': interpretation,
            'min_expected_frequency': min_expected,
            'pct_cells_below_5': pct_below_5,
            'warning': warning
        }

    # =========================================================================
    # CONTINGENCY TABLES
    # =========================================================================

    def create_contingency_table(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str,
        normalize: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create contingency table (cross-tabulation)

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_column: Name of external label column
            normalize: Normalization mode
                       - None: Absolute counts
                       - 'index': Row percentages (per cluster)
                       - 'columns': Column percentages (per external label)
                       - 'all': Overall percentages

        Returns:
            Contingency table as DataFrame
        """
        # Remove missing values
        df_clean = df[[cluster_column, external_column]].dropna()

        if len(df_clean) == 0:
            logger.warning(f"  ⚠️  No valid data for contingency table")
            return pd.DataFrame()

        # Create contingency table
        if normalize is None:
            contingency = pd.crosstab(
                df_clean[cluster_column],
                df_clean[external_column],
                margins=True,
                margins_name='Total'
            )
        else:
            contingency = pd.crosstab(
                df_clean[cluster_column],
                df_clean[external_column],
                normalize=normalize,
                margins=True,
                margins_name='Total'
            )
            # Format percentages if normalized
            contingency = contingency * 100  # Convert to percentages

        return contingency

    # =========================================================================
    # DISTRIBUTION ANALYSIS
    # =========================================================================

    def analyze_cluster_distribution_in_external(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str
    ) -> pd.DataFrame:
        """
        Analyze cluster distribution within each external label

        For each external label: show percentage of each cluster

        Example:
        - GICS 'Manufacturing': 40% Cluster 0, 35% Cluster 1, 25% Cluster 2
        - GICS 'Finance': 20% Cluster 0, 50% Cluster 1, 30% Cluster 2

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_column: Name of external label column

        Returns:
            DataFrame with cluster percentages per external label
        """
        # Remove missing values
        df_clean = df[[cluster_column, external_column]].dropna()

        if len(df_clean) == 0:
            logger.warning(f"  ⚠️  No valid data for distribution analysis")
            return pd.DataFrame()

        # Group by external label and cluster, count occurrences
        distribution = df_clean.groupby([external_column, cluster_column]).size().reset_index(name='count')

        # Calculate total per external label
        totals = distribution.groupby(external_column)['count'].sum().reset_index(name='total')

        # Merge and calculate percentage
        distribution = distribution.merge(totals, on=external_column)
        distribution['percentage'] = distribution['count'] / distribution['total'] * 100

        # Pivot to wide format
        distribution_pivot = distribution.pivot(
            index=external_column,
            columns=cluster_column,
            values='percentage'
        ).fillna(0)

        # Add total count column
        distribution_pivot = distribution_pivot.merge(
            totals.set_index(external_column),
            left_index=True,
            right_index=True
        )

        return distribution_pivot

    # =========================================================================
    # OUTLIER IDENTIFICATION
    # =========================================================================

    def identify_external_outliers(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_column: str,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Identify unusual cluster-external combinations

        Finds combinations that occur less than threshold percentage.

        Example outliers:
        - Cluster 0 (High Performers) in GICS 'Construction' (only 5%)
        - Cluster 2 (Low Performers) in Size 'Large' (only 3%)

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_column: Name of external label column
            threshold: Percentage threshold (0-1)

        Returns:
            DataFrame with outlier combinations
        """
        # Remove missing values
        df_clean = df[[cluster_column, external_column]].dropna()

        if len(df_clean) == 0:
            logger.warning(f"  ⚠️  No valid data for outlier identification")
            return pd.DataFrame()

        # Create contingency table (counts)
        contingency = pd.crosstab(df_clean[cluster_column], df_clean[external_column])

        # Calculate percentages (normalize by total)
        total = contingency.sum().sum()
        percentages = contingency / total

        # Find outliers (below threshold)
        outliers = []

        for cluster in contingency.index:
            for external in contingency.columns:
                count = contingency.loc[cluster, external]
                pct = percentages.loc[cluster, external]

                if pct < threshold and count > 0:
                    outliers.append({
                        'cluster': cluster,
                        'external_label': external,
                        'count': count,
                        'percentage': pct * 100,
                        'total': total,
                        'rarity': 'rare' if pct < threshold / 2 else 'uncommon'
                    })

        outliers_df = pd.DataFrame(outliers)

        # Sort by percentage (ascending)
        if len(outliers_df) > 0:
            outliers_df = outliers_df.sort_values('percentage')

        return outliers_df

    # =========================================================================
    # VALIDATION REPORT
    # =========================================================================

    def generate_validation_report(
        self,
        df: pd.DataFrame,
        cluster_column: str,
        external_columns: List[str],
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive validation report

        Args:
            df: DataFrame with cluster and external columns
            cluster_column: Name of cluster column
            external_columns: List of external label column names
            save_path: Optional path to save report

        Returns:
            Dictionary with all validation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔬 EXTERNAL VALIDATION REPORT")
        logger.info("=" * 80)
        logger.info(f"  Cluster Column: {cluster_column}")
        logger.info(f"  External Labels: {external_columns}")
        logger.info(f"  Total Companies: {len(df)}\n")

        report = {
            'cluster_column': cluster_column,
            'external_columns': external_columns,
            'n_companies': len(df),
            'cramers_v': {},
            'chi_square': {},
            'contingency_tables': {},
            'distribution_analysis': {},
            'outliers': {}
        }

        for ext_col in external_columns:
            if ext_col not in df.columns:
                logger.warning(f"  ⚠️  External column '{ext_col}' not found, skipping")
                continue

            logger.info(f"\n📊 Analyzing: {ext_col}")

            # 1. Cramér's V
            cramers_v = self.calculate_cramers_v(df, cluster_column, ext_col)
            interpretation = self._interpret_cramers_v(cramers_v)
            report['cramers_v'][ext_col] = {
                'value': cramers_v,
                'interpretation': interpretation
            }
            logger.info(f"  Cramér's V: {cramers_v:.3f} ({interpretation})")

            # 2. Chi²-Test
            chi2_result = self.perform_chi_square_test(df, cluster_column, ext_col)
            report['chi_square'][ext_col] = chi2_result
            logger.info(f"  Chi² p-value: {chi2_result['p_value']:.4f} "
                       f"({'Significant' if chi2_result['significant'] else 'Not Significant'})")

            if chi2_result['warning']:
                logger.warning(f"  ⚠️  {chi2_result['warning']}")

            # 3. Contingency Table
            contingency = self.create_contingency_table(df, cluster_column, ext_col)
            report['contingency_tables'][ext_col] = contingency
            logger.info(f"  Contingency Table: {contingency.shape[0]-1} x {contingency.shape[1]-1}")

            # 4. Distribution Analysis
            distribution = self.analyze_cluster_distribution_in_external(df, cluster_column, ext_col)
            report['distribution_analysis'][ext_col] = distribution
            logger.info(f"  Distribution: {len(distribution)} external categories")

            # 5. Outliers
            outliers = self.identify_external_outliers(df, cluster_column, ext_col, threshold=0.05)
            report['outliers'][ext_col] = outliers
            logger.info(f"  Outliers (<5%): {len(outliers)} combinations")

        # Overall summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        if report['cramers_v']:
            avg_cramers = np.mean([v['value'] for v in report['cramers_v'].values()])
            logger.info(f"  Average Cramér's V: {avg_cramers:.3f}")

            # Find strongest and weakest associations
            cramers_sorted = sorted(
                report['cramers_v'].items(),
                key=lambda x: x[1]['value'],
                reverse=True
            )
            strongest = cramers_sorted[0]
            weakest = cramers_sorted[-1]

            logger.info(f"  Strongest Association: {strongest[0]} (V={strongest[1]['value']:.3f})")
            logger.info(f"  Weakest Association: {weakest[0]} (V={weakest[1]['value']:.3f})")

        logger.info("=" * 80 + "\n")

        # Save report if requested
        if save_path:
            self._save_report(report, save_path)

        return report

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _save_report(self, report: Dict, save_path: Path):
        """
        Save validation report to files

        Args:
            report: Validation report dictionary
            save_path: Directory to save report
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Cramér's V summary
        cramers_data = []
        for ext_col, data in report['cramers_v'].items():
            cramers_data.append({
                'external_label': ext_col,
                'cramers_v': data['value'],
                'interpretation': data['interpretation']
            })

        if cramers_data:
            cramers_df = pd.DataFrame(cramers_data)
            cramers_df.to_csv(save_path / 'cramers_v_summary.csv', index=False)

        # 2. Chi² results
        chi2_data = []
        for ext_col, data in report['chi_square'].items():
            chi2_data.append({
                'external_label': ext_col,
                'chi2_statistic': data['chi2_statistic'],
                'p_value': data['p_value'],
                'dof': data['dof'],
                'significant': data['significant'],
                'warning': data.get('warning', '')
            })

        if chi2_data:
            chi2_df = pd.DataFrame(chi2_data)
            chi2_df.to_csv(save_path / 'chi_square_tests.csv', index=False)

        # 3. Contingency tables
        for ext_col, contingency in report['contingency_tables'].items():
            if not contingency.empty:
                filename = f'contingency_{ext_col}.csv'
                contingency.to_csv(save_path / filename)

        # 4. Distribution analysis
        for ext_col, distribution in report['distribution_analysis'].items():
            if not distribution.empty:
                filename = f'distribution_{ext_col}.csv'
                distribution.to_csv(save_path / filename)

        # 5. Outliers
        for ext_col, outliers in report['outliers'].items():
            if not outliers.empty:
                filename = f'outliers_{ext_col}.csv'
                outliers.to_csv(save_path / filename, index=False)

        logger.info(f"  ✓ Report saved to: {save_path}/")


if __name__ == "__main__":
    # Test ExternalValidation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("EXTERNAL VALIDATION TEST")
    print("=" * 80)

    # Mock data
    np.random.seed(42)
    n = 200

    # Create correlated data (clusters somewhat related to external labels)
    df = pd.DataFrame({
        'gvkey': range(n),
        'company_name': [f'Company {i}' for i in range(n)],
        'cluster': np.random.randint(0, 4, n)
    })

    # Add external labels with some correlation to clusters
    # GICS: moderate correlation
    gics_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    df['gics_code'] = df['cluster'].map(gics_mapping)
    # Add some noise (30% different)
    noise_idx = np.random.choice(n, int(n * 0.3), replace=False)
    df.loc[noise_idx, 'gics_code'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], len(noise_idx))

    # Size: weak correlation
    df['size_category'] = np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.3, 0.5, 0.2])

    # Country: very weak correlation (nearly independent)
    df['country'] = np.random.choice(['DE', 'FR', 'UK', 'IT'], n, p=[0.4, 0.3, 0.2, 0.1])

    print(f"\nMock data created:")
    print(f"  Companies: {n}")
    print(f"  Clusters: {df['cluster'].nunique()}")
    print(f"  External Labels: GICS, Size, Country")

    # Test ExternalValidation
    print("\n" + "-" * 80)
    print("Testing ExternalValidation")
    print("-" * 80)

    try:
        validator = ExternalValidation()

        # 1. Single Cramér's V
        print("\n1. Testing single Cramér's V calculation:")
        cramers_v = validator.calculate_cramers_v(df, 'cluster', 'gics_code')
        print(f"   Cramér's V (Cluster vs GICS): {cramers_v:.3f}")

        # 2. Multiple externals
        print("\n2. Testing multiple external labels:")
        comparison_df = validator.compare_with_multiple_externals(
            df, 'cluster', ['gics_code', 'size_category', 'country']
        )
        print(comparison_df.to_string(index=False))

        # 3. Chi²-Test
        print("\n3. Testing Chi²-Test:")
        chi2_result = validator.perform_chi_square_test(df, 'cluster', 'gics_code')
        print(f"   Chi² = {chi2_result['chi2_statistic']:.2f}")
        print(f"   p-value = {chi2_result['p_value']:.4f}")
        print(f"   Significant: {chi2_result['significant']}")

        # 4. Contingency table
        print("\n4. Testing contingency table:")
        contingency = validator.create_contingency_table(df, 'cluster', 'gics_code')
        print(contingency)

        # 5. Distribution analysis
        print("\n5. Testing distribution analysis:")
        distribution = validator.analyze_cluster_distribution_in_external(
            df, 'cluster', 'gics_code'
        )
        print(distribution.round(1))

        # 6. Outliers
        print("\n6. Testing outlier identification:")
        outliers = validator.identify_external_outliers(
            df, 'cluster', 'gics_code', threshold=0.05
        )
        print(f"   Outliers found: {len(outliers)}")
        if len(outliers) > 0:
            print(outliers.head().to_string(index=False))

        # 7. Full validation report
        print("\n7. Testing full validation report:")
        report = validator.generate_validation_report(
            df, 'cluster', ['gics_code', 'size_category', 'country']
        )

        print("\n✓ ExternalValidation test successful!")
        print(f"\nReport keys: {list(report.keys())}")
        print(f"Cramér's V results: {len(report['cramers_v'])}")
        print(f"Chi² results: {len(report['chi_square'])}")

    except Exception as e:
        print(f"\n❌ ExternalValidation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
