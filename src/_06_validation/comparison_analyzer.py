"""
Comparison Analyzer - Analyze and report on clustering comparisons

Provides tools for:
- Orchestrating multi-algorithm comparisons
- Generating comprehensive summaries
- Reporting and visualization support
- Statistical interpretations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from pathlib import Path

# Import the metrics calculator
from .comparison_metrics import ComparisonMetrics

logger = logging.getLogger(__name__)


class ComparisonAnalyzer:
    """
    Analyze and report on clustering algorithm comparisons

    Workflow:
    1. Calculate pairwise ARI between all algorithm pairs
    2. Create confusion matrices for each pair
    3. Identify consensus clusters (high agreement)
    4. Find disagreement cases (low agreement)
    5. Generate comprehensive comparison summary
    """

    def __init__(self):
        """Initialize Comparison Analyzer"""
        self.metrics = ComparisonMetrics()
        logger.info("✓ ComparisonAnalyzer initialized")

    # =========================================================================
    # MAIN COMPARISON METHOD
    # =========================================================================

    def compare_multiple_algorithms(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str = 'cluster',
        consensus_threshold: float = 0.8,
        disagreement_top_n: int = 20
    ) -> Dict:
        """
        Compare clustering results across multiple algorithms

        Args:
            results_dict: Dict with format {'kmeans': df, 'hierarchical': df, ...}
                         Each DataFrame must have 'gvkey' and cluster_column
            cluster_column: Name of cluster column
            consensus_threshold: Threshold for consensus (0-1)
            disagreement_top_n: Number of top disagreement cases to return

        Returns:
            Dictionary with:
            - ari_matrix: Pairwise ARI matrix
            - confusion_matrices: Dict of confusion matrices for each pair
            - consensus_clusters: Companies with high agreement
            - disagreement_cases: Companies with low agreement
            - summary: Summary statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔬 ALGORITHM COMPARISON")
        logger.info("=" * 80)
        logger.info(f"  Algorithms: {list(results_dict.keys())}")
        logger.info(f"  Consensus Threshold: {consensus_threshold:.0%}")
        logger.info(f"  Disagreement Top N: {disagreement_top_n}\n")

        # Validate input
        self._validate_input(results_dict, cluster_column)

        # 1. Calculate ARI matrix
        logger.info("📊 Step 1/5: Calculating ARI Matrix")
        ari_matrix = self.metrics.calculate_ari_matrix(results_dict, cluster_column)
        logger.info(f"  ✓ ARI matrix computed ({len(results_dict)} x {len(results_dict)})")

        # 2. Create confusion matrices for all pairs
        logger.info("\n📊 Step 2/5: Creating Confusion Matrices")
        confusion_matrices = self.metrics.create_all_confusion_matrices(
            results_dict, cluster_column
        )
        logger.info(f"  ✓ {len(confusion_matrices)} confusion matrices created")

        # 3. Identify consensus clusters
        logger.info("\n📊 Step 3/5: Identifying Consensus Clusters")
        consensus_clusters = self.metrics.identify_consensus_clusters(
            results_dict, cluster_column, consensus_threshold
        )
        logger.info(f"  ✓ {len(consensus_clusters)} consensus clusters found")

        # 4. Find disagreement cases
        logger.info("\n📊 Step 4/5: Finding Disagreement Cases")
        disagreement_cases = self.metrics.find_disagreement_cases(
            results_dict, cluster_column, disagreement_top_n
        )
        logger.info(f"  ✓ {len(disagreement_cases)} disagreement cases identified")

        # 5. Generate summary
        logger.info("\n📊 Step 5/5: Generating Summary")
        summary = self.generate_comparison_summary(results_dict, ari_matrix)
        logger.info(f"  ✓ Summary generated")

        # Print summary
        self._print_summary(ari_matrix, summary, consensus_clusters, disagreement_cases)

        return {
            'ari_matrix': ari_matrix,
            'confusion_matrices': confusion_matrices,
            'consensus_clusters': consensus_clusters,
            'disagreement_cases': disagreement_cases,
            'summary': summary
        }

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def generate_comparison_summary(
        self,
        results_dict: Dict[str, pd.DataFrame],
        ari_matrix: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive comparison summary

        Args:
            results_dict: Dict with clustering results
            ari_matrix: ARI matrix

        Returns:
            Dictionary with summary statistics
        """
        # Extract upper triangle of ARI matrix (exclude diagonal)
        n = len(ari_matrix)
        upper_triangle = []
        algorithms = list(ari_matrix.index)

        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append({
                    'algo1': algorithms[i],
                    'algo2': algorithms[j],
                    'ari': ari_matrix.iloc[i, j]
                })

        upper_triangle_df = pd.DataFrame(upper_triangle)

        # Calculate statistics
        average_ari = upper_triangle_df['ari'].mean()
        std_ari = upper_triangle_df['ari'].std()

        # Find highest and lowest congruence pairs
        highest_pair = upper_triangle_df.loc[upper_triangle_df['ari'].idxmax()]
        lowest_pair = upper_triangle_df.loc[upper_triangle_df['ari'].idxmin()]

        # Cluster counts per algorithm
        cluster_counts = {
            algo: df['cluster'].nunique()
            for algo, df in results_dict.items()
        }

        # Company counts per algorithm
        company_counts = {
            algo: len(df)
            for algo, df in results_dict.items()
        }

        # Robustness interpretation
        if average_ari >= 0.75:
            robustness = "High - Robust clustering"
        elif average_ari >= 0.50:
            robustness = "Moderate - Partially robust"
        elif average_ari >= 0.25:
            robustness = "Low - Algorithm-dependent"
        else:
            robustness = "Very Low - Highly unstable"

        summary = {
            'n_algorithms': len(results_dict),
            'average_ari': average_ari,
            'std_ari': std_ari,
            'min_ari': upper_triangle_df['ari'].min(),
            'max_ari': upper_triangle_df['ari'].max(),
            'highest_congruence': {
                'algorithms': (highest_pair['algo1'], highest_pair['algo2']),
                'ari': highest_pair['ari']
            },
            'lowest_congruence': {
                'algorithms': (lowest_pair['algo1'], lowest_pair['algo2']),
                'ari': lowest_pair['ari']
            },
            'cluster_counts': cluster_counts,
            'company_counts': company_counts,
            'robustness': robustness,
            'pairwise_comparisons': upper_triangle_df
        }

        return summary

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _validate_input(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str
    ):
        """
        Validate input data

        Args:
            results_dict: Dict with clustering results
            cluster_column: Name of cluster column

        Raises:
            ValueError: If validation fails
        """
        if len(results_dict) < 2:
            raise ValueError("Need at least 2 algorithms to compare")

        for algo, df in results_dict.items():
            if 'gvkey' not in df.columns:
                raise ValueError(f"Algorithm '{algo}': Missing 'gvkey' column")

            if cluster_column not in df.columns:
                raise ValueError(f"Algorithm '{algo}': Missing '{cluster_column}' column")

            if df['gvkey'].duplicated().any():
                raise ValueError(f"Algorithm '{algo}': Duplicate gvkey values found")

    def _print_summary(
        self,
        ari_matrix: pd.DataFrame,
        summary: Dict,
        consensus_clusters: pd.DataFrame,
        disagreement_cases: pd.DataFrame
    ):
        """Print comparison summary"""
        logger.info("\n" + "=" * 80)
        logger.info("ALGORITHM COMPARISON COMPLETE")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("✓ Algorithm Comparison Complete")
        print("=" * 80)

        print("\n📊 ARI MATRIX")
        print(ari_matrix.round(3).to_string())

        print(f"\n📊 SUMMARY")
        print(f"   Average ARI: {summary['average_ari']:.3f}")
        print(f"   Std ARI: {summary['std_ari']:.3f}")
        print(f"   Range: [{summary['min_ari']:.3f}, {summary['max_ari']:.3f}]")
        print(f"   Robustness: {summary['robustness']}")

        print(f"\n📊 HIGHEST CONGRUENCE")
        algo1, algo2 = summary['highest_congruence']['algorithms']
        ari = summary['highest_congruence']['ari']
        print(f"   {algo1} ↔ {algo2}: {ari:.3f}")

        print(f"\n📊 LOWEST CONGRUENCE")
        algo1, algo2 = summary['lowest_congruence']['algorithms']
        ari = summary['lowest_congruence']['ari']
        print(f"   {algo1} ↔ {algo2}: {ari:.3f}")

        print(f"\n📊 CONSENSUS CLUSTERS")
        print(f"   Companies with consensus: {len(consensus_clusters)}")
        if len(consensus_clusters) > 0:
            avg_agreement = consensus_clusters['agreement_pct'].mean()
            print(f"   Average agreement: {avg_agreement:.1%}")

        print(f"\n📊 DISAGREEMENT CASES")
        print(f"   Top disagreement cases: {len(disagreement_cases)}")
        if len(disagreement_cases) > 0:
            avg_disagreement = disagreement_cases['disagreement_score'].mean()
            print(f"   Average disagreement: {avg_disagreement:.1%}")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test ComparisonAnalyzer
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("COMPARISON ANALYZER TEST")
    print("=" * 80)

    # Mock data
    np.random.seed(42)
    n = 100

    # Create correlated clusterings
    base_clusters = np.random.randint(0, 4, n)

    # K-Means: similar to base
    kmeans_clusters = base_clusters.copy()
    kmeans_clusters[np.random.choice(n, 10, replace=False)] = np.random.randint(0, 4, 10)

    # Hierarchical: more similar to base
    hierarchical_clusters = base_clusters.copy()
    hierarchical_clusters[np.random.choice(n, 5, replace=False)] = np.random.randint(0, 4, 5)

    # DBSCAN: less similar (includes noise -1)
    dbscan_clusters = np.random.randint(-1, 4, n)

    results_dict = {
        'kmeans': pd.DataFrame({
            'gvkey': range(n),
            'company_name': [f'Company {i}' for i in range(n)],
            'cluster': kmeans_clusters
        }),
        'hierarchical': pd.DataFrame({
            'gvkey': range(n),
            'company_name': [f'Company {i}' for i in range(n)],
            'cluster': hierarchical_clusters
        }),
        'dbscan': pd.DataFrame({
            'gvkey': range(n),
            'company_name': [f'Company {i}' for i in range(n)],
            'cluster': dbscan_clusters
        })
    }

    print(f"\nMock data created:")
    print(f"  Algorithms: {list(results_dict.keys())}")
    print(f"  Companies: {n}")

    # Test ComparisonAnalyzer
    print("\n" + "-" * 80)
    print("Testing ComparisonAnalyzer")
    print("-" * 80)

    try:
        analyzer = ComparisonAnalyzer()
        results = analyzer.compare_multiple_algorithms(results_dict)

        print("\n✓ ComparisonAnalyzer test successful!")
        print(f"\nResults keys: {list(results.keys())}")
        print(f"\nARI Matrix:\n{results['ari_matrix'].round(3)}")
        print(f"\nConsensus Clusters: {len(results['consensus_clusters'])}")
        print(f"Disagreement Cases: {len(results['disagreement_cases'])}")
        print(f"\nAverage ARI: {results['summary']['average_ari']:.3f}")
        print(f"Robustness: {results['summary']['robustness']}")

    except Exception as e:
        print(f"\n❌ ComparisonAnalyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
