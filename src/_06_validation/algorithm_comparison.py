"""
Algorithm Comparison - Compare clustering results across different algorithms

Provides tools for:
- Calculating ARI matrices between algorithm pairs
- Creating confusion matrices
- Identifying consensus clusters (high agreement)
- Finding disagreement cases (low agreement)
- Generating comparison summaries
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from collections import Counter

logger = logging.getLogger(__name__)


class AlgorithmComparison:
    """
    Compare clustering results across multiple algorithms

    Workflow:
    1. Calculate pairwise ARI between all algorithm pairs
    2. Create confusion matrices for each pair
    3. Identify consensus clusters (high agreement)
    4. Find disagreement cases (low agreement)
    5. Generate comprehensive comparison summary
    """

    def __init__(self):
        """Initialize Algorithm Comparison"""
        logger.info("✓ AlgorithmComparison initialized")

    # =========================================================================
    # RUN AND COMPARE ALGORITHMS
    # =========================================================================

    def run_and_compare_algorithms(
        self,
        df: pd.DataFrame,
        features: List[str],
        original_labels: np.ndarray,
        n_clusters: int
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run alternative clustering algorithms and compare them

        Args:
            df: DataFrame with features
            features: List of feature names to use for clustering
            original_labels: Original cluster labels from primary algorithm
            n_clusters: Number of clusters to use

        Returns:
            Tuple of (comparison_results, df_with_alt_clusters)
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        import numpy as np

        # Check if features exist in df
        available_features = [f for f in features if f in df.columns]
        if len(available_features) == 0:
            raise ValueError(f"None of the features {features} found in DataFrame columns: {list(df.columns)[:10]}")

        # Use only available features
        if len(available_features) < len(features):
            logger.warning(f"  ⚠️  Only {len(available_features)}/{len(features)} features available")

        X = df[available_features].values

        # Handle NaN and inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run alternative algorithms
        results_dict = {}

        # 1. K-Means (alternative)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        results_dict['kmeans_alt'] = pd.DataFrame({
            'gvkey': df['gvkey'].values if 'gvkey' in df.columns else range(len(df)),
            'cluster': kmeans_labels
        })

        # 2. Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        results_dict['hierarchical'] = pd.DataFrame({
            'gvkey': df['gvkey'].values if 'gvkey' in df.columns else range(len(df)),
            'cluster': hierarchical_labels
        })

        # 3. DBSCAN (auto determine eps)
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(X_scaled)
        distances, _ = neighbors.kneighbors(X_scaled)
        eps = np.median(distances[:, -1])

        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        results_dict['dbscan'] = pd.DataFrame({
            'gvkey': df['gvkey'].values if 'gvkey' in df.columns else range(len(df)),
            'cluster': dbscan_labels
        })

        # Compare algorithms
        comparison_results = self.compare_multiple_algorithms(
            results_dict=results_dict,
            cluster_column='cluster'
        )

        # Create DataFrame with alternative cluster assignments
        df_alt = df[['gvkey']].copy() if 'gvkey' in df.columns else pd.DataFrame({'gvkey': range(len(df))})
        df_alt['cluster_kmeans_alt'] = kmeans_labels
        df_alt['cluster_hierarchical'] = hierarchical_labels
        df_alt['cluster_dbscan'] = dbscan_labels

        return comparison_results, df_alt

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
        ari_matrix = self.calculate_ari_matrix(results_dict, cluster_column)
        logger.info(f"  ✓ ARI matrix computed ({len(results_dict)} x {len(results_dict)})")

        # 2. Create confusion matrices for all pairs
        logger.info("\n📊 Step 2/5: Creating Confusion Matrices")
        confusion_matrices = self._create_all_confusion_matrices(
            results_dict, cluster_column
        )
        logger.info(f"  ✓ {len(confusion_matrices)} confusion matrices created")

        # 3. Identify consensus clusters
        logger.info("\n📊 Step 3/5: Identifying Consensus Clusters")
        consensus_clusters = self.identify_consensus_clusters(
            results_dict, cluster_column, consensus_threshold
        )
        logger.info(f"  ✓ {len(consensus_clusters)} consensus clusters found")

        # 4. Find disagreement cases
        logger.info("\n📊 Step 4/5: Finding Disagreement Cases")
        disagreement_cases = self.find_disagreement_cases(
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
    # ARI CALCULATION
    # =========================================================================

    def calculate_ari_matrix(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str = 'cluster'
    ) -> pd.DataFrame:
        """
        Calculate pairwise ARI matrix between all algorithms

        Args:
            results_dict: Dict with clustering results
            cluster_column: Name of cluster column

        Returns:
            Symmetric ARI matrix (DataFrame)
        """
        algorithms = list(results_dict.keys())
        n_algos = len(algorithms)

        # Initialize matrix
        ari_matrix = np.zeros((n_algos, n_algos))

        # Calculate pairwise ARI
        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i == j:
                    ari_matrix[i, j] = 1.0
                elif i < j:
                    # Merge on gvkey to align
                    df_merged = results_dict[algo1][['gvkey', cluster_column]].merge(
                        results_dict[algo2][['gvkey', cluster_column]],
                        on='gvkey',
                        suffixes=('_1', '_2')
                    )

                    # Calculate ARI
                    ari = adjusted_rand_score(
                        df_merged[f'{cluster_column}_1'],
                        df_merged[f'{cluster_column}_2']
                    )
                    ari_matrix[i, j] = ari
                    ari_matrix[j, i] = ari  # Symmetric

        # Convert to DataFrame
        ari_df = pd.DataFrame(
            ari_matrix,
            index=algorithms,
            columns=algorithms
        )

        return ari_df

    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================

    def create_confusion_matrix(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        cluster_col1: str = 'cluster',
        cluster_col2: str = 'cluster',
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Create confusion matrix between two clustering results

        Args:
            df1: First clustering result
            df2: Second clustering result
            cluster_col1: Cluster column in df1
            cluster_col2: Cluster column in df2
            normalize: If True, show percentages instead of counts

        Returns:
            Confusion matrix as DataFrame
        """
        # Merge on gvkey
        df1_subset = df1[['gvkey', cluster_col1]].rename(columns={cluster_col1: 'cluster_1'})
        df2_subset = df2[['gvkey', cluster_col2]].rename(columns={cluster_col2: 'cluster_2'})

        df_merged = df1_subset.merge(df2_subset, on='gvkey')

        # Get unique cluster labels (including DBSCAN noise -1)
        clusters1 = sorted(df_merged['cluster_1'].unique())
        clusters2 = sorted(df_merged['cluster_2'].unique())

        # Create confusion matrix with explicit labels
        conf_matrix = confusion_matrix(
            df_merged['cluster_1'],
            df_merged['cluster_2'],
            labels=clusters1  # Use clusters1 as row labels
        )

        # Ensure matrix dimensions match labels
        # confusion_matrix returns shape (len(labels), n_unique_in_y_true)
        # We need to handle the case where clusters2 has different labels
        if conf_matrix.shape[1] != len(clusters2):
            # Rebuild with both label sets
            all_labels = sorted(set(clusters1) | set(clusters2))
            conf_matrix = confusion_matrix(
                df_merged['cluster_1'],
                df_merged['cluster_2'],
                labels=all_labels
            )
            clusters1 = all_labels
            clusters2 = all_labels

        # Normalize if requested
        if normalize:
            conf_matrix = conf_matrix.astype(float)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            conf_matrix = conf_matrix / row_sums * 100

        # Convert to DataFrame
        conf_df = pd.DataFrame(
            conf_matrix,
            index=[f'C{c}' for c in clusters1],
            columns=[f'C{c}' for c in clusters2]
        )

        return conf_df

    def _create_all_confusion_matrices(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Create confusion matrices for all algorithm pairs

        Args:
            results_dict: Dict with clustering results
            cluster_column: Name of cluster column

        Returns:
            Dict mapping (algo1, algo2) -> confusion_matrix
        """
        algorithms = list(results_dict.keys())
        confusion_matrices = {}

        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i < j:  # Only upper triangle
                    conf_matrix = self.create_confusion_matrix(
                        results_dict[algo1],
                        results_dict[algo2],
                        cluster_column,
                        cluster_column
                    )
                    confusion_matrices[(algo1, algo2)] = conf_matrix

        return confusion_matrices

    # =========================================================================
    # CONSENSUS & DISAGREEMENT ANALYSIS
    # =========================================================================

    def identify_consensus_clusters(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str = 'cluster',
        threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Identify companies with high agreement across algorithms

        A company has consensus if ≥threshold of algorithms assign it to
        the "same" cluster (using mode).

        Args:
            results_dict: Dict with clustering results
            cluster_column: Name of cluster column
            threshold: Agreement threshold (0-1)

        Returns:
            DataFrame with gvkey, consensus_cluster, agreement_pct, algorithms
        """
        # Merge all results on gvkey
        df_merged = None
        for algo, df in results_dict.items():
            df_algo = df[['gvkey', cluster_column]].rename(
                columns={cluster_column: f'cluster_{algo}'}
            )
            if df_merged is None:
                df_merged = df_algo
            else:
                df_merged = df_merged.merge(df_algo, on='gvkey')

        # Calculate consensus for each company
        consensus_data = []
        cluster_cols = [f'cluster_{algo}' for algo in results_dict.keys()]
        n_algorithms = len(cluster_cols)

        for _, row in df_merged.iterrows():
            # Get cluster assignments
            clusters = [row[col] for col in cluster_cols]

            # Find mode (most common cluster)
            cluster_counts = Counter(clusters)
            consensus_cluster, count = cluster_counts.most_common(1)[0]

            # Agreement percentage
            agreement_pct = count / n_algorithms

            # Only keep if above threshold
            if agreement_pct >= threshold:
                consensus_data.append({
                    'gvkey': row['gvkey'],
                    'consensus_cluster': consensus_cluster,
                    'agreement_pct': agreement_pct,
                    'n_algorithms_agree': count,
                    'n_algorithms_total': n_algorithms,
                    'algorithms': {algo: row[f'cluster_{algo}'] for algo in results_dict.keys()}
                })

        consensus_df = pd.DataFrame(consensus_data)

        return consensus_df

    def find_disagreement_cases(
        self,
        results_dict: Dict[str, pd.DataFrame],
        cluster_column: str = 'cluster',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Find companies with maximum disagreement across algorithms

        Disagreement = all (or most) algorithms assign to different clusters

        Args:
            results_dict: Dict with clustering results
            cluster_column: Name of cluster column
            top_n: Number of top disagreement cases to return

        Returns:
            DataFrame with gvkey, disagreement_score, cluster assignments
        """
        # Merge all results on gvkey
        df_merged = None
        for algo, df in results_dict.items():
            df_algo = df[['gvkey', cluster_column]].copy()

            # Add company name if available
            if 'company_name' in df.columns:
                df_algo['company_name'] = df['company_name']
            elif 'conm' in df.columns:
                df_algo['company_name'] = df['conm']

            df_algo = df_algo.rename(columns={cluster_column: f'cluster_{algo}'})

            if df_merged is None:
                df_merged = df_algo
            else:
                df_merged = df_merged.merge(df_algo, on='gvkey', how='outer')

        # Calculate disagreement for each company
        disagreement_data = []
        cluster_cols = [f'cluster_{algo}' for algo in results_dict.keys()]
        n_algorithms = len(cluster_cols)

        for _, row in df_merged.iterrows():
            # Get cluster assignments
            clusters = [row[col] for col in cluster_cols if pd.notna(row[col])]

            if len(clusters) == 0:
                continue

            # Count unique clusters
            n_unique = len(set(clusters))

            # Disagreement score = unique clusters / total algorithms
            disagreement_score = n_unique / len(clusters)

            disagreement_data.append({
                'gvkey': row['gvkey'],
                'company_name': row.get('company_name', ''),
                'disagreement_score': disagreement_score,
                'n_unique_clusters': n_unique,
                'n_algorithms': len(clusters),
                **{f'cluster_{algo}': row[f'cluster_{algo}'] for algo in results_dict.keys()}
            })

        disagreement_df = pd.DataFrame(disagreement_data)

        # Sort by disagreement score (descending) and take top N
        disagreement_df = disagreement_df.sort_values(
            'disagreement_score', ascending=False
        ).head(top_n)

        return disagreement_df

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
    # Test AlgorithmComparison
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON TEST")
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

    # Test AlgorithmComparison
    print("\n" + "-" * 80)
    print("Testing AlgorithmComparison")
    print("-" * 80)

    try:
        comparator = AlgorithmComparison()
        results = comparator.compare_multiple_algorithms(results_dict)

        print("\n✓ AlgorithmComparison test successful!")
        print(f"\nResults keys: {list(results.keys())}")
        print(f"\nARI Matrix:\n{results['ari_matrix'].round(3)}")
        print(f"\nConsensus Clusters: {len(results['consensus_clusters'])}")
        print(f"Disagreement Cases: {len(results['disagreement_cases'])}")
        print(f"\nAverage ARI: {results['summary']['average_ari']:.3f}")
        print(f"Robustness: {results['summary']['robustness']}")

    except Exception as e:
        print(f"\n❌ AlgorithmComparison test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
