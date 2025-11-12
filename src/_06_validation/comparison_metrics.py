"""
Comparison Metrics - Calculate similarity metrics between clustering results

Provides tools for:
- Calculating ARI (Adjusted Rand Index) between algorithm pairs
- Creating confusion matrices
- Identifying consensus clusters (high agreement)
- Finding disagreement cases (low agreement)
- Statistical comparisons
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from collections import Counter

logger = logging.getLogger(__name__)


class ComparisonMetrics:
    """
    Calculate similarity metrics between clustering results

    Provides methods for:
    - Pairwise ARI calculation
    - Confusion matrix generation
    - Consensus cluster identification
    - Disagreement case detection
    """

    def __init__(self):
        """Initialize Comparison Metrics"""
        logger.info("✓ ComparisonMetrics initialized")

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

    def create_all_confusion_matrices(
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


if __name__ == "__main__":
    # Test ComparisonMetrics
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("COMPARISON METRICS TEST")
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
        })
    }

    print(f"\nMock data created:")
    print(f"  Algorithms: {list(results_dict.keys())}")
    print(f"  Companies: {n}")

    # Test ComparisonMetrics
    print("\n" + "-" * 80)
    print("Testing ComparisonMetrics")
    print("-" * 80)

    try:
        metrics = ComparisonMetrics()

        # Test ARI matrix
        print("\n1. Testing ARI matrix...")
        ari_matrix = metrics.calculate_ari_matrix(results_dict)
        print(f"   ARI Matrix:\n{ari_matrix.round(3)}")

        # Test confusion matrix
        print("\n2. Testing confusion matrix...")
        conf_matrix = metrics.create_confusion_matrix(
            results_dict['kmeans'],
            results_dict['hierarchical']
        )
        print(f"   Confusion Matrix shape: {conf_matrix.shape}")

        # Test consensus clusters
        print("\n3. Testing consensus clusters...")
        consensus = metrics.identify_consensus_clusters(results_dict, threshold=0.7)
        print(f"   Consensus clusters: {len(consensus)}")

        # Test disagreement cases
        print("\n4. Testing disagreement cases...")
        disagreement = metrics.find_disagreement_cases(results_dict, top_n=10)
        print(f"   Disagreement cases: {len(disagreement)}")

        print("\n✓ ComparisonMetrics test successful!")

    except Exception as e:
        print(f"\n❌ ComparisonMetrics test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
