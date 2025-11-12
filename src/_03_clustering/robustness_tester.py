"""
Robustness Testing for Clustering Algorithms
Tests stability across multiple random initializations
"""

import numpy as np
import logging
from typing import Dict, List
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Tests clustering stability across multiple runs"""

    def __init__(self, n_runs: int = 10):
        """
        Args:
            n_runs: Number of runs with different random seeds
        """
        self.n_runs = n_runs
        self.results = []

    def test_kmeans_stability(
        self,
        X: np.ndarray,
        n_clusters: int,
        base_seed: int = 42
    ) -> Dict:
        """
        Tests K-Means stability across multiple initializations.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            base_seed: Base random seed

        Returns:
            Dict with stability metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info("K-MEANS ROBUSTNESS TEST")
        logger.info(f"{'='*80}")
        logger.info(f"Runs: {self.n_runs}")
        logger.info(f"Clusters: {n_clusters}")
        logger.info(f"Samples: {X.shape[0]}\n")

        # Store all labelings
        all_labels = []
        all_inertias = []

        # Run clustering with different seeds
        for i in range(self.n_runs):
            seed = base_seed + i

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=10
            )
            labels = kmeans.fit_predict(X)

            all_labels.append(labels)
            all_inertias.append(kmeans.inertia_)

        # Calculate pairwise ARI between all runs
        ari_scores = []
        nmi_scores = []

        for i in range(self.n_runs):
            for j in range(i+1, self.n_runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])

                ari_scores.append(ari)
                nmi_scores.append(nmi)

        # Statistics
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        min_ari = np.min(ari_scores)
        max_ari = np.max(ari_scores)

        mean_nmi = np.mean(nmi_scores)
        std_nmi = np.std(nmi_scores)

        mean_inertia = np.mean(all_inertias)
        std_inertia = np.std(all_inertias)

        # Log results
        logger.info(f"📊 Stability Metrics:")
        logger.info(f"  Mean ARI:  {mean_ari:.3f} ± {std_ari:.3f} (range: {min_ari:.3f} - {max_ari:.3f})")
        logger.info(f"  Mean NMI:  {mean_nmi:.3f} ± {std_nmi:.3f}")
        logger.info(f"  Inertia:   {mean_inertia:.1f} ± {std_inertia:.1f}")

        # Interpretation
        if mean_ari >= 0.8:
            stability = "Excellent"
        elif mean_ari >= 0.6:
            stability = "Good"
        elif mean_ari >= 0.4:
            stability = "Moderate"
        else:
            stability = "Poor"

        logger.info(f"\n  → Stability Assessment: {stability}")

        if mean_ari < 0.6:
            logger.warning(
                f"  ⚠️  Low stability detected! Consider:\n"
                f"     - Increasing number of initializations (n_init)\n"
                f"     - Trying different k values\n"
                f"     - Checking data quality and preprocessing"
            )

        results = {
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'min_ari': min_ari,
            'max_ari': max_ari,
            'mean_nmi': mean_nmi,
            'std_nmi': std_nmi,
            'mean_inertia': mean_inertia,
            'std_inertia': std_inertia,
            'stability_assessment': stability,
            'all_ari_scores': ari_scores,
            'all_nmi_scores': nmi_scores,
            'all_inertias': all_inertias
        }

        self.results.append(results)

        logger.info(f"{'='*80}\n")

        return results

    def test_algorithm_congruence(
        self,
        labels_dict: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Tests agreement between different clustering algorithms.

        Args:
            labels_dict: Dict mapping algorithm names to their labels

        Returns:
            Dict with congruence metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info("ALGORITHM CONGRUENCE TEST")
        logger.info(f"{'='*80}")
        logger.info(f"Algorithms: {list(labels_dict.keys())}\n")

        # Pairwise ARI between all algorithm pairs
        algorithm_names = list(labels_dict.keys())
        ari_matrix = np.zeros((len(algorithm_names), len(algorithm_names)))

        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names):
                if i == j:
                    ari_matrix[i, j] = 1.0
                else:
                    ari = adjusted_rand_score(
                        labels_dict[alg1],
                        labels_dict[alg2]
                    )
                    ari_matrix[i, j] = ari

        # Log pairwise ARI
        logger.info("📊 Pairwise ARI:")
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names):
                if i < j:
                    ari = ari_matrix[i, j]
                    logger.info(f"  {alg1:15s} vs {alg2:15s}: {ari:.3f}")

        # Average ARI (excluding diagonal)
        mask = ~np.eye(len(algorithm_names), dtype=bool)
        mean_ari = ari_matrix[mask].mean()
        logger.info(f"\n  Mean ARI: {mean_ari:.3f}")

        # Interpretation
        if mean_ari >= 0.7:
            congruence = "High"
        elif mean_ari >= 0.5:
            congruence = "Moderate"
        elif mean_ari >= 0.3:
            congruence = "Low"
        else:
            congruence = "Very Low"

        logger.info(f"  → Congruence Assessment: {congruence}")

        if mean_ari < 0.5:
            logger.warning(
                f"  ⚠️  Low algorithm congruence! This suggests:\n"
                f"     - Algorithms find different structures in the data\n"
                f"     - Data may not have clear cluster structure\n"
                f"     - Consider ensemble methods or alternative distance metrics"
            )

        results = {
            'ari_matrix': ari_matrix,
            'algorithm_names': algorithm_names,
            'mean_ari': mean_ari,
            'congruence_assessment': congruence
        }

        logger.info(f"{'='*80}\n")

        return results


if __name__ == "__main__":
    """Test robustness tester"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("ROBUSTNESS TESTER - DEMONSTRATION")
    print("="*80)

    # Mock data with clear clusters
    np.random.seed(42)
    n_samples = 300

    # 3 well-separated clusters
    centers = [[0, 0], [5, 5], [0, 5]]
    X = []
    for center in centers:
        cluster_data = np.random.randn(n_samples // 3, 2) * 0.5 + center
        X.append(cluster_data)

    X = np.vstack(X)

    print(f"\nMock data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Ground truth: 3 clusters\n")

    # Test K-Means stability
    tester = RobustnessTester(n_runs=10)
    results = tester.test_kmeans_stability(X, n_clusters=3)

    # Test algorithm congruence (simulate different algorithms)
    from sklearn.cluster import AgglomerativeClustering, DBSCAN

    kmeans_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    hierarchical_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    dbscan_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)

    labels_dict = {
        'K-Means': kmeans_labels,
        'Hierarchical': hierarchical_labels,
        'DBSCAN': dbscan_labels
    }

    congruence_results = tester.test_algorithm_congruence(labels_dict)

    print("\n✓ Robustness test complete!")
    print("="*80 + "\n")
