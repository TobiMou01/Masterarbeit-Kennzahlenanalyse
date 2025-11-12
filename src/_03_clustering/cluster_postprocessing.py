"""
Cluster Post-Processing Utilities
Handles minimum cluster size constraints and cluster merging
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


def enforce_min_cluster_size(
    X: np.ndarray,
    labels: np.ndarray,
    min_size: int = 10,
    merge_strategy: str = 'nearest'
) -> Tuple[np.ndarray, dict]:
    """
    Enforces minimum cluster size constraint by merging small clusters.

    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        min_size: Minimum cluster size (default: 10, ~6% for n=160)
        merge_strategy: How to merge small clusters ('nearest', 'largest')
            - 'nearest': Merge into nearest cluster (by centroid distance)
            - 'largest': Merge into largest cluster

    Returns:
        Tuple of (new_labels, merge_info_dict)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Filter out noise label (-1) for DBSCAN
    mask = unique_labels != -1
    unique_labels = unique_labels[mask]
    counts = counts[mask]

    # Find small clusters
    small_clusters = unique_labels[counts < min_size]

    if len(small_clusters) == 0:
        logger.info(f"  ✓ All clusters meet minimum size constraint ({min_size})")
        return labels, {'n_merges': 0, 'merged_clusters': []}

    logger.info(f"  ⚠️  Found {len(small_clusters)} clusters below minimum size ({min_size}):")
    for cluster_id in small_clusters:
        size = counts[unique_labels == cluster_id][0]
        logger.info(f"     Cluster {cluster_id}: {size} samples")

    # Make a copy of labels
    new_labels = labels.copy()

    # Calculate cluster centroids
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)

    merge_info = {
        'n_merges': 0,
        'merged_clusters': []
    }

    # Merge small clusters
    for small_cluster in small_clusters:
        cluster_mask = labels == small_cluster
        cluster_samples = X[cluster_mask]
        cluster_size = cluster_mask.sum()

        if merge_strategy == 'nearest':
            # Find nearest cluster by centroid distance
            small_centroid = centroids[small_cluster]

            # Calculate distances to all other clusters
            distances = {}
            for other_cluster in unique_labels:
                if other_cluster == small_cluster:
                    continue
                # Skip if already small (will be merged too)
                if other_cluster in small_clusters:
                    continue

                other_centroid = centroids[other_cluster]
                dist = np.linalg.norm(small_centroid - other_centroid)
                distances[other_cluster] = dist

            if len(distances) == 0:
                # All other clusters are also small - merge into largest remaining
                remaining_clusters = unique_labels[~np.isin(unique_labels, small_clusters)]
                if len(remaining_clusters) > 0:
                    target_cluster = remaining_clusters[0]
                else:
                    # All clusters are small - keep as is
                    logger.warning(f"     Cannot merge cluster {small_cluster}: All clusters are small")
                    continue
            else:
                # Merge into nearest
                target_cluster = min(distances, key=distances.get)

        elif merge_strategy == 'largest':
            # Merge into largest cluster
            target_cluster = unique_labels[counts.argmax()]

        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Perform merge
        new_labels[cluster_mask] = target_cluster

        logger.info(f"     → Merged cluster {small_cluster} ({cluster_size} samples) into cluster {target_cluster}")

        merge_info['n_merges'] += 1
        merge_info['merged_clusters'].append({
            'from_cluster': int(small_cluster),
            'to_cluster': int(target_cluster),
            'size': int(cluster_size)
        })

    # Relabel to have consecutive cluster IDs
    new_labels = relabel_consecutive(new_labels)

    # Summary
    final_unique, final_counts = np.unique(new_labels[new_labels != -1], return_counts=True)
    logger.info(f"\n  ✓ Post-merge: {len(final_unique)} clusters, "
               f"sizes: {final_counts.min()}-{final_counts.max()}")

    return new_labels, merge_info


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """
    Relabels clusters to have consecutive IDs (0, 1, 2, ...).
    Preserves noise label (-1) for DBSCAN.

    Args:
        labels: Original cluster labels

    Returns:
        Relabeled array with consecutive IDs
    """
    # Separate noise from clusters
    noise_mask = labels == -1
    cluster_labels = labels[~noise_mask]

    if len(cluster_labels) == 0:
        return labels

    # Create mapping
    unique_clusters = np.unique(cluster_labels)
    mapping = {old: new for new, old in enumerate(sorted(unique_clusters))}

    # Apply mapping
    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels


def get_cluster_statistics(labels: np.ndarray) -> pd.DataFrame:
    """
    Calculates cluster size statistics.

    Args:
        labels: Cluster labels

    Returns:
        DataFrame with cluster statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    stats = []
    total = len(labels)

    for label, count in zip(unique_labels, counts):
        stats.append({
            'cluster_id': int(label) if label != -1 else 'Noise',
            'size': int(count),
            'percentage': f"{count/total*100:.1f}%"
        })

    df = pd.DataFrame(stats)
    df = df.sort_values('size', ascending=False)

    return df


def check_cluster_balance(labels: np.ndarray, max_imbalance_ratio: float = 10.0) -> dict:
    """
    Checks if clusters are reasonably balanced.

    Args:
        labels: Cluster labels
        max_imbalance_ratio: Maximum allowed ratio between largest and smallest cluster

    Returns:
        Dict with balance check results
    """
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if len(counts) == 0:
        return {'balanced': False, 'reason': 'No clusters found'}

    min_size = counts.min()
    max_size = counts.max()
    ratio = max_size / min_size if min_size > 0 else float('inf')

    balanced = ratio <= max_imbalance_ratio

    return {
        'balanced': balanced,
        'min_size': int(min_size),
        'max_size': int(max_size),
        'imbalance_ratio': float(ratio),
        'threshold': max_imbalance_ratio,
        'warning': None if balanced else f"Clusters are imbalanced (ratio: {ratio:.1f}x)"
    }


if __name__ == "__main__":
    """Test cluster post-processing"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("CLUSTER POST-PROCESSING TEST")
    print("="*80)

    # Mock data with small clusters
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # Create labels with some small clusters
    labels = np.array([0]*50 + [1]*30 + [2]*15 + [3]*3 + [4]*2)
    np.random.shuffle(labels)

    print("\nOriginal cluster sizes:")
    print(get_cluster_statistics(labels))

    print("\nBalance check:")
    balance = check_cluster_balance(labels, max_imbalance_ratio=10.0)
    print(f"  Balanced: {balance['balanced']}")
    print(f"  Min size: {balance['min_size']}")
    print(f"  Max size: {balance['max_size']}")
    print(f"  Ratio: {balance['imbalance_ratio']:.1f}x")

    print("\nEnforcing minimum cluster size (min_size=10)...")
    new_labels, merge_info = enforce_min_cluster_size(X, labels, min_size=10)

    print("\nPost-merge cluster sizes:")
    print(get_cluster_statistics(new_labels))

    print("\nMerge info:")
    print(f"  Number of merges: {merge_info['n_merges']}")
    for merge in merge_info['merged_clusters']:
        print(f"  Cluster {merge['from_cluster']} ({merge['size']} samples) → Cluster {merge['to_cluster']}")

    print("\n✓ Test complete!")
    print("="*80 + "\n")
