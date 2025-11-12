"""
Plot Engine Algorithm Congruence Module
Algorithm congruence analysis and robustness testing
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def create_algorithm_congruence_plots(
    df: pd.DataFrame,
    analysis_type: str
):
    """
    Create algorithm congruence plots

    Tests clustering robustness by running multiple initializations
    and comparing results using ARI (Adjusted Rand Index).

    Generates:
    - ARI heatmap (multiple runs)
    - Stability analysis
    - Confusion matrix

    Args:
        df: DataFrame with cluster assignments
        analysis_type: 'static', 'dynamic', or 'combined'

    External Dependencies:
        - validation_enabled: Global flag to enable validation
        - skip_plots: Global flag to skip plot generation
        - output: Output directory manager with get_algorithm_congruence_dir() method
        - plot_engine_validation: Module with plotting functions for validation
        - algorithm: String indicating the clustering algorithm used
    """
    # Import here to avoid circular dependencies
    from src._05_visualization.plot_engine import (
        validation_enabled, skip_plots, output,
        plot_engine_validation, algorithm
    )

    if not validation_enabled or skip_plots:
        return

    logger.info(f"\n  🔄 Creating Algorithm Congruence Plots ({analysis_type})...")

    # Output directory: 2_algorithm_congruence/plots/
    congruence_dir = output.get_algorithm_congruence_dir(analysis_type) / 'plots'
    congruence_dir.mkdir(parents=True, exist_ok=True)

    # Get features and cluster column
    if 'cluster' not in df.columns:
        logger.warning(f"     ⚠️  No cluster column found")
        return

    # Run multiple clusterings with different seeds to test robustness
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    n_runs = 5  # Number of random initializations
    cluster_assignments = {}
    cluster_assignments['original'] = df['cluster'].values

    # Store features used for clustering (try to infer from DataFrame)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_candidates = [col for col in numeric_cols
                        if col not in ['cluster', 'gvkey', 'fyear', 'datadate', 'gsector', 'sic', 'naics']
                        and not col.endswith('_outlier')
                        and not col.endswith('_score')]

    features = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates

    if len(features) < 2:
        logger.warning(f"     ⚠️  Not enough features ({len(features)}) for re-clustering")
        # Create placeholder info file
        info_path = congruence_dir / 'info.txt'
        with open(info_path, 'w') as f:
            f.write("Algorithm Congruence Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write("Algorithm congruence plots are only available when:\n")
            f.write("1. Multiple clustering algorithms are compared, OR\n")
            f.write("2. The same algorithm runs with different parameters\n\n")
            f.write(f"Current run: Single {algorithm} clustering\n")
            f.write(f"For multi-algorithm comparison, use:\n")
            f.write("  pipeline.run_multi_algorithm_comparison()\n")
        logger.info(f"     ℹ️  Created info file: {info_path.name}")
        return

    # Perform multiple runs with different seeds
    logger.info(f"     Running {n_runs} clusterings with different initializations...")

    for i in range(n_runs):
        try:
            # Re-run clustering with different seed
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            X = df[features].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_clusters = len(df['cluster'].unique())
            model = KMeans(n_clusters=n_clusters, random_state=42 + i, n_init=10)
            labels = model.fit_predict(X_scaled)

            cluster_assignments[f'run_{i+1}'] = labels

        except Exception as e:
            logger.warning(f"     ⚠️  Run {i+1} failed: {e}")

    if len(cluster_assignments) < 2:
        logger.warning(f"     ⚠️  Not enough successful runs for comparison")
        return

    # Calculate ARI matrix
    run_names = list(cluster_assignments.keys())
    n_comparisons = len(run_names)
    ari_matrix = np.ones((n_comparisons, n_comparisons))

    for i, name1 in enumerate(run_names):
        for j, name2 in enumerate(run_names):
            if i < j:
                ari = adjusted_rand_score(
                    cluster_assignments[name1],
                    cluster_assignments[name2]
                )
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari

    ari_df = pd.DataFrame(ari_matrix, index=run_names, columns=run_names)

    logger.info(f"     ✓ ARI Matrix calculated (mean ARI: {ari_df.values[np.triu_indices_from(ari_df.values, k=1)].mean():.3f})")

    # Create ARI heatmap
    try:
        plot_engine_validation.plot_ari_heatmap(
            ari_df,
            output_path=congruence_dir / 'ari_heatmap_robustness.png',
            title=f'Clustering Robustness: ARI Across {n_runs+1} Runs'
        )
    except Exception as e:
        logger.warning(f"     ⚠️  ARI heatmap failed: {e}")

    # Create confusion matrix (original vs run_1)
    if 'run_1' in cluster_assignments:
        try:
            conf_matrix = sklearn_confusion_matrix(
                cluster_assignments['original'],
                cluster_assignments['run_1']
            )
            conf_df = pd.DataFrame(conf_matrix)

            plot_engine_validation.plot_confusion_matrix(
                conf_df,
                'Original',
                'Run 1',
                output_path=congruence_dir / 'confusion_matrix.png',
                title='Cluster Assignment Consistency'
            )
        except Exception as e:
            logger.warning(f"     ⚠️  Confusion matrix failed: {e}")

    # Save ARI matrix
    ari_df.to_csv(congruence_dir.parent / 'ari_matrix.csv')

    logger.info(f"     ✓ Algorithm congruence plots saved to {congruence_dir.name}/")
