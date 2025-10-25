"""
Plot Engine - Orchestrator for all visualizations
Delegates to cluster_visualizer and comparison_visualizer
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional

from src._05_visualization import cluster_visualizer
from src._05_visualization import comparison_visualizer

logger = logging.getLogger(__name__)


def create_all_plots(df, cluster_profiles, features, analysis_type='static', output_dir='output/germany/plots'):
    """
    Orchestrates creation of all plots

    Args:
        df: DataFrame with clustered data
        cluster_profiles: DataFrame with cluster profiles/means
        features: List of feature names
        analysis_type: Type of analysis ('static', 'dynamic', 'combined')
        output_dir: Output directory for plots

    Returns:
        dict: Results including number of plots created
    """
    logger.info(f"Creating all plots for {analysis_type} analysis...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots_created = 0

    try:
        # Cluster-specific plots
        cluster_visualizer.plot_cluster_distribution(df, output_dir)
        plots_created += 1

        cluster_visualizer.plot_cluster_characteristics(cluster_profiles, output_dir)
        plots_created += 1

        cluster_visualizer.plot_correlation_heatmap(df, output_dir)
        plots_created += 1

        cluster_visualizer.plot_pca_clusters(df, features, output_dir)
        plots_created += 1

        cluster_visualizer.create_performance_dashboard(df, cluster_profiles, features, output_dir)
        plots_created += 1

        logger.info(f"✓ Created {plots_created} cluster visualization plots")

    except Exception as e:
        logger.error(f"Error creating plots: {e}", exc_info=True)

    return {
        'plots_created': plots_created,
        'output_dir': str(output_path)
    }


def create_comparison_plots(results_dict, output_dir='output/germany/comparisons'):
    """
    Orchestrates creation of all comparison plots

    Args:
        results_dict: Dictionary with comparison results
        output_dir: Output directory for comparison plots

    Returns:
        dict: Results including number of plots created
    """
    logger.info("Creating comparison plots...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots_created = 0

    try:
        if 'gics' in results_dict:
            comparison_visualizer.plot_gics_comparison(results_dict['gics'], output_dir)
            plots_created += 1

        if 'algorithms' in results_dict:
            comparison_visualizer.plot_algorithm_comparison(results_dict['algorithms'], output_dir)
            plots_created += 1

        if 'features' in results_dict:
            comparison_visualizer.plot_feature_importance(results_dict['features'], output_dir)
            plots_created += 1

        if 'temporal' in results_dict:
            comparison_visualizer.plot_temporal_stability(results_dict['temporal'], output_dir)
            plots_created += 1

        logger.info(f"✓ Created {plots_created} comparison plots")

    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}", exc_info=True)

    return {
        'plots_created': plots_created,
        'output_dir': str(output_path)
    }
