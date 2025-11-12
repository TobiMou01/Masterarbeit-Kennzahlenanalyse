"""
Plot Engine Company Insights Module
Company-specific visualizations and score analysis
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def create_company_insights_plots(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    features: List[str],
    analysis_type: str
):
    """
    Create company insights plots

    Generates:
    - Top/Bottom performers per cluster
    - Outlier detection (companies far from cluster center)
    - Feature distributions per cluster
    - Cluster size distribution

    Args:
        df: DataFrame with cluster assignments and features
        profiles: Cluster profiles (means)
        features: List of features used for clustering
        analysis_type: 'static', 'dynamic', or 'combined'

    External Dependencies:
        - skip_plots: Global flag to skip plot generation
        - output: Output directory manager with get_company_insights_dir() method
    """
    # Import here to avoid circular dependencies
    from src._05_visualization.plot_engine import skip_plots, output

    if skip_plots:
        return

    logger.info(f"\n  💼 Creating Company Insights Plots ({analysis_type})...")

    # Output directory: 4_company_insights/plots/
    insights_dir = output.get_company_insights_dir(analysis_type) / 'plots'
    insights_dir.mkdir(parents=True, exist_ok=True)

    if 'cluster' not in df.columns:
        logger.warning(f"     ⚠️  No cluster column found")
        return

    # 1. Cluster Size Distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        cluster_sizes = df['cluster'].value_counts().sort_index()
        colors = sns.color_palette("husl", n_colors=len(cluster_sizes))

        bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors,
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(insights_dir / 'cluster_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"     ✓ Cluster size distribution plot created")

    except Exception as e:
        logger.warning(f"     ⚠️  Cluster size plot failed: {e}")

    # 2. Feature Distributions per Cluster
    try:
        # Select top 6 most important features (highest variance across clusters)
        available_features = [f for f in features if f in df.columns][:6]

        if len(available_features) >= 2:
            n_features = len(available_features)
            n_cols = 2
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]

            for idx, feature in enumerate(available_features):
                ax = axes[idx]

                # Create violin plot
                data_for_plot = []
                labels_for_plot = []

                for cluster_id in sorted(df['cluster'].unique()):
                    cluster_data = df[df['cluster'] == cluster_id][feature].dropna()
                    data_for_plot.append(cluster_data)
                    labels_for_plot.append(f'C{cluster_id}')

                parts = ax.violinplot(data_for_plot, positions=range(len(data_for_plot)),
                                     showmeans=True, showmedians=True)

                # Color the violins
                colors_violin = sns.color_palette("husl", n_colors=len(data_for_plot))
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_violin[i])
                    pc.set_alpha(0.7)

                ax.set_xticks(range(len(labels_for_plot)))
                ax.set_xticklabels(labels_for_plot)
                ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
                ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

            # Hide unused subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')

            fig.suptitle('Feature Distributions Across Clusters', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(insights_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"     ✓ Feature distribution plots created")

    except Exception as e:
        logger.warning(f"     ⚠️  Feature distribution plots failed: {e}")

    # 3. Top/Bottom Performers (based on overall_score if available)
    try:
        score_column = None
        for col in ['overall_score', 'proximity_score', 'roa', 'roe']:
            if col in df.columns:
                score_column = col
                break

        if score_column:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Top Performers
            ax_top = axes[0]
            top_n = min(10, len(df))
            top_performers = df.nlargest(top_n, score_column)[['conm' if 'conm' in df.columns else 'gvkey', score_column, 'cluster']]

            y_pos = np.arange(top_n)
            colors_top = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in top_performers['cluster'].values]

            ax_top.barh(y_pos, top_performers[score_column].values, color=colors_top, edgecolor='black', alpha=0.8)
            ax_top.set_yticks(y_pos)

            company_col = 'conm' if 'conm' in df.columns else 'gvkey'
            labels_top = [f"{name[:20]}... (C{int(c)})" if len(str(name)) > 20 else f"{name} (C{int(c)})"
                        for name, c in zip(top_performers[company_col].values, top_performers['cluster'].values)]
            ax_top.set_yticklabels(labels_top, fontsize=9)

            ax_top.set_xlabel(score_column.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax_top.set_title(f'Top {top_n} Performers', fontsize=12, fontweight='bold')
            ax_top.grid(True, alpha=0.3, axis='x')

            # Bottom Performers
            ax_bottom = axes[1]
            bottom_performers = df.nsmallest(top_n, score_column)[['conm' if 'conm' in df.columns else 'gvkey', score_column, 'cluster']]

            colors_bottom = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in bottom_performers['cluster'].values]

            ax_bottom.barh(y_pos, bottom_performers[score_column].values, color=colors_bottom, edgecolor='black', alpha=0.8)
            ax_bottom.set_yticks(y_pos)

            labels_bottom = [f"{name[:20]}... (C{int(c)})" if len(str(name)) > 20 else f"{name} (C{int(c)})"
                           for name, c in zip(bottom_performers[company_col].values, bottom_performers['cluster'].values)]
            ax_bottom.set_yticklabels(labels_bottom, fontsize=9)

            ax_bottom.set_xlabel(score_column.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax_bottom.set_title(f'Bottom {top_n} Performers', fontsize=12, fontweight='bold')
            ax_bottom.grid(True, alpha=0.3, axis='x')

            fig.suptitle(f'Company Performance Ranking (by {score_column.replace("_", " ").title()})',
                        fontsize=14, fontweight='bold', y=0.98)

            plt.tight_layout()
            plt.savefig(insights_dir / 'top_bottom_performers.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"     ✓ Top/bottom performers plot created")

    except Exception as e:
        logger.warning(f"     ⚠️  Top/bottom performers plot failed: {e}")

    # 4. Outlier Detection (Distance from Cluster Center)
    try:
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial.distance import cdist

        available_features = [f for f in features if f in df.columns]
        if len(available_features) >= 2:
            X = df[available_features].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Calculate distance to cluster center for each point
            distances = []
            for cluster_id in df['cluster'].unique():
                mask = df['cluster'].values == cluster_id
                cluster_points = X_scaled[mask]

                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0).reshape(1, -1)
                    dists = cdist(cluster_points, center, metric='euclidean').flatten()
                    distances.extend(dists)
                else:
                    distances.extend([0] * mask.sum())

            df_temp = df.copy()
            df_temp['distance_to_center'] = distances

            # Identify outliers (top 5% furthest from center)
            threshold = df_temp['distance_to_center'].quantile(0.95)
            outliers = df_temp[df_temp['distance_to_center'] > threshold].nlargest(15, 'distance_to_center')

            if len(outliers) > 0:
                fig, ax = plt.subplots(figsize=(12, 7))

                y_pos = np.arange(len(outliers))
                colors_outliers = [sns.color_palette("husl", n_colors=10)[int(c) % 10] for c in outliers['cluster'].values]

                ax.barh(y_pos, outliers['distance_to_center'].values, color=colors_outliers,
                       edgecolor='black', alpha=0.8, linewidth=1.5)

                ax.set_yticks(y_pos)
                company_col = 'conm' if 'conm' in df_temp.columns else 'gvkey'
                labels_outliers = [f"{name[:25]}... (C{int(c)})" if len(str(name)) > 25 else f"{name} (C{int(c)})"
                                 for name, c in zip(outliers[company_col].values, outliers['cluster'].values)]
                ax.set_yticklabels(labels_outliers, fontsize=9)

                ax.set_xlabel('Distance to Cluster Center', fontsize=11, fontweight='bold')
                ax.set_title('Outlier Companies (Furthest from Cluster Center)', fontsize=13, fontweight='bold', pad=20)
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'95th Percentile ({threshold:.2f})', alpha=0.7)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3, axis='x')

                plt.tight_layout()
                plt.savefig(insights_dir / 'outliers.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"     ✓ Outlier detection plot created")

    except Exception as e:
        logger.warning(f"     ⚠️  Outlier detection plot failed: {e}")

    logger.info(f"     ✓ Company insights plots saved to {insights_dir.name}/")


def create_score_visualizations(
    df: pd.DataFrame,
    cluster_column: str,
    analysis_type: str
):
    """
    Create score visualizations

    Generates:
    - Score distributions (box plots)
    - Dimensional heatmaps
    - Score correlations
    - Homogeneity comparisons

    Args:
        df: DataFrame with scores
        cluster_column: Name of cluster column
        analysis_type: 'static', 'dynamic', or 'combined'

    External Dependencies:
        - scoring_enabled: Global flag to enable scoring
        - skip_plots: Global flag to skip plot generation
        - output: Output directory manager with get_cluster_quality_dir() method
        - plot_engine_scores: Module with plotting functions
        - score_analyzer: Analyzer object with analyze_cluster_homogeneity() method
    """
    # Import here to avoid circular dependencies
    from src._05_visualization.plot_engine import (
        scoring_enabled, skip_plots, output,
        plot_engine_scores, score_analyzer
    )

    if not scoring_enabled or skip_plots:
        return

    logger.info(f"\n  📊 Creating Score Visualizations ({analysis_type})...")

    # Output directory: 1_cluster_quality/plots/
    viz_dir = output.get_cluster_quality_dir(analysis_type) / 'plots'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Get score columns
    score_columns = [col for col in df.columns if 'score' in col.lower()]
    dimensional_scores = [col for col in score_columns if col.startswith('dim_')]

    # 1. Score distributions
    if 'overall_score' in df.columns:
        plot_engine_scores.plot_score_distribution(
            df=df,
            score_column='overall_score',
            cluster_column=cluster_column,
            output_path=viz_dir / 'score_distribution_overall.png'
        )

    # 2. Dimensional heatmap
    if len(dimensional_scores) > 0:
        plot_engine_scores.plot_dimensional_heatmap(
            df=df,
            dimensional_score_columns=dimensional_scores,
            cluster_column=cluster_column,
            output_path=viz_dir / 'dimensional_heatmap.png'
        )

    # 3. Score correlation matrix
    if len(score_columns) >= 2:
        plot_engine_scores.plot_score_correlation_matrix(
            df=df,
            score_columns=score_columns,
            output_path=viz_dir / 'score_correlations.png'
        )

    # 4. Homogeneity comparison
    if 'overall_score' in df.columns:
        # First analyze homogeneity
        homogeneity_df = score_analyzer.analyze_cluster_homogeneity(
            df=df,
            score_column='overall_score',
            cluster_column=cluster_column
        )

        # Then plot it
        plot_engine_scores.plot_homogeneity_comparison(
            homogeneity_df=homogeneity_df,
            output_path=viz_dir / 'cluster_homogeneity.png'
        )

    logger.info(f"     ✓ Score visualizations created in {viz_dir.name}/")
