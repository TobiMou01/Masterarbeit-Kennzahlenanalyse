"""
Plot Engine Insights Module
Advanced visualizations for clustering analysis
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def create_pca_plots(
    df: pd.DataFrame,
    features: List[str],
    analysis_type: str
):
    """
    Create PCA visualization plots

    Generates:
    - Scree plot (variance explained)
    - Component loadings heatmap
    - Biplot (observations + features)
    - Cluster separation in PCA space

    Args:
        df: DataFrame with features and cluster assignments
        features: List of feature names used for clustering
        analysis_type: 'static', 'dynamic', or 'combined'
    """
    # PCA plots are always enabled when this function is called

    logger.info(f"\n  🔬 Creating PCA Plots ({analysis_type})...")

    from src._02_preprocessing.pca_transformer import PCATransformer

    # Output directory: 5_pca_analysis/plots/
    pca_dir = output.get_pca_analysis_dir(analysis_type) / 'plots'
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Validate features exist in DataFrame
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        logger.warning(f"     ⚠️  Not enough features ({len(available_features)}) for PCA")
        return

    # Apply PCA transformation
    pca_transformer = PCATransformer(n_components=0.85)
    try:
        X_pca, variance_summary = pca_transformer.fit_transform(
            df=df,
            features=available_features
        )
    except Exception as e:
        logger.warning(f"     ⚠️  PCA transformation failed: {e}")
        return

    n_components = pca_transformer.pca.n_components_
    explained_variance = pca_transformer.pca.explained_variance_ratio_

    logger.info(f"     ✓ PCA complete: {len(available_features)} features → {n_components} components")

    # Get cluster labels
    cluster_labels = df['cluster'].values if 'cluster' in df.columns else np.zeros(len(df))

    # 1. Scree Plot
    try:
        plot_engine_pca.plot_scree_plot(
            explained_variance,
            output_path=pca_dir / 'scree_plot.png',
            cumulative=True,
            threshold=0.85
        )
    except Exception as e:
        logger.warning(f"     ⚠️  Scree plot failed: {e}")

    # 2. Component Loadings Heatmap
    try:
        component_loadings = pca_transformer.get_component_loadings()
        plot_engine_pca.plot_component_loadings_heatmap(
            component_loadings,
            output_path=pca_dir / 'component_loadings.png',
            top_n_features=min(15, len(available_features))
        )
    except Exception as e:
        logger.warning(f"     ⚠️  Component loadings plot failed: {e}")

    # 3. Biplot (if we have at least 2 components)
    if n_components >= 2:
        try:
            loadings = pca_transformer.pca.components_.T
            plot_engine_pca.plot_biplot(
                X_pca,
                loadings,
                available_features,
                cluster_labels,
                output_path=pca_dir / 'biplot_pc1_pc2.png',
                pc_x=0,
                pc_y=1,
                n_features_show=min(10, len(available_features))
            )
        except Exception as e:
            logger.warning(f"     ⚠️  Biplot failed: {e}")

    # 4. Cluster Separation in PCA Space (if we have clusters)
    if n_components >= 2 and len(np.unique(cluster_labels)) > 1:
        try:
            cluster_names = [f'Cluster {i}' for i in sorted(np.unique(cluster_labels))]

            # Determine which component pairs to plot
            components_to_plot = [(0, 1)]  # Always PC1 vs PC2
            if n_components >= 3:
                components_to_plot.extend([(0, 2), (1, 2)])

            plot_engine_pca.plot_cluster_separation_in_pca_space(
                X_pca,
                cluster_labels,
                cluster_names,
                output_path=pca_dir / 'cluster_separation.png',
                components=components_to_plot
            )
        except Exception as e:
            logger.warning(f"     ⚠️  Cluster separation plot failed: {e}")

    logger.info(f"     ✓ PCA plots saved to {pca_dir.name}/")



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
    """
    # Company insights plots are always enabled when this function is called

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

# =========================================================================
# ORCHESTRATION METHODS
# =========================================================================



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
    """
    # Validation plots are always enabled when this function is called

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
    """
    # Score visualizations are always enabled when this function is called

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



