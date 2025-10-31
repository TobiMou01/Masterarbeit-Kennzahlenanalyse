"""
Cluster Visualization Module
Contains all cluster-specific plotting functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Plotting Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_cluster_distribution(df, output_dir='output/plots'):
    """
    Erstellt Balkendiagramm der Cluster-Verteilung.

    Args:
        df: DataFrame mit 'cluster' oder 'cluster_name' Spalte
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Cluster-Verteilung Plot...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cluster zählen
    cluster_col = 'cluster_name' if 'cluster_name' in df.columns else 'cluster'
    cluster_counts = df[cluster_col].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(cluster_counts)), cluster_counts.values,
                   color=sns.color_palette("husl", len(cluster_counts)))

    # Beschriftungen
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Anzahl Unternehmen', fontsize=12, fontweight='bold')
    ax.set_title('Verteilung der Unternehmen auf Cluster',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels(cluster_counts.index, rotation=45, ha='right')

    # Prozentangaben auf Balken
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        percentage = count / cluster_counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    save_path = output_path / 'cluster_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Gespeichert: {save_path}")


def plot_cluster_characteristics(cluster_means, output_dir='output/plots'):
    """
    Erstellt Grouped Bar Chart der Cluster-Charakteristika.

    Args:
        cluster_means: DataFrame mit Cluster-Mittelwerten
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Cluster-Charakteristika Plot...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Wichtigste Kennzahlen auswählen
    key_metrics = ['roa', 'roe', 'ebit_margin', 'equity_ratio', 'current_ratio']
    available_metrics = [m for m in key_metrics if m in cluster_means.columns]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    cluster_means_subset = cluster_means[available_metrics]

    x = np.arange(len(available_metrics))
    width = 0.25

    colors = sns.color_palette("husl", len(cluster_means))

    for i, (idx, row) in enumerate(cluster_means.iterrows()):
        offset = (i - len(cluster_means)/2) * width
        ax.bar(x + offset, row[available_metrics].values,
               width, label=f'Cluster {idx}', color=colors[i])

    ax.set_xlabel('Kennzahlen', fontsize=12, fontweight='bold')
    ax.set_ylabel('Durchschnittswert', fontsize=12, fontweight='bold')
    ax.set_title('Cluster-Charakteristika im Vergleich',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics],
                        rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = output_path / 'cluster_characteristics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Gespeichert: {save_path}")


# Additional functions would be added here (plot_scatter_matrix, plot_correlation_heatmap, plot_pca_clusters, create_performance_dashboard)
# For brevity, I'll add placeholders

def plot_scatter_matrix(df, output_dir='output/plots'):
    """Scatter matrix plot - TODO: Copy implementation from plot_engine.py"""
    pass


def plot_correlation_heatmap(df, output_dir='output/plots'):
    """
    Erstellt Korrelations-Heatmap der Features.

    Args:
        df: DataFrame mit Features
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Korrelations-Heatmap...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Nur numerische Features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude metadata columns
    exclude_cols = ['gvkey', 'fyear', 'cluster', 'datadate']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.endswith('_flag')]

    # Limit to top 15 features if too many
    if len(feature_cols) > 15:
        # Use variance to select most interesting features
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(15).index.tolist()

    if len(feature_cols) == 0:
        logger.warning("Keine Features für Korrelationsheatmap gefunden")
        return

    # Calculate correlation
    corr_matrix = df[feature_cols].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                ax=ax)

    ax.set_title('Korrelationsmatrix der Features',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    save_path = output_path / 'correlation_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Gespeichert: {save_path}")


def plot_pca_clusters(df, features, output_dir='output/plots'):
    """
    Erstellt PCA Visualisierung der Cluster in 2D.

    Args:
        df: DataFrame mit Cluster-Zuordnungen
        features: Liste der Features für PCA
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle PCA Cluster Plot...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter features that exist in df
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        logger.warning(f"Nicht genug Features für PCA ({len(available_features)})")
        return

    # Prepare data
    X = df[available_features].fillna(0).values

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Get cluster info
    cluster_col = 'cluster_name' if 'cluster_name' in df.columns else 'cluster'
    clusters = df[cluster_col].values

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_clusters = sorted(df['cluster'].unique()) if 'cluster' in df.columns else sorted(df[cluster_col].unique())
    colors = sns.color_palette("husl", len(unique_clusters))

    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id if cluster_col == 'cluster' else clusters == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Varianz)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Varianz)',
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA: Cluster in 2D Projektion',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_path = output_path / 'pca_clusters.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Gespeichert: {save_path}")


def create_performance_dashboard(df, cluster_means, features, output_dir='output/plots'):
    """
    Erstellt Performance Dashboard mit mehreren Subplots.

    Args:
        df: DataFrame mit Cluster-Zuordnungen
        cluster_means: DataFrame mit Cluster-Mittelwerten
        features: Liste der Features
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Performance Dashboard...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    cluster_col = 'cluster_name' if 'cluster_name' in df.columns else 'cluster'

    # 1. Cluster Size
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = df[cluster_col].value_counts().sort_index()
    bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values,
                   color=sns.color_palette("husl", len(cluster_counts)))
    ax1.set_title('Cluster-Größen', fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Anzahl Unternehmen')
    ax1.set_xticks(range(len(cluster_counts)))
    ax1.set_xticklabels(cluster_counts.index, rotation=45, ha='right')

    # 2. Top 3 Kennzahlen
    ax2 = fig.add_subplot(gs[0, 1])
    key_metrics = ['roa', 'roe', 'ebit_margin']
    available = [m for m in key_metrics if m in cluster_means.columns][:3]

    if available:
        x = np.arange(len(available))
        width = 0.15
        colors = sns.color_palette("husl", len(cluster_means))

        for i, (idx, row) in enumerate(cluster_means.iterrows()):
            offset = (i - len(cluster_means)/2) * width
            ax2.bar(x + offset, row[available].values,
                   width, label=f'Cluster {idx}', color=colors[i])

        ax2.set_title('Key Profitability Metrics', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in available])
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

    # 3. Liquidität & Leverage
    ax3 = fig.add_subplot(gs[1, 0])
    liquidity_metrics = ['current_ratio', 'equity_ratio']
    available = [m for m in liquidity_metrics if m in cluster_means.columns]

    if available:
        x = np.arange(len(available))
        width = 0.15
        for i, (idx, row) in enumerate(cluster_means.iterrows()):
            offset = (i - len(cluster_means)/2) * width
            ax3.bar(x + offset, row[available].values,
                   width, label=f'Cluster {idx}', color=colors[i])

        ax3.set_title('Liquidität & Kapitalstruktur', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in available], rotation=15)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)

    # 4. Feature Importance (variance)
    ax4 = fig.add_subplot(gs[1, 1])
    available_features = [f for f in features if f in df.columns][:10]

    if available_features:
        variances = df[available_features].var().sort_values(ascending=False).head(8)
        ax4.barh(range(len(variances)), variances.values, color='steelblue')
        ax4.set_yticks(range(len(variances)))
        ax4.set_yticklabels([f.replace('_', ' ').title() for f in variances.index], fontsize=9)
        ax4.set_xlabel('Varianz')
        ax4.set_title('Feature Importance (Varianz)', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

    # 5. Distribution of key metric
    ax5 = fig.add_subplot(gs[2, :])
    key_metric = 'roa' if 'roa' in df.columns else available_features[0] if available_features else None

    if key_metric:
        for cluster_id in sorted(df['cluster'].unique()) if 'cluster' in df.columns else []:
            cluster_data = df[df['cluster'] == cluster_id][key_metric].dropna()
            ax5.hist(cluster_data, bins=30, alpha=0.5,
                    label=f'Cluster {cluster_id}', edgecolor='black')

        ax5.set_xlabel(key_metric.upper(), fontweight='bold')
        ax5.set_ylabel('Häufigkeit')
        ax5.set_title(f'Verteilung: {key_metric.upper()} pro Cluster', fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(axis='y', alpha=0.3)

    # Overall title
    fig.suptitle('Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)

    save_path = output_path / 'performance_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Gespeichert: {save_path}")
