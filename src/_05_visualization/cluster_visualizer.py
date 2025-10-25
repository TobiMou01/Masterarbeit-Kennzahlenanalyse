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
    """Correlation heatmap - TODO: Copy implementation from plot_engine.py"""
    pass


def plot_pca_clusters(df, features, output_dir='output/plots'):
    """PCA cluster visualization - TODO: Copy implementation from plot_engine.py"""
    pass


def create_performance_dashboard(df, cluster_means, features, output_dir='output/plots'):
    """Performance dashboard - TODO: Copy implementation from plot_engine.py"""
    pass
