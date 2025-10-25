"""
Modul 4: Visualizer
Erstellt Visualisierungen der Analyseergebnisse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        cluster_name = cluster_means.loc[idx, 'cluster_name'] if 'cluster_name' in cluster_means.columns else f'Cluster {idx}'
        offset = width * (i - len(cluster_means)/2 + 0.5)
        ax.bar(x + offset, row[available_metrics], width, 
               label=cluster_name, color=colors[i])
    
    ax.set_xlabel('Kennzahlen', fontsize=12, fontweight='bold')
    ax.set_ylabel('Durchschnittswert (%)', fontsize=12, fontweight='bold')
    ax.set_title('Durchschnittliche Kennzahlen pro Cluster', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in available_metrics], 
                       rotation=45, ha='right')
    ax.legend(title='Cluster', loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_path / 'cluster_characteristics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gespeichert: {save_path}")


def plot_scatter_matrix(df, output_dir='output/plots'):
    """
    Erstellt Scatter-Matrix der wichtigsten Kennzahlen.
    
    Args:
        df: DataFrame mit Kennzahlen und Cluster
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Scatter Matrix...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Wichtige Kennzahlen
    key_metrics = ['roa', 'roe', 'ebit_margin', 'debt_to_equity']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        logger.warning("Nicht genug Metriken für Scatter Matrix")
        return
    
    # Nur Zeilen mit Cluster-Zuordnung
    df_plot = df[df['cluster'] >= 0][available_metrics + ['cluster']].dropna()
    
    # Begrenzen auf Hauptbereich (ohne extreme Ausreißer)
    for col in available_metrics:
        q99 = df_plot[col].quantile(0.99)
        q01 = df_plot[col].quantile(0.01)
        df_plot = df_plot[(df_plot[col] >= q01) & (df_plot[col] <= q99)]
    
    # Scatter Matrix
    fig = plt.figure(figsize=(12, 12))
    n_clusters = int(df_plot['cluster'].max()) + 1
    colors = sns.color_palette("husl", n_clusters)
    
    pd.plotting.scatter_matrix(
        df_plot[available_metrics],
        figsize=(12, 12),
        c=[colors[int(c)] for c in df_plot['cluster']],
        alpha=0.6,
        diagonal='kde',
        ax=None
    )
    
    plt.suptitle('Scatter Matrix der Kennzahlen (nach Cluster farbcodiert)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    save_path = output_path / 'scatter_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gespeichert: {save_path}")


def plot_correlation_heatmap(df, output_dir='output/plots'):
    """
    Erstellt Korrelations-Heatmap der Kennzahlen.
    
    Args:
        df: DataFrame mit Kennzahlen
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Korrelations-Heatmap...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Nur numerische Kennzahl-Spalten
    ratio_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in 
        ['ratio', 'margin', 'turnover', 'roa', 'roe', 'growth', 'equity']
    ) and 'outlier' not in col]
    
    # Korrelationsmatrix
    corr_matrix = df[ratio_columns].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Korrelationsmatrix der Finanzkennzahlen', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    save_path = output_path / 'correlation_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gespeichert: {save_path}")


def plot_pca_clusters(df, features, output_dir='output/plots'):
    """
    PCA-Visualisierung der Cluster in 2D.
    
    Args:
        df: DataFrame mit Features und Cluster
        features: Liste der Feature-Namen
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle PCA Cluster-Visualisierung...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Nur Zeilen mit Cluster und ohne NaN
    df_plot = df[df['cluster'] >= 0][features + ['cluster']].dropna()
    
    if len(df_plot) < 10:
        logger.warning("Zu wenig Datenpunkte für PCA")
        return
    
    # PCA auf 2 Dimensionen
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_plot[features])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_clusters = int(df_plot['cluster'].max()) + 1
    colors = sns.color_palette("husl", n_clusters)
    
    for cluster in sorted(df_plot['cluster'].unique()):
        mask = df_plot['cluster'] == cluster
        cluster_name = df_plot[mask]['cluster_name'].iloc[0] if 'cluster_name' in df_plot.columns else f'Cluster {cluster}'
        
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=[colors[int(cluster) % len(colors)]], 
                  label=cluster_name,
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianz)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianz)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA-Projektion der Cluster (2D)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Cluster', loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_path / 'pca_clusters.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gespeichert: {save_path}")
    logger.info(f"  Erklärte Varianz: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
               f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")


def create_performance_dashboard(df, cluster_means, features, output_dir='output/plots'):
    """
    Kombiniert mehrere Plots in einem Dashboard.
    
    Args:
        df: DataFrame mit allen Daten
        cluster_means: Cluster-Charakteristika
        features: Feature-Liste
        output_dir: Ausgabeverzeichnis
    """
    logger.info("Erstelle Performance Dashboard...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cluster-Verteilung
    ax1 = plt.subplot(2, 3, 1)
    cluster_col = 'cluster_name' if 'cluster_name' in df.columns else 'cluster'
    cluster_counts = df[cluster_col].value_counts().sort_index()
    colors = sns.color_palette("husl", len(cluster_counts))
    ax1.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
    ax1.set_title('Cluster-Verteilung', fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Anzahl Unternehmen')
    ax1.set_xticks(range(len(cluster_counts)))
    ax1.set_xticklabels(cluster_counts.index, rotation=45, ha='right')
    
    # 2. ROA Verteilung pro Cluster
    ax2 = plt.subplot(2, 3, 2)
    df_valid = df[df['cluster'] >= 0]
    for cluster in sorted(df_valid['cluster'].unique()):
        data = df_valid[df_valid['cluster'] == cluster]['roa'].dropna()
        data = data[(data > -50) & (data < 100)]  # Ausreißer filtern
        ax2.hist(data, bins=30, alpha=0.5, 
                label=f'Cluster {cluster}', color=colors[int(cluster) % len(colors)])
    ax2.set_title('ROA Verteilung', fontweight='bold')
    ax2.set_xlabel('ROA (%)')
    ax2.set_ylabel('Häufigkeit')
    ax2.legend()
    
    # 3. Cluster-Charakteristika
    ax3 = plt.subplot(2, 3, 3)
    key_metrics = ['roa', 'roe', 'ebit_margin']
    available = [m for m in key_metrics if m in cluster_means.columns]
    cluster_means[available].T.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Durchschnittliche Kennzahlen', fontweight='bold')
    ax3.set_ylabel('Wert (%)')
    ax3.set_xticklabels([m.upper() for m in available], rotation=45, ha='right')
    ax3.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
    
    # 4. ROA vs ROE Scatter
    ax4 = plt.subplot(2, 3, 4)
    if 'roe' in df.columns:
        for cluster in sorted(df_valid['cluster'].unique()):
            data = df_valid[df_valid['cluster'] == cluster][['roa', 'roe']].dropna()
            data = data[(data['roa'] > -50) & (data['roa'] < 100) & 
                       (data['roe'] > -50) & (data['roe'] < 150)]
            ax4.scatter(data['roa'], data['roe'], 
                       alpha=0.5, s=20, color=colors[int(cluster) % len(colors)],
                       label=f'Cluster {cluster}')
        ax4.set_title('ROA vs ROE', fontweight='bold')
        ax4.set_xlabel('ROA (%)')
        ax4.set_ylabel('ROE (%)')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # 5. Equity Ratio Distribution
    ax5 = plt.subplot(2, 3, 5)
    if 'equity_ratio' in df.columns:
        for cluster in sorted(df_valid['cluster'].unique()):
            data = df_valid[df_valid['cluster'] == cluster]['equity_ratio'].dropna()
            data = data[(data > 0) & (data < 100)]
            ax5.hist(data, bins=20, alpha=0.5,
                label=f'Cluster {cluster}', color=colors[int(cluster) % len(colors)])
        ax5.set_title('Eigenkapitalquote Verteilung', fontweight='bold')
        ax5.set_xlabel('Eigenkapitalquote (%)')
        ax5.set_ylabel('Häufigkeit')
        ax5.legend()
    
    # 6. Box Plot Key Metrics
    ax6 = plt.subplot(2, 3, 6)
    box_data = []
    labels = []
    for cluster in sorted(df_valid['cluster'].unique()):
        data = df_valid[df_valid['cluster'] == cluster]['roa'].dropna()
        data = data[(data > -50) & (data < 100)]
        box_data.append(data)
        labels.append(f'C{cluster}')
    
    bp = ax6.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax6.set_title('ROA Box Plot pro Cluster', fontweight='bold')
    ax6.set_ylabel('ROA (%)')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Dashboard - Unternehmensklassifikation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_path / 'performance_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gespeichert: {save_path}")


def export_plots_to_pdf(output_dir='output/plots'):
    """
    Kombiniert alle PNG-Plots in ein PDF.
    
    Args:
        output_dir: Verzeichnis mit PNG-Dateien
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    logger.info("Exportiere Plots zu PDF...")
    
    output_path = Path(output_dir)
    png_files = sorted(output_path.glob('*.png'))
    
    if not png_files:
        logger.warning("Keine PNG-Dateien gefunden")
        return
    
    pdf_path = output_path / 'all_plots.pdf'
    
    with PdfPages(pdf_path) as pdf:
        for png_file in png_files:
            img = plt.imread(png_file)
            fig = plt.figure(figsize=(11, 8.5))  # Letter size
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()
    
    logger.info(f"✓ PDF erstellt: {pdf_path} ({len(png_files)} Plots)")


def create_all_plots(df, cluster_profiles, features, analysis_type='static', output_dir='output/germany/plots'):
    """
    Erstellt alle Visualisierungen für eine Analyse.

    Args:
        df: DataFrame mit Cluster-Zuordnungen
        cluster_profiles: DataFrame mit Cluster-Profilen
        features: Liste der verwendeten Features
        analysis_type: 'static', 'dynamic', oder 'combined'
        output_dir: Ausgabeverzeichnis

    Returns:
        List der erstellten Plot-Pfade
    """
    logger.info(f"\n  Erstelle Visualisierungen für {analysis_type} Analysis...")

    # Erstelle Unterordner pro Analyse-Typ
    output_path = Path(output_dir) / analysis_type
    output_path.mkdir(parents=True, exist_ok=True)

    created_plots = []

    try:
        # 1. Cluster-Verteilung
        plot_cluster_distribution(df, str(output_path))
        created_plots.append(output_path / 'cluster_distribution.png')

        # 2. Cluster-Charakteristika
        plot_cluster_characteristics(cluster_profiles, str(output_path))
        created_plots.append(output_path / 'cluster_characteristics.png')

        # 3. PCA-Visualisierung
        plot_pca_clusters(df, features, str(output_path))
        created_plots.append(output_path / 'pca_clusters.png')

        # 4. Performance Dashboard
        create_performance_dashboard(df, cluster_profiles, features, str(output_path))
        created_plots.append(output_path / 'performance_dashboard.png')

        # 5. Correlation Heatmap (nur wenn genug Features)
        if len(features) > 3:
            plot_correlation_heatmap(df, str(output_path))
            created_plots.append(output_path / 'correlation_heatmap.png')

        logger.info(f"    ✓ {len(created_plots)} Visualisierungen erstellt")

    except Exception as e:
        logger.warning(f"    ⚠ Fehler bei Visualisierung: {e}")

    return created_plots


def main():
    """Beispiel-Verwendung des Moduls."""

    # Klassifizierte Daten laden
    df = pd.read_csv('output/germany/data/static_assignments.csv')
    cluster_profiles = pd.read_csv('output/germany/data/static_profiles.csv', index_col=0)

    logger.info(f"Geladen: {len(df)} Unternehmen")

    # Features für Static Analysis
    features = ['roa', 'roe', 'ebit_margin', 'equity_ratio',
                'current_ratio', 'debt_to_equity', 'net_profit_margin']
    available_features = [f for f in features if f in df.columns]

    # Alle Plots erstellen
    create_all_plots(df, cluster_profiles, available_features,
                    analysis_type='static',
                    output_dir='output/germany/plots')

    logger.info("\n✅ Alle Visualisierungen erstellt!")


if __name__ == "__main__":
    main()