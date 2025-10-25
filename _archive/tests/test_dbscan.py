"""
Test-Skript für DBSCAN Clustering
Demonstriert Verwendung des Density-Based Algorithmus
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Module
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.clustering import DBSCANClusterer, KMeansClusterer, ClustererFactory
from src.config_loader import get_config

print("\n" + "=" * 80)
print("DBSCAN CLUSTERING TEST")
print("=" * 80 + "\n")

# ===== TEST 1: Verfügbare Algorithmen =====
print("1. VERFÜGBARE ALGORITHMEN")
print("-" * 40)
algorithms = ClustererFactory.get_available_algorithms()
for algo in algorithms:
    print(f"  ✓ {algo}")

# ===== TEST 2: Config laden =====
print("\n2. KONFIGURATION")
print("-" * 40)
config = get_config('config/clustering_config.yaml')
current_algo = config.get('classification', 'algorithm', 'kmeans')
print(f"  Aktueller Algorithmus: {current_algo}")

# ===== TEST 3: Synthetische Daten mit Noise =====
print("\n3. TEST-DATEN ERSTELLEN (mit Noise)")
print("-" * 40)
np.random.seed(42)

# 3 dichte Cluster
n_samples_per_cluster = 100
cluster_centers = [
    [0, 0],
    [5, 5],
    [-5, 5]
]

X_list = []
for center in cluster_centers:
    X_cluster = np.random.randn(n_samples_per_cluster, 2) * 0.5 + center
    X_list.append(X_cluster)

X_clusters = np.vstack(X_list)

# Noise hinzufügen (uniform verteilt)
n_noise = 30
X_noise = np.random.uniform(-10, 10, size=(n_noise, 2))

X = np.vstack([X_clusters, X_noise])

print(f"  Cluster-Samples: {len(X_clusters)}")
print(f"  Noise-Samples: {n_noise}")
print(f"  Total: {len(X)}")
print(f"  Features: {X.shape[1]}")

# ===== TEST 4: K-Means (Referenz) =====
print("\n4. K-MEANS CLUSTERING (Referenz)")
print("-" * 40)
kmeans_config = {'n_clusters': 3}
kmeans = ClustererFactory.create('kmeans', kmeans_config)
labels_kmeans = kmeans.fit_predict(X)
metrics_kmeans = kmeans.get_metrics(X, labels_kmeans)

print(f"  Algorithmus: {kmeans.get_algorithm_name()}")
print(f"  Cluster: {kmeans.get_n_clusters()}")
print(f"  Silhouette Score: {metrics_kmeans['silhouette']:.4f}")
print(f"  Davies-Bouldin: {metrics_kmeans['davies_bouldin']:.4f}")

# ===== TEST 5: DBSCAN mit Default-Parametern =====
print("\n5. DBSCAN mit DEFAULT-Parametern")
print("-" * 40)
dbscan_config = config.get('classification', 'dbscan', {})
print(f"  Config: eps={dbscan_config.get('eps')}, min_samples={dbscan_config.get('min_samples')}")

dbscan_default = ClustererFactory.create('dbscan', dbscan_config)
labels_dbscan_default = dbscan_default.fit_predict(X)
metrics_dbscan_default = dbscan_default.get_metrics(X, labels_dbscan_default)

print(f"  Algorithmus: {dbscan_default.get_algorithm_name()}")
print(f"  Cluster gefunden: {dbscan_default.get_n_clusters()}")
print(f"  Noise-Punkte: {dbscan_default.get_n_noise()}")
print(f"  Noise-Anteil: {metrics_dbscan_default['noise_percentage']:.1f}%")
if metrics_dbscan_default['silhouette'] != -999:
    print(f"  Silhouette Score: {metrics_dbscan_default['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {metrics_dbscan_default['davies_bouldin']:.4f}")
else:
    print(f"  Silhouette Score: N/A (zu wenige Cluster)")

# ===== TEST 6: DBSCAN mit optimierten Parametern =====
print("\n6. DBSCAN mit OPTIMIERTEN Parametern")
print("-" * 40)

# Parameter-Empfehlung holen
recommendations = dbscan_default.recommend_parameters(X)
print(f"  Empfohlenes eps: {recommendations['recommended_eps']:.3f}")
print(f"  Median k-distance: {recommendations['median_k_distance']:.3f}")

# Optimierte Parameter
dbscan_optimized_config = {
    'eps': 0.8,  # Angepasst für 2D-Daten
    'min_samples': 5
}

dbscan_optimized = ClustererFactory.create('dbscan', dbscan_optimized_config)
labels_dbscan_optimized = dbscan_optimized.fit_predict(X)
metrics_dbscan_optimized = dbscan_optimized.get_metrics(X, labels_dbscan_optimized)

print(f"  Verwendetes eps: {dbscan_optimized_config['eps']}")
print(f"  Cluster gefunden: {dbscan_optimized.get_n_clusters()}")
print(f"  Noise-Punkte: {dbscan_optimized.get_n_noise()}")
print(f"  Noise-Anteil: {metrics_dbscan_optimized['noise_percentage']:.1f}%")

if metrics_dbscan_optimized['silhouette'] != -999:
    print(f"  Silhouette Score: {metrics_dbscan_optimized['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {metrics_dbscan_optimized['davies_bouldin']:.4f}")
else:
    print(f"  Silhouette Score: N/A")

# Cluster-Verteilung
distribution = dbscan_optimized.get_cluster_distribution()
print(f"\n  Cluster-Verteilung:")
for cluster_id, count in distribution.items():
    print(f"    {cluster_id}: {count} Punkte")

# ===== TEST 7: Verschiedene eps-Werte testen =====
print("\n7. EPS-PARAMETER EXPERIMENT")
print("-" * 40)

eps_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
results = []

for eps in eps_values:
    dbscan_temp = DBSCANClusterer({'eps': eps, 'min_samples': 5})
    labels_temp = dbscan_temp.fit_predict(X)
    n_clusters = dbscan_temp.get_n_clusters()
    n_noise = dbscan_temp.get_n_noise()
    results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_pct': (n_noise / len(X)) * 100
    })

print(f"  {'eps':<6} {'Cluster':<10} {'Noise':<10} {'Noise %':<10}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
for r in results:
    print(f"  {r['eps']:<6.1f} {r['n_clusters']:<10} {r['n_noise']:<10} {r['noise_pct']:<10.1f}")

# ===== TEST 8: Mit DataFrame =====
print("\n8. INTEGRATION MIT DATAFRAME")
print("-" * 40)

df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['gvkey'] = range(len(df))

# Preprocessing
X_scaled, valid_idx = dbscan_optimized.preprocess_data(df, ['feature_1', 'feature_2'])
print(f"  Samples vorher: {len(df)}")
print(f"  Samples nachher: {len(valid_idx)}")
print(f"  ✓ Preprocessing erfolgreich")

# Clustering auf skalierte Daten
labels_scaled = dbscan_optimized.fit_predict(X_scaled)
print(f"  Cluster (nach Scaling): {dbscan_optimized.get_n_clusters()}")
print(f"  Noise (nach Scaling): {dbscan_optimized.get_n_noise()}")

# ===== TEST 9: Vergleich K-Means vs DBSCAN =====
print("\n9. VERGLEICH: K-MEANS vs DBSCAN")
print("-" * 40)
print(f"  {'Metrik':<25} {'K-Means':<15} {'DBSCAN':<15}")
print(f"  {'-'*25} {'-'*15} {'-'*15}")
print(f"  {'Cluster-Anzahl':<25} {kmeans.get_n_clusters():<15} {dbscan_optimized.get_n_clusters():<15}")
print(f"  {'Noise erkannt':<25} {'Nein':<15} {'Ja ({})'.format(dbscan_optimized.get_n_noise()):<15}")
print(f"  {'Silhouette Score':<25} {metrics_kmeans['silhouette']:<15.4f}", end="")
if metrics_dbscan_optimized['silhouette'] != -999:
    print(f" {metrics_dbscan_optimized['silhouette']:<15.4f}")
else:
    print(f" {'N/A':<15}")
print(f"  {'Deterministisch':<25} {'Nein':<15} {'Ja':<15}")

# ===== TEST 10: Visualisierung (optional) =====
print("\n10. VISUALISIERUNG")
print("-" * 40)

try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original Daten
    axes[0].scatter(X[:len(X_clusters), 0], X[:len(X_clusters), 1],
                    c='blue', alpha=0.6, s=50, label='Cluster')
    axes[0].scatter(X[len(X_clusters):, 0], X[len(X_clusters):, 1],
                    c='red', alpha=0.6, s=50, marker='x', label='Noise')
    axes[0].set_title('Original Daten (True Labels)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # K-Means
    scatter_kmeans = axes[1].scatter(X[:, 0], X[:, 1],
                                     c=labels_kmeans, cmap='viridis',
                                     alpha=0.6, s=50)
    axes[1].set_title(f'K-Means (k={kmeans.get_n_clusters()})')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter_kmeans, ax=axes[1])

    # DBSCAN
    # Noise als -1 darstellen
    scatter_dbscan = axes[2].scatter(X[:, 0], X[:, 1],
                                     c=labels_dbscan_optimized, cmap='viridis',
                                     alpha=0.6, s=50)
    # Noise-Punkte hervorheben
    noise_mask = labels_dbscan_optimized == -1
    if np.any(noise_mask):
        axes[2].scatter(X[noise_mask, 0], X[noise_mask, 1],
                       c='red', marker='x', s=100, alpha=0.8, label='Noise')
    axes[2].set_title(f'DBSCAN (k={dbscan_optimized.get_n_clusters()}, noise={dbscan_optimized.get_n_noise()})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter_dbscan, ax=axes[2])

    plt.tight_layout()
    plot_path = 'dbscan_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualisierung gespeichert: {plot_path}")
    plt.close()

except Exception as e:
    print(f"  ⚠ Visualisierung fehlgeschlagen: {e}")

# ===== ZUSAMMENFASSUNG =====
print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG")
print("=" * 80)
print("\n✅ DBSCAN Clustering erfolgreich implementiert!")
print("\nWICHTIGE ERKENNTNISSE:")
print("  • DBSCAN findet Cluster-Anzahl automatisch")
print("  • eps-Parameter ist kritisch und muss angepasst werden")
print("  • Noise/Outlier werden mit Label -1 markiert")
print("  • Deterministisch (kein random_state)")
print("  • Gut für Cluster mit beliebiger Form")
print("\nVERWENDUNG in config/clustering_config.yaml:")
print("  classification:")
print("    algorithm: 'dbscan'")
print("    dbscan:")
print("      eps: 0.8           # Muss angepasst werden!")
print("      min_samples: 5")
print("\nOUTPUT-STRUKTUR:")
print("  output/{market}_{timestamp}/")
print("  └── dbscan/")
print("      ├── clusters/              # Pro Cluster eine CSV")
print("      │   ├── cluster_0.csv")
print("      │   ├── cluster_1.csv")
print("      │   └── noise.csv          # Noise-Punkte separat!")
print("      └── reports/")
print("          └── data/")
print("              ├── cluster_assignment.csv  # Mit Noise-Spalte")
print("              └── cluster_profiles.csv")
print("\nEMPFEHLUNG:")
print("  • Beginne mit K-Means oder Hierarchical")
print("  • Nutze DBSCAN wenn:")
print("    - Ausreißer erwartet werden")
print("    - Cluster-Anzahl unbekannt")
print("    - Cluster nicht-konvex sind")
print("  • Experimentiere mit eps (0.3-2.0)")
print("  • Nutze recommend_parameters() für Hinweise")
print("=" * 80 + "\n")
