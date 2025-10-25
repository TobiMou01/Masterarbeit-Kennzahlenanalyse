"""
Test-Skript für Hierarchical Clustering
Demonstriert Verwendung des neuen Algorithmus
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Module
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.clustering import HierarchicalClusterer, KMeansClusterer, ClustererFactory
from src.config_loader import get_config

print("\n" + "=" * 80)
print("HIERARCHICAL CLUSTERING TEST")
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

# ===== TEST 3: Synthetische Daten erstellen =====
print("\n3. TEST-DATEN ERSTELLEN")
print("-" * 40)
np.random.seed(42)

# 3 Cluster mit unterschiedlichen Zentren
n_samples = 300
cluster_centers = [
    [0, 0, 0, 0, 0],      # Cluster 1: Durchschnitt
    [3, 3, 3, 3, 3],      # Cluster 2: Hoch
    [-3, -3, -3, -3, -3]  # Cluster 3: Niedrig
]

X_list = []
for center in cluster_centers:
    X_cluster = np.random.randn(n_samples // 3, 5) + center
    X_list.append(X_cluster)

X = np.vstack(X_list)
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Echte Cluster: {len(cluster_centers)}")

# ===== TEST 4: K-Means Clustering =====
print("\n4. K-MEANS CLUSTERING")
print("-" * 40)
kmeans_config = config.get('classification', 'kmeans', {})
kmeans_config['n_clusters'] = 3

kmeans = ClustererFactory.create('kmeans', kmeans_config)
labels_kmeans = kmeans.fit_predict(X)
metrics_kmeans = kmeans.get_metrics(X, labels_kmeans)

print(f"  Algorithmus: {kmeans.get_algorithm_name()}")
print(f"  Silhouette Score: {metrics_kmeans['silhouette']:.4f}")
print(f"  Davies-Bouldin: {metrics_kmeans['davies_bouldin']:.4f}")
print(f"  Inertia: {metrics_kmeans['inertia']:.2f}")

# ===== TEST 5: Hierarchical Clustering =====
print("\n5. HIERARCHICAL CLUSTERING (Ward's Methode)")
print("-" * 40)
hierarchical_config = config.get('classification', 'hierarchical', {})
hierarchical_config['n_clusters'] = 3

hierarchical = ClustererFactory.create('hierarchical', hierarchical_config)
labels_hierarchical = hierarchical.fit_predict(X)
metrics_hierarchical = hierarchical.get_metrics(X, labels_hierarchical)

print(f"  Algorithmus: {hierarchical.get_algorithm_name()}")
print(f"  Silhouette Score: {metrics_hierarchical['silhouette']:.4f}")
print(f"  Davies-Bouldin: {metrics_hierarchical['davies_bouldin']:.4f}")
print(f"  Linkage: {metrics_hierarchical['linkage']}")
print(f"  Distanz-Metrik: {metrics_hierarchical['distance_metric']}")

# ===== TEST 6: Vergleich =====
print("\n6. VERGLEICH")
print("-" * 40)
print(f"  K-Means Silhouette:      {metrics_kmeans['silhouette']:.4f}")
print(f"  Hierarchical Silhouette: {metrics_hierarchical['silhouette']:.4f}")
print(f"  Differenz:               {abs(metrics_kmeans['silhouette'] - metrics_hierarchical['silhouette']):.4f}")

better = "K-Means" if metrics_kmeans['silhouette'] > metrics_hierarchical['silhouette'] else "Hierarchical"
print(f"\n  → Besserer Silhouette Score: {better}")

# ===== TEST 7: Mit DataFrame =====
print("\n7. INTEGRATION MIT DATAFRAME")
print("-" * 40)

# Erstelle DataFrame
df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
df['gvkey'] = range(len(df))

# Preprocessing
X_scaled, valid_idx = hierarchical.preprocess_data(df, df.columns[:-1].tolist())
print(f"  Samples vorher: {len(df)}")
print(f"  Samples nachher: {len(valid_idx)}")
print(f"  ✓ Preprocessing erfolgreich")

# ===== ZUSAMMENFASSUNG =====
print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG")
print("=" * 80)
print("\n✅ Hierarchical Clustering erfolgreich implementiert!")
print("\nVERWENDUNG in config/clustering_config.yaml:")
print("  classification:")
print("    algorithm: 'hierarchical'  # Wechsel von 'kmeans' zu 'hierarchical'")
print("\nOUTPUT-STRUKTUR:")
print("  output/")
print("  └── {market}_{timestamp}/")
print("      └── hierarchical/           # Neuer Ordner für Hierarchical")
print("          ├── clusters/")
print("          ├── reports/")
print("          └── visualizations/")
print("\nVORTEILE:")
print("  • Deterministisch (kein random_state benötigt)")
print("  • Ward's Methode minimiert Varianz")
print("  • Hierarchische Struktur verfügbar")
print("  • Gleiche Metriken wie K-Means")
print("=" * 80 + "\n")
