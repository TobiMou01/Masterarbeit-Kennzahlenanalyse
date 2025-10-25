# Hierarchical Clustering - Ward's Methode

## Übersicht

Hierarchical Clustering wurde als alternativer Clustering-Algorithmus implementiert. Du kannst jetzt in der Config zwischen **K-Means** und **Hierarchical Clustering** wählen.

## Verwendung

### 1. Algorithmus in Config wählen

Editiere `config/clustering_config.yaml`:

```yaml
classification:
  algorithm: 'hierarchical'  # Wechsel von 'kmeans' zu 'hierarchical'

  hierarchical:
    linkage: 'ward'              # Linkage-Methode
    distance_metric: 'euclidean' # Distanzmetrik (bei ward: euclidean)
```

### 2. Pipeline normal ausführen

```bash
python main.py --market germany
```

### 3. Output-Struktur

Output wird automatisch im richtigen Ordner gespeichert:

```
output/
└── germany_2024-01-15_143022/
    └── hierarchical/              # <- Neuer Ordner!
        ├── clusters/
        │   ├── cluster_0.csv
        │   ├── cluster_1.csv
        │   └── ...
        ├── reports/
        │   ├── data/
        │   │   ├── cluster_assignment.csv
        │   │   └── cluster_profiles.csv
        │   ├── models/
        │   │   ├── model_static.pkl
        │   │   └── scaler_static.pkl
        │   └── summary.txt
        └── visualizations/
            ├── static/
            ├── dynamic/
            └── combined/
```

## Funktionsweise

### Hierarchical Clustering (Ward's Methode)

- **Algorithmus**: Agglomerative (Bottom-Up) Clustering
- **Linkage**: Ward's Methode (minimiert Varianz innerhalb der Cluster)
- **Distanz**: Euklidische Distanz
- **Implementierung**: `sklearn.cluster.AgglomerativeClustering`

### Vorteile gegenüber K-Means

1. **Deterministisch**: Keine zufällige Initialisierung, immer gleiches Ergebnis
2. **Hierarchische Struktur**: Cluster-Hierarchie kann visualisiert werden (Dendrogram)
3. **Robustheit**: Weniger anfällig für lokale Minima
4. **Ward's Methode**: Minimiert Varianz, ähnlich zu K-Means Objective

### Nachteile

1. **Performance**: Langsamer als K-Means bei großen Datensätzen (O(n³) vs O(n))
2. **Speicher**: Benötigt mehr RAM für Distanz-Matrix
3. **Keine Reassignment**: Cluster können nicht nachträglich optimiert werden

## Vergleich K-Means vs. Hierarchical

| Aspekt | K-Means | Hierarchical (Ward) |
|--------|---------|---------------------|
| **Geschwindigkeit** | Schnell (O(n)) | Langsamer (O(n³)) |
| **Reproduzierbarkeit** | Random Init | Deterministisch |
| **Cluster-Form** | Rund/Konvex | Flexibler |
| **Hierarchie** | Nein | Ja (Dendrogram) |
| **Memory** | Niedrig | Höher |
| **Robustheit** | Lokale Minima | Robust |

## Metriken

Beide Algorithmen berechnen die gleichen Qualitätsmetriken:

- **Silhouette Score**: Cluster-Qualität (-1 bis 1, höher = besser)
- **Davies-Bouldin Index**: Cluster-Separation (niedriger = besser)
- **Anzahl Cluster**: Konfigurierbar

Zusätzlich bei Hierarchical:
- **Linkage**: ward, complete, average, single
- **Distance Metric**: euclidean (bei ward erforderlich)

## Kompatibilität

✅ **Vollständig kompatibel** mit bestehender Pipeline:

- Static Analysis
- Dynamic Analysis
- Combined Analysis
- Alle Visualisierungen
- Alle Reports
- Model-Speicherung

## Implementierung

### Neue Dateien

1. **`src/clustering/hierarchical_clusterer.py`**
   - `HierarchicalClusterer` Klasse
   - Implementiert `BaseClusterer` Interface
   - Ward's Linkage, Euklidische Distanz

2. **`config/clustering_config.yaml`**
   - Neue `hierarchical` Sektion
   - Konfigurierbare Parameter

3. **`src/clustering/factory.py`**
   - Registrierung: `'hierarchical': HierarchicalClusterer`

### Code-Beispiel

```python
from src.clustering import ClustererFactory

# Hierarchical Clusterer erstellen
config = {
    'n_clusters': 5,
    'linkage': 'ward',
    'distance_metric': 'euclidean'
}

clusterer = ClustererFactory.create('hierarchical', config)
labels = clusterer.fit_predict(X)
metrics = clusterer.get_metrics(X, labels)
```

## Test

Test-Skript ausführen:

```bash
python test_hierarchical.py
```

Output:
```
✅ Hierarchical Clustering erfolgreich implementiert!

VORTEILE:
  • Deterministisch (kein random_state benötigt)
  • Ward's Methode minimiert Varianz
  • Hierarchische Struktur verfügbar
  • Gleiche Metriken wie K-Means
```

## Empfehlung

**Wann K-Means verwenden?**
- Große Datensätze (> 10.000 Unternehmen)
- Schnelle Iteration gewünscht
- Runde Cluster erwartet

**Wann Hierarchical verwenden?**
- Reproduzierbarkeit wichtig
- Hierarchische Struktur interessant
- Kleinere/mittlere Datensätze (< 5.000 Unternehmen)
- Robuste Cluster-Bildung gewünscht

## Nächste Schritte

1. **Beide Algorithmen vergleichen**:
   - Führe Pipeline mit K-Means aus
   - Wechsle zu Hierarchical
   - Vergleiche Silhouette Scores

2. **Dendrogram-Visualisierung hinzufügen** (optional):
   - Nutze `get_dendrogram_linkage_matrix()` Methode
   - Visualisiere Cluster-Hierarchie mit `scipy.dendrogram`

3. **Weitere Algorithmen hinzufügen** (später):
   - DBSCAN für Density-Based Clustering
   - Gaussian Mixture für probabilistische Cluster
