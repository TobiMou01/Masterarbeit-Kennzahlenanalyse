# Clustering Algorithms - Komplette Übersicht

## Verfügbare Algorithmen

Die Pipeline unterstützt nun **3 Clustering-Algorithmen**:

1. **K-Means** - Partitioning-based
2. **Hierarchical** (Ward's Method) - Hierarchical
3. **DBSCAN** - Density-based

## Quick Start

### Algorithmus wählen

Editiere `config/clustering_config.yaml`:

```yaml
classification:
  algorithm: 'kmeans'  # Wähle: 'kmeans', 'hierarchical', oder 'dbscan'
```

### Pipeline ausführen

```bash
python main.py --market germany
```

## Algorithmen-Vergleich

### Übersichtstabelle

| Feature | K-Means | Hierarchical | DBSCAN |
|---------|---------|--------------|--------|
| **Cluster-Anzahl** | Vorgegeben (k) | Vorgegeben (k) | **Automatisch** |
| **Noise Detection** | ❌ Nein | ❌ Nein | ✅ **Ja** |
| **Cluster-Form** | Rund/Konvex | Flexibler | **Beliebig** |
| **Deterministisch** | ❌ Nein | ✅ Ja | ✅ Ja |
| **Geschwindigkeit** | ✅ **Sehr schnell** | ⚠️ Langsam | ⚠️ Mittel |
| **Skalierbarkeit** | ✅ Gut (>100k) | ⚠️ Schlecht | ⚠️ Mittel |
| **Parameter-Wahl** | ✅ **Einfach** | ✅ Einfach | ⚠️ **Schwierig** |
| **Outlier-Robust** | ❌ Nein | ❌ Nein | ✅ **Ja** |
| **High-Dimensional** | ✅ OK | ✅ OK | ⚠️ Problematisch |
| **Hierarchie** | ❌ Nein | ✅ **Ja** | ❌ Nein |

### Komplexität

| Algorithmus | Zeit-Komplexität | Speicher |
|-------------|------------------|----------|
| K-Means | O(n·k·i·d) | O(n·d) |
| Hierarchical | O(n²·log n) | O(n²) |
| DBSCAN | O(n·log n)* | O(n) |

*Mit räumlichem Index; sonst O(n²)

## Detaillierte Beschreibung

### 1. K-Means

**Typ**: Partitioning-based Clustering

**Wie es funktioniert**:
1. Wähle k zufällige Cluster-Zentren
2. Weise jeden Punkt dem nächsten Zentrum zu
3. Berechne neue Zentren als Mittelwert
4. Wiederhole bis Konvergenz

**Parameter**:
```yaml
kmeans:
  n_init: 20         # Anzahl Initialisierungen
  max_iter: 300      # Max. Iterationen
  algorithm: 'lloyd' # Algorithmus-Variante
```

**Vorteile**:
- ✅ Sehr schnell
- ✅ Einfach zu verstehen
- ✅ Gut für große Datensätze
- ✅ Einfache Parameter-Wahl

**Nachteile**:
- ❌ Cluster-Anzahl muss vorgegeben werden
- ❌ Anfällig für lokale Minima (random init)
- ❌ Nur runde/konvexe Cluster
- ❌ Sensitiv auf Outliers

**Wann verwenden**:
- Standard-Fall
- Große Datensätze
- Schnelle Iteration gewünscht
- Cluster-Anzahl bekannt

---

### 2. Hierarchical Clustering (Ward's Method)

**Typ**: Hierarchical Agglomerative Clustering

**Wie es funktioniert**:
1. Starte mit jedem Punkt als eigenem Cluster
2. Merge die zwei nächsten Cluster
3. Wiederhole bis k Cluster übrig
4. Ward's Methode minimiert Varianz

**Parameter**:
```yaml
hierarchical:
  linkage: 'ward'              # Linkage-Methode
  distance_metric: 'euclidean' # Distanzmetrik
```

**Vorteile**:
- ✅ Deterministisch (immer gleiches Ergebnis)
- ✅ Hierarchische Struktur (Dendrogram)
- ✅ Robuster als K-Means
- ✅ Flexiblere Cluster-Formen

**Nachteile**:
- ❌ Langsam (O(n²·log n))
- ❌ Hoher Speicherbedarf
- ❌ Cluster-Anzahl muss vorgegeben werden
- ❌ Keine Reassignment möglich

**Wann verwenden**:
- Reproduzierbarkeit wichtig
- Hierarchische Struktur interessant
- Kleinere/mittlere Datensätze (< 5.000)
- Robuste Cluster-Bildung gewünscht

---

### 3. DBSCAN

**Typ**: Density-Based Clustering

**Wie es funktioniert**:
1. Finde Core-Punkte (≥ min_samples im eps-Radius)
2. Verbinde Core-Punkte zu Clustern
3. Weise Border-Punkte zu
4. Markiere Rest als Noise (-1)

**Parameter**:
```yaml
dbscan:
  eps: 0.8           # Maximaler Radius (kritisch!)
  min_samples: 5     # Min. Punkte für Core-Punkt
  metric: 'euclidean'
```

**Vorteile**:
- ✅ **Findet Cluster-Anzahl automatisch**
- ✅ **Erkennt Outliers/Noise**
- ✅ Cluster mit beliebiger Form
- ✅ Deterministisch
- ✅ Robust gegen Outliers

**Nachteile**:
- ❌ **eps-Parameter schwer zu wählen**
- ❌ Probleme mit unterschiedlichen Dichten
- ❌ Nicht für High-Dimensional Data
- ❌ Kein Predict für neue Daten

**Wann verwenden**:
- Cluster-Anzahl unbekannt
- Ausreißer erwartet
- Nicht-konvexe Cluster
- Explorative Analyse

## Entscheidungshilfe

### Flowchart

```
START
  │
  ├─ Cluster-Anzahl bekannt?
  │  ├─ JA
  │  │  ├─ Große Daten (>10k)?
  │  │  │  ├─ JA → K-MEANS
  │  │  │  └─ NEIN
  │  │  │     ├─ Reproduzierbar?
  │  │  │     │  ├─ JA → HIERARCHICAL
  │  │  │     │  └─ NEIN → K-MEANS
  │  │
  │  └─ NEIN
  │     ├─ Outliers erwartet?
  │     │  ├─ JA → DBSCAN
  │     │  └─ NEIN → K-MEANS (teste verschiedene k)
```

### Szenario-basierte Empfehlung

**Szenario 1: Standard Financial Analysis**
- **Algorithmus**: K-Means
- **Grund**: Schnell, einfach, robust für financial data
- **Config**: `algorithm: 'kmeans'`, `n_clusters: 5`

**Szenario 2: Research Paper**
- **Algorithmus**: Hierarchical
- **Grund**: Deterministisch, reproduzierbar, hierarchische Struktur
- **Config**: `algorithm: 'hierarchical'`, `n_clusters: 5`

**Szenario 3: Outlier Detection**
- **Algorithmus**: DBSCAN
- **Grund**: Erkennt automatisch Ausreißer
- **Config**: `algorithm: 'dbscan'`, `eps: 0.8`

**Szenario 4: Explorative Analyse**
- **Algorithmus**: DBSCAN dann K-Means
- **Grund**: DBSCAN findet natürliche Gruppierungen, dann K-Means verfeinern
- **Workflow**:
  1. Nutze DBSCAN um Cluster-Anzahl zu schätzen
  2. Wechsle zu K-Means mit gefundener Anzahl

**Szenario 5: Große Datensätze (>50k)**
- **Algorithmus**: K-Means
- **Grund**: Einziger Algorithmus der skaliert
- **Config**: `algorithm: 'kmeans'`

## Best Practices

### 1. Beginne mit K-Means

Start immer mit K-Means als Baseline:
```bash
# Config: algorithm: 'kmeans'
python main.py --market germany
```

### 2. Teste verschiedene k-Werte

```yaml
static_analysis:
  n_clusters: 5  # Teste: 3, 5, 7, 10
```

### 3. Vergleiche Algorithmen

Führe Pipeline mit allen 3 aus:
```bash
# K-Means
vim config/clustering_config.yaml  # algorithm: 'kmeans'
python main.py --market germany

# Hierarchical
vim config/clustering_config.yaml  # algorithm: 'hierarchical'
python main.py --market germany

# DBSCAN
vim config/clustering_config.yaml  # algorithm: 'dbscan'
python main.py --market germany
```

Vergleiche:
- Silhouette Scores
- Cluster-Interpretierbarkeit
- Laufzeit

### 4. DBSCAN Parameter-Tuning

Wenn DBSCAN:
```python
from src.clustering import DBSCANClusterer

# Parameter-Empfehlung
clusterer = DBSCANClusterer({'eps': 0.5, 'min_samples': 5})
recommendations = clusterer.recommend_parameters(X)
print(f"Empfohlenes eps: {recommendations['recommended_eps']}")

# Experimentiere mit verschiedenen Werten
for eps in [0.3, 0.5, 0.8, 1.0]:
    # Teste...
```

### 5. Dokumentiere deine Wahl

Im Paper/Report:
```
Clustering-Algorithmus: K-Means
Begründung:
  - Große Datensätze (15.000 Unternehmen)
  - Klare Cluster-Struktur erwartet
  - Reproduzierbarkeit durch festen random_state
Parameter:
  - n_clusters: 5
  - n_init: 20
  - random_state: 42
Silhouette Score: 0.654
```

## Code-Beispiele

### Algorithmus-Vergleich im Code

```python
from src.clustering import ClustererFactory
import numpy as np

# Daten
X = np.random.randn(1000, 5)

# K-Means
kmeans = ClustererFactory.create('kmeans', {'n_clusters': 5})
labels_kmeans = kmeans.fit_predict(X)
metrics_kmeans = kmeans.get_metrics(X, labels_kmeans)

# Hierarchical
hierarchical = ClustererFactory.create('hierarchical', {'n_clusters': 5, 'linkage': 'ward'})
labels_hier = hierarchical.fit_predict(X)
metrics_hier = hierarchical.get_metrics(X, labels_hier)

# DBSCAN
dbscan = ClustererFactory.create('dbscan', {'eps': 0.8, 'min_samples': 5})
labels_dbscan = dbscan.fit_predict(X)
metrics_dbscan = dbscan.get_metrics(X, labels_dbscan)

# Vergleich
print(f"K-Means Silhouette: {metrics_kmeans['silhouette']:.4f}")
print(f"Hierarchical Silhouette: {metrics_hier['silhouette']:.4f}")
print(f"DBSCAN Silhouette: {metrics_dbscan['silhouette']:.4f}")
print(f"DBSCAN Noise: {metrics_dbscan['n_noise']}")
```

### Factory Pattern

```python
from src.clustering import ClustererFactory

# Verfügbare Algorithmen
algorithms = ClustererFactory.get_available_algorithms()
print(f"Verfügbar: {algorithms}")
# Output: ['kmeans', 'hierarchical', 'dbscan']

# Dynamisches Erstellen
for algo in algorithms:
    config = {'n_clusters': 5} if algo != 'dbscan' else {'eps': 0.8, 'min_samples': 5}
    clusterer = ClustererFactory.create(algo, config)
    print(f"{algo}: {clusterer.get_algorithm_name()}")
```

## Output-Struktur

Alle Algorithmen verwenden dieselbe Output-Struktur:

```
output/
└── {market}_{timestamp}/
    └── {algorithm}/           # kmeans / hierarchical / dbscan
        ├── clusters/
        │   ├── cluster_0.csv
        │   ├── cluster_1.csv
        │   ├── ...
        │   └── noise.csv      # Nur bei DBSCAN
        ├── reports/
        │   ├── data/
        │   │   ├── cluster_assignment.csv
        │   │   └── cluster_profiles.csv
        │   ├── analysis/
        │   ├── models/
        │   │   ├── model_{type}.pkl
        │   │   └── scaler_{type}.pkl
        │   └── summary.txt
        └── visualizations/
            ├── static/
            ├── dynamic/
            └── combined/
```

## Metriken

### Gemeinsame Metriken (alle Algorithmen)

- **Silhouette Score** (-1 bis 1, höher = besser)
- **Davies-Bouldin Index** (niedriger = besser)
- **Anzahl Cluster**

### Algorithmus-spezifische Metriken

**K-Means**:
- `inertia`: Summe der quadrierten Distanzen zu Zentren
- `n_iter`: Anzahl Iterationen bis Konvergenz

**Hierarchical**:
- `linkage`: Verwendete Linkage-Methode
- `n_leaves`: Anzahl Leaves im Dendrogram

**DBSCAN**:
- `n_noise`: Anzahl Noise-Punkte
- `noise_percentage`: Anteil Noise
- `n_core_samples`: Anzahl Core-Punkte

## Tests

Führe Tests aus um Algorithmen zu verstehen:

```bash
# K-Means + Hierarchical
python test_hierarchical.py

# DBSCAN
python test_dbscan.py
```

## Dokumentation

- **K-Means**: Bereits in Pipeline vorhanden
- **Hierarchical**: [HIERARCHICAL_CLUSTERING.md](HIERARCHICAL_CLUSTERING.md)
- **DBSCAN**: [DBSCAN_CLUSTERING.md](DBSCAN_CLUSTERING.md)

## Troubleshooting

### Problem: Welchen Algorithmus wählen?

**Antwort**: Start mit K-Means, wechsle wenn:
- Reproduzierbarkeit wichtig → Hierarchical
- Outliers problematisch → DBSCAN
- Cluster-Anzahl unklar → DBSCAN

### Problem: Schlechte Cluster-Qualität

**Lösungen**:
1. Teste verschiedene k-Werte
2. Wechsle Algorithmus
3. Prüfe Feature-Auswahl
4. Prüfe Datenskalierung

### Problem: DBSCAN findet zu viele/wenige Cluster

**Lösung**: Siehe [DBSCAN_CLUSTERING.md](DBSCAN_CLUSTERING.md) → Parameter-Tuning

### Problem: Unterschiedliche Ergebnisse

**Erklärung**:
- K-Means: Random initialization → `random_state: 42` setzen
- Hierarchical: Immer gleich (deterministisch)
- DBSCAN: Immer gleich (deterministisch)

## Zusammenfassung

Du hast jetzt **3 robuste Clustering-Algorithmen** zur Verfügung:

1. **K-Means** - Standard, schnell, einfach
2. **Hierarchical** - Deterministisch, hierarchisch, robust
3. **DBSCAN** - Automatisch, Noise-Detection, flexibel

**Empfehlung**: Start mit K-Means, experimentiere mit allen drei, wähle den besten für deine Daten.

**Next Steps**:
1. Führe Tests aus
2. Vergleiche Algorithmen auf deinen Daten
3. Dokumentiere Ergebnisse
4. Wähle finalen Algorithmus für Paper
