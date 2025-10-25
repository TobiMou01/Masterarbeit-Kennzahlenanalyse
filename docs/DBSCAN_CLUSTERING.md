# DBSCAN Clustering - Density-Based Spatial Clustering

## Übersicht

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) wurde als dritter Clustering-Algorithmus implementiert. Im Gegensatz zu K-Means und Hierarchical Clustering findet DBSCAN die **Cluster-Anzahl automatisch** und **erkennt Ausreißer/Noise**.

## Verwendung

### 1. Algorithmus in Config wählen

Editiere `config/clustering_config.yaml`:

```yaml
classification:
  algorithm: 'dbscan'  # Von 'kmeans'/'hierarchical' zu 'dbscan'

  dbscan:
    eps: 0.8           # Kritischer Parameter! Muss angepasst werden
    min_samples: 5     # Minimale Punkte für Core-Punkt
    metric: 'euclidean'  # Distanzmetrik
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
    └── dbscan/                    # <- DBSCAN Ordner
        ├── clusters/
        │   ├── cluster_0.csv      # Cluster 0
        │   ├── cluster_1.csv      # Cluster 1
        │   ├── cluster_N.csv      # Cluster N (automatisch!)
        │   └── noise.csv          # Noise/Outliers (Label -1)
        ├── reports/
        │   ├── data/
        │   │   ├── cluster_assignment.csv  # Mit Noise-Spalte
        │   │   └── cluster_profiles.csv
        │   └── summary.txt        # Enthält Noise-Statistiken
        └── visualizations/
```

## Funktionsweise

### DBSCAN Algorithmus

- **Typ**: Density-Based Clustering
- **Parameter**:
  - `eps`: Maximaler Radius um einen Punkt
  - `min_samples`: Minimale Anzahl Punkte im eps-Radius für Core-Punkt
- **Besonderheiten**:
  - Findet Cluster-Anzahl automatisch
  - Markiert Noise mit Label `-1`
  - Deterministisch (kein random_state)
  - Cluster können beliebige Form haben

### Core-, Border- und Noise-Punkte

1. **Core-Punkt**: Mindestens `min_samples` Punkte im eps-Radius
2. **Border-Punkt**: Im eps-Radius eines Core-Punkts, aber selbst kein Core-Punkt
3. **Noise-Punkt**: Weder Core noch Border → Label `-1`

## Vorteile & Nachteile

### Vorteile gegenüber K-Means/Hierarchical

1. ✅ **Automatische Cluster-Anzahl**: Keine Vorgabe nötig
2. ✅ **Noise Detection**: Erkennt Ausreißer automatisch
3. ✅ **Beliebige Cluster-Form**: Nicht nur konvexe Cluster
4. ✅ **Deterministisch**: Immer gleiches Ergebnis
5. ✅ **Robust gegen Outliers**: Noise wird separiert

### Nachteile

1. ❌ **Parameter-Wahl schwierig**: `eps` ist kritisch und datenabhängig
2. ❌ **Nicht für High-Dimensional Data**: Curse of Dimensionality
3. ❌ **Unterschiedliche Dichten**: Probleme bei variierenden Cluster-Dichten
4. ❌ **Performance**: O(n²) im worst case (mit Index: O(n log n))
5. ❌ **Kein Predict für neue Daten**: Clustering nur für Trainingsdaten

## Parameter-Wahl

### Kritischer Parameter: `eps`

`eps` ist der wichtigste Parameter und **muss an die Daten angepasst werden**.

#### Empfohlene Vorgehensweise:

1. **k-distance Graph**:
   ```python
   from src.clustering import DBSCANClusterer

   clusterer = DBSCANClusterer({'eps': 0.5, 'min_samples': 5})
   recommendations = clusterer.recommend_parameters(X)
   print(f"Empfohlenes eps: {recommendations['recommended_eps']}")
   ```

2. **Experimentieren**:
   - Beginne mit `eps = 0.5` (Standard)
   - Zu wenige Cluster? → Erhöhe `eps`
   - Zu viele Cluster? → Reduziere `eps`
   - Zu viel Noise? → Erhöhe `eps` oder reduziere `min_samples`
   - Zu wenig Noise? → Reduziere `eps` oder erhöhe `min_samples`

3. **Typische Werte**:
   - Gut skalierte Daten: `eps = 0.3 - 2.0`
   - Mehr Features: Höheres `eps` nötig
   - Dichtere Cluster: Niedrigeres `eps`

### Parameter: `min_samples`

- **Standard**: 5 (gut für die meisten Fälle)
- **Faustregel**: `min_samples ≥ Dimensionen + 1`
- **Mehr Dimensions**: Höheres `min_samples`
- **Weniger Noise**: Höheres `min_samples`

## Vergleich der Algorithmen

| Aspekt | K-Means | Hierarchical | **DBSCAN** |
|--------|---------|--------------|------------|
| **Cluster-Anzahl** | Vorgegeben | Vorgegeben | **Automatisch** |
| **Noise Detection** | Nein | Nein | **Ja (-1)** |
| **Cluster-Form** | Rund/Konvex | Flexibler | **Beliebig** |
| **Deterministisch** | Nein | Ja | **Ja** |
| **Hauptparameter** | k | k, linkage | **eps, min_samples** |
| **Parameter-Wahl** | Einfach | Einfach | **Schwierig** |
| **Geschwindigkeit** | Schnell | Langsam | Mittel |
| **High-Dimensional** | OK | OK | **Problematisch** |
| **Outlier-Robust** | Nein | Nein | **Ja** |

## Metriken

DBSCAN berechnet spezielle Metriken:

### Standard-Metriken (ohne Noise)

- **Silhouette Score**: Nur für Non-Noise-Punkte
- **Davies-Bouldin Index**: Nur für Non-Noise-Punkte

### DBSCAN-spezifische Metriken

- **n_clusters**: Anzahl gefundener Cluster (automatisch)
- **n_noise**: Anzahl Noise-Punkte
- **noise_percentage**: Anteil Noise an allen Punkten
- **n_core_samples**: Anzahl Core-Punkte
- **core_sample_percentage**: Anteil Core-Punkte

### Interpretation

```
Beispiel-Output:
  DBSCAN: 5 Cluster, 23 Noise (7.0%)
  Silhouette Score: 0.8774

Interpretation:
  ✓ 5 Cluster automatisch gefunden
  ✓ 7% der Unternehmen als Outliers identifiziert
  ✓ Hoher Silhouette Score → Gute Cluster-Qualität
```

## Kompatibilität

✅ **Vollständig kompatibel** mit bestehender Pipeline:

- Static Analysis
- Dynamic Analysis
- Combined Analysis
- Alle Visualisierungen
- Alle Reports
- Model-Speicherung

⚠️ **Besonderheiten**:

- Noise-Punkte (Label -1) werden separat behandelt
- `cluster_assignment.csv` enthält Noise-Spalte
- `noise.csv` enthält alle Outliers

## Implementierung

### Neue Dateien

1. **`src/clustering/dbscan_clusterer.py`**
   - `DBSCANClusterer` Klasse
   - Implementiert `BaseClusterer` Interface
   - Noise Detection mit Label -1
   - Parameter-Empfehlung mit `recommend_parameters()`

2. **`config/clustering_config.yaml`**
   - Neue `dbscan` Sektion
   - `eps`, `min_samples`, `metric` konfigurierbar

3. **`src/clustering/factory.py`**
   - Registrierung: `'dbscan': DBSCANClusterer`

### Code-Beispiel

```python
from src.clustering import ClustererFactory

# DBSCAN Clusterer erstellen
config = {
    'eps': 0.8,
    'min_samples': 5,
    'metric': 'euclidean'
}

clusterer = ClustererFactory.create('dbscan', config)
labels = clusterer.fit_predict(X)  # Label -1 = Noise

# Metriken
metrics = clusterer.get_metrics(X, labels)
print(f"Cluster: {metrics['n_clusters']}")
print(f"Noise: {metrics['n_noise']} ({metrics['noise_percentage']:.1f}%)")

# Noise-Maske
noise_mask = clusterer.get_noise_mask()
X_noise = X[noise_mask]

# Cluster-Verteilung
distribution = clusterer.get_cluster_distribution()
# {'Noise': 23, 'Cluster_0': 150, 'Cluster_1': 127, ...}

# Parameter-Empfehlung
recommendations = clusterer.recommend_parameters(X)
print(f"Empfohlenes eps: {recommendations['recommended_eps']:.3f}")
```

## Test

Test-Skript ausführen:

```bash
python test_dbscan.py
```

Output:
```
✅ DBSCAN Clustering erfolgreich implementiert!

WICHTIGE ERKENNTNISSE:
  • DBSCAN findet Cluster-Anzahl automatisch
  • eps-Parameter ist kritisch und muss angepasst werden
  • Noise/Outlier werden mit Label -1 markiert
  • Deterministisch (kein random_state)
  • Gut für Cluster mit beliebiger Form
```

## Wann DBSCAN verwenden?

### ✅ DBSCAN ist gut wenn:

1. **Cluster-Anzahl unbekannt**: Du weißt nicht wie viele Cluster
2. **Ausreißer erwartet**: Daten enthalten Noise/Outliers
3. **Nicht-konvexe Cluster**: Cluster haben komplexe Formen
4. **Robustheit wichtig**: Outliers sollen Clustering nicht beeinflussen
5. **Explorative Analyse**: Du möchtest natürliche Gruppierungen finden

### ❌ DBSCAN ist schlecht wenn:

1. **High-Dimensional Data**: Viele Features (> 10-15)
2. **Unterschiedliche Dichten**: Cluster haben stark variierende Dichten
3. **Parameter unklar**: Du kannst `eps` nicht sinnvoll wählen
4. **Feste Cluster-Anzahl gewünscht**: Du willst exakt k Cluster
5. **Große Datensätze**: > 100.000 Unternehmen (Performance)

## Praktische Empfehlung

### Workflow für Financial Data

1. **Start**: Beginne mit K-Means oder Hierarchical
2. **Analyse**: Prüfe Outliers in den Daten
3. **DBSCAN Test**: Wenn Outliers problematisch:
   ```yaml
   classification:
     algorithm: 'dbscan'
     dbscan:
       eps: 0.5      # Start-Wert
       min_samples: 5
   ```
4. **Parameter-Tuning**:
   - Prüfe Anzahl gefundener Cluster
   - Prüfe Noise-Anteil
   - Adjustiere `eps` wenn nötig
5. **Vergleich**: Vergleiche mit K-Means/Hierarchical
6. **Entscheidung**: Wähle besten Algorithmus für deine Daten

### Beispiel-Szenarien

**Szenario 1: Standard-Analyse**
```yaml
algorithm: 'kmeans'  # Start mit K-Means (schnell, einfach)
```

**Szenario 2: Outlier-Problem**
```yaml
algorithm: 'dbscan'  # Wenn Outliers stören
dbscan:
  eps: 0.8
  min_samples: 5
```

**Szenario 3: Reproduzierbarkeit**
```yaml
algorithm: 'hierarchical'  # Deterministisch, robust
```

**Szenario 4: Automatische Cluster-Anzahl**
```yaml
algorithm: 'dbscan'  # Cluster-Anzahl unbekannt
```

## Troubleshooting

### Problem: Zu viele Cluster (> 20)

**Lösung**:
- Erhöhe `eps` (z.B. von 0.5 → 0.8 → 1.0)
- Erhöhe `min_samples` (z.B. von 5 → 10)

### Problem: Keine Cluster gefunden (nur Noise)

**Lösung**:
- Erhöhe `eps` deutlich (z.B. 0.5 → 2.0)
- Reduziere `min_samples` (z.B. 10 → 5 → 3)

### Problem: Zu viel Noise (> 30%)

**Lösung**:
- Erhöhe `eps`
- Reduziere `min_samples`
- Prüfe ob Daten gut skaliert sind

### Problem: Silhouette Score = N/A

**Grund**: Zu wenige Cluster oder zu viele Noise-Punkte

**Lösung**:
- Adjustiere Parameter bis mindestens 2 Cluster gefunden werden
- Prüfe mit `recommend_parameters()` für Hinweise

## Nächste Schritte

1. **Teste DBSCAN**:
   ```bash
   python test_dbscan.py
   ```

2. **Vergleiche Algorithmen**:
   - Führe Pipeline mit allen 3 Algorithmen aus
   - Vergleiche Silhouette Scores
   - Prüfe Noise-Anteil bei DBSCAN

3. **Visualisierung** (optional):
   - Nutze `dbscan_comparison.png` aus Test
   - Erstelle eigene Visualisierungen

4. **Parameter-Tuning**:
   - Experimentiere mit verschiedenen `eps`-Werten
   - Nutze `recommend_parameters()` als Startpunkt

5. **Dokumentiere Ergebnisse**:
   - Welcher Algorithmus funktioniert am besten?
   - Wie viele Outliers wurden gefunden?
   - Sind die Cluster interpretierbar?
