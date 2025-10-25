# Clustering Module

Modulare und erweiterbare Clustering-Implementierung für die Kennzahlenanalyse.

## Übersicht

Dieses Modul implementiert eine flexible Architektur für Clustering-Algorithmen basierend auf dem **Factory Pattern** und **Abstract Base Classes**.

### Aktuell verfügbare Algorithmen:
- **K-Means** (`kmeans`)

### Geplant für später:
- Hierarchical Clustering (`hierarchical`)
- DBSCAN (`dbscan`)
- Gaussian Mixture Models (`gaussian_mixture`)
- und weitere...

## Architektur

```
src/clustering/
├── __init__.py           # Modul-Exports
├── base.py               # Abstract Base Class (BaseClusterer)
├── factory.py            # Factory Pattern für Clusterer-Erstellung
├── kmeans_clusterer.py   # K-Means Implementierung
└── README.md             # Diese Datei
```

## Verwendung

### 1. Einfache Verwendung mit Factory

```python
from src.clustering.factory import ClustererFactory

# Clusterer erstellen
clusterer = ClustererFactory.create(
    algorithm='kmeans',
    config={'n_clusters': 5},
    random_state=42
)

# Clustering durchführen
X_scaled, valid_idx = clusterer.preprocess_data(df, features)
labels = clusterer.fit_predict(X_scaled)

# Metriken abrufen
metrics = clusterer.get_metrics(X_scaled, labels)
```

### 2. Konfiguration über YAML

```yaml
classification:
  algorithm: 'kmeans'  # Algorithmus auswählen

  kmeans:
    n_clusters: 5      # Wird durch Analyse-Config überschrieben
    n_init: 20
    max_iter: 300
```

### 3. In der Hauptpipeline

Die ClusteringEngine verwendet automatisch den konfigurierten Algorithmus:

```python
engine = ClusteringEngine(config_path='config/clustering_config.yaml')
df_result, profiles, metrics = engine.perform_clustering(
    df, features, n_clusters=5, analysis_type='static'
)
```

## Neuen Algorithmus hinzufügen

### Schritt 1: Klasse erstellen

Erstelle eine neue Datei `src/clustering/my_algorithm_clusterer.py`:

```python
from src.clustering.base import BaseClusterer
import numpy as np
from typing import Dict

class MyAlgorithmClusterer(BaseClusterer):

    def __init__(self, config: Dict, random_state: int = 42):
        super().__init__(config, random_state)
        # Deine Parameter initialisieren
        self.param1 = config.get('param1', default_value)

    def fit(self, X: np.ndarray) -> 'MyAlgorithmClusterer':
        # Implementiere Training
        self.model = YourModel(...)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Implementiere Vorhersage
        return self.model.predict(X)

    def get_n_clusters(self) -> int:
        # Gib Anzahl Cluster zurück
        return self.n_clusters

    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        # Berechne Metriken
        from sklearn.metrics import silhouette_score
        return {
            'silhouette': silhouette_score(X, labels),
            # ... weitere Metriken
        }

    def get_algorithm_name(self) -> str:
        return 'my_algorithm'
```

### Schritt 2: In Factory registrieren

In `src/clustering/factory.py`:

```python
from src.clustering.my_algorithm_clusterer import MyAlgorithmClusterer

class ClustererFactory:
    _registry: Dict[str, Type[BaseClusterer]] = {
        'kmeans': KMeansClusterer,
        'my_algorithm': MyAlgorithmClusterer,  # Hinzufügen
    }
```

### Schritt 3: Config erweitern

In `config/clustering_config.yaml`:

```yaml
classification:
  algorithm: 'my_algorithm'  # Neuen Algorithmus nutzen

  my_algorithm:
    param1: value1
    param2: value2
```

### Schritt 4: Exports aktualisieren (optional)

In `src/clustering/__init__.py`:

```python
from src.clustering.my_algorithm_clusterer import MyAlgorithmClusterer

__all__ = [
    'BaseClusterer',
    'ClustererFactory',
    'KMeansClusterer',
    'MyAlgorithmClusterer',  # Hinzufügen
]
```

**Das war's!** Keine Änderungen in der Hauptpipeline nötig.

## BaseClusterer Interface

Alle Clusterer müssen folgende Methoden implementieren:

### Pflicht-Methoden

| Methode | Beschreibung | Return |
|---------|--------------|--------|
| `fit(X)` | Trainiert den Algorithmus | `self` |
| `predict(X)` | Weist Cluster zu | `np.ndarray` |
| `get_n_clusters()` | Anzahl Cluster | `int` |
| `get_metrics(X, labels)` | Qualitätsmetriken | `Dict` |
| `get_algorithm_name()` | Algorithmus-Name | `str` |

### Optional überschreibbar

| Methode | Beschreibung | Standard |
|---------|--------------|----------|
| `preprocess_data(df, features)` | Daten-Preprocessing | Outlier-Removal, Scaling |
| `get_algorithm_params()` | Parameter-Dictionary | `self.config` |
| `fit_predict(X)` | Kombiniert fit + predict | `fit()` + `predict()` |

## Output-Struktur

Jeder Algorithmus speichert seine Ergebnisse in einheitlichem Format:

```
output/germany/
├── clusters/                     # Cluster-spezifische Listen
│   ├── 0_high_performers.csv
│   └── ...
├── reports/
│   ├── data/
│   │   ├── static_assignments.csv    # Cluster-Zuordnungen
│   │   ├── static_profiles.csv       # Cluster-Profile
│   │   └── static_metrics.json       # Metriken + Algorithmus-Info
│   ├── analysis/
│   └── models/
│       ├── static_scaler.pkl
│       └── static_model.pkl
└── visualizations/
```

## Metriken-Format

Die `*_metrics.json` Dateien enthalten:

```json
{
  "algorithm": "kmeans",
  "n_clusters": 5,
  "n_companies": 158,
  "silhouette": 0.198,
  "davies_bouldin": 1.141,
  // Algorithmus-spezifische Metriken:
  "inertia": 510.18,    // nur K-Means
  "n_iter": 8           // nur K-Means
}
```

## Best Practices

### 1. Reproduzierbarkeit
Verwende immer `random_state` für deterministisches Verhalten.

### 2. Preprocessing
Die Standard-Methode `preprocess_data()` führt aus:
- Outlier-Removal (0.1% und 99.9% Quantile)
- Missing Value Imputation (Median)
- StandardScaler Normalisierung

Überschreibe diese nur wenn dein Algorithmus spezielle Anforderungen hat.

### 3. Metriken
Implementiere mindestens `silhouette_score` und `davies_bouldin_score` für Vergleichbarkeit.

### 4. Fehlerbehandlung
Nutze die Factory für saubere Fehlerbehandlung bei unbekannten Algorithmen.

## Beispiel: Vollständiger Workflow

```python
# 1. Verfügbare Algorithmen anzeigen
from src.clustering.factory import ClustererFactory
print(ClustererFactory.get_available_algorithms())
# Output: ['kmeans']

# 2. Prüfen ob Algorithmus verfügbar
if ClustererFactory.is_available('kmeans'):
    print("K-Means ist verfügbar!")

# 3. Clusterer erstellen und verwenden
clusterer = ClustererFactory.create(
    algorithm='kmeans',
    config={'n_clusters': 5, 'n_init': 20},
    random_state=42
)

# 4. Clustering durchführen
X_scaled, valid_idx = clusterer.preprocess_data(df, features)
labels = clusterer.fit_predict(X_scaled)

# 5. Metriken und Modell abrufen
metrics = clusterer.get_metrics(X_scaled, labels)
model = clusterer.get_model()
scaler = clusterer.get_scaler()

print(f"Silhouette Score: {metrics['silhouette']:.3f}")
print(f"Algorithmus: {clusterer.get_algorithm_name()}")
```

## Erweiterbarkeit

Die Architektur ist bewusst offen gestaltet:

- **Neue Algorithmen**: Einfach neue Klasse + Factory-Registration
- **Custom Preprocessing**: Überschreibe `preprocess_data()`
- **Zusätzliche Metriken**: Erweitere `get_metrics()`
- **Dynamische Registration**: Nutze `ClustererFactory.register()` zur Laufzeit

## Support

Bei Fragen oder Problemen:
1. Prüfe die Logs: Logger gibt detaillierte Infos
2. Nutze die `get_algorithm_params()` Methode für Debugging
3. Teste Algorithmen einzeln vor Integration in Pipeline
