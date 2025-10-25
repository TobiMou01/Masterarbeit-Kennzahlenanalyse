# 🏗️ Code Architecture & Execution Flow

## 📂 Projekt-Struktur

```
masterarbeit-kennzahlenanalyse/
├── main.py                          # Entry Point
├── config.yaml                      # Konfiguration
│
├── src/
│   ├── core/                        # SCHRITT 1-3: Kern-Module (in Reihenfolge)
│   │   ├── config.py                # 1️⃣ Konfiguration laden
│   │   ├── data_loader.py           # 2️⃣ Rohdaten laden & bereinigen
│   │   ├── feature_engineering.py   # 3️⃣ Kennzahlen berechnen
│   │   └── preprocessing.py         # 2️⃣ Preprocessing koordinieren
│   │
│   ├── clustering/                  # SCHRITT 4: Clustering-Algorithmen
│   │   ├── base.py                  # 4️⃣ Basis-Klasse
│   │   ├── factory.py               # 4️⃣ Factory Pattern
│   │   ├── kmeans_clusterer.py      # 4️⃣ K-Means Implementation
│   │   ├── hierarchical_clusterer.py# 4️⃣ Hierarchical Implementation
│   │   └── dbscan_clusterer.py      # 4️⃣ DBSCAN Implementation
│   │
│   ├── analysis/                    # SCHRITT 5: Analyse-Engine
│   │   ├── clustering_engine.py     # 5️⃣ Clustering-Logik
│   │   └── pipeline.py              # 6️⃣ 3-Stage Pipeline
│   │
│   ├── comparison/                  # SCHRITT 7: Vergleiche
│   │   ├── comparison_pipeline.py   # 7️⃣ Comparison Orchestrator
│   │   ├── gics_comparison.py       # 7️⃣ GICS-Vergleich
│   │   ├── algorithm_comparison.py  # 7️⃣ Algorithmen-Vergleich
│   │   ├── feature_importance.py    # 7️⃣ Feature Importance
│   │   └── temporal_stability.py    # 7️⃣ Zeitliche Stabilität
│   │
│   └── output/                      # SCHRITT 8: Output-Verwaltung
│       ├── output_handler.py        # 8️⃣ Einzelne Ergebnisse
│       ├── comparison_handler.py    # 8️⃣ Vergleichs-Ergebnisse
│       └── visualizer.py            # 8️⃣ Visualisierungen
│
├── data/
│   ├── raw/{market}/                # Input-Daten
│   └── processed/{market}/          # Verarbeitete Daten
│
└── output/{market}/
    ├── algorithms/                  # Einzelne Algorithmen
    │   ├── kmeans/
    │   ├── hierarchical/
    │   └── dbscan/
    └── comparisons/                 # Vergleiche
        ├── 01_gics_comparison/
        ├── 02_algorithm_comparison/
        ├── 03_feature_importance/
        └── 04_temporal_stability/
```

---

## 🔄 Execution Flow (Detailliert)

### **Mode 1: Single Algorithm**
```python
python src/main.py --market germany
```

**Execution Path:**
```
main.py
  ↓
1️⃣ core/config.py
   → load_config('config.yaml')
   → get_value(...) für Settings
  ↓
2️⃣ core/preprocessing.py
   → run_preprocessing()
     ↓
     2a. core/data_loader.py
         → load_all_csv_from_directory()
         → clean_data()
         → filter_relevant_columns()
     ↓
     2b. core/feature_engineering.py
         → create_all_features()
           - Profitabilität (ROA, ROE, etc.)
           - Liquidität (Current Ratio, etc.)
           - Verschuldung (Debt-to-Equity, etc.)
           - Effizienz & Wachstum
  ↓
3️⃣ analysis/pipeline.py (ClusteringPipeline)
   → run_analysis()
     ↓
     4️⃣ analysis/clustering_engine.py
        → perform_clustering()
          ↓
          4a. clustering/factory.py
              → create(algorithm='kmeans')
          ↓
          4b. clustering/{algorithm}_clusterer.py
              → preprocess_data()
              → fit_predict()
              → get_metrics()
     ↓
     5️⃣ 3-Stage Analysis:
        → _run_static_analysis()    # Querschnitt aktuelles Jahr
        → _run_dynamic_analysis()   # Zeitreihen-Trends
        → _run_combined_analysis()  # Kombiniert Static+Dynamic
  ↓
6️⃣ output/output_handler.py
   → save_cluster_data()
   → save_cluster_lists()
   → save_models()
  ↓
7️⃣ output/visualizer.py
   → create_all_plots()
     - Cluster Scatter
     - Distribution
     - Feature Distributions
     - Heatmaps
```

### **Mode 2: Comparison Analysis**
```python
python src/main.py --market germany --compare
```

**Execution Path:**
```
main.py
  ↓
1️⃣-2️⃣ [Gleich wie Single Mode]
  ↓
3️⃣ comparison/comparison_pipeline.py (ComparisonPipeline)
   → run_full_comparison_pipeline()
     ↓
     4️⃣ run_all_algorithms()
        → Für jeden Algorithmus (kmeans, hierarchical, dbscan):
          → analysis/pipeline.py → run_analysis()
     ↓
     5️⃣ run_comparisons()
        ↓
        5a. comparison/gics_comparison.py
            → compare_all_algorithms()
            → Berechne: Cramér's V, Rand Index, Chi²
            → Erstelle: Contingency Tables, Heatmaps
        ↓
        5b. comparison/algorithm_comparison.py
            → compare_metrics()
            → compute_cluster_overlap()
            → Erstelle: Metrics Comparison, Overlap Heatmaps
        ↓
        5c. comparison/feature_importance.py
            → compute_all_algorithms()
            → Random Forest basierte Importance
            → Erstelle: Top-N Feature Plots
        ↓
        5d. comparison/temporal_stability.py
            → analyze_all_algorithms()
            → compute_migration_matrix()
            → Erstelle: Migration Heatmaps, Stability Plots
  ↓
6️⃣ output/comparison_handler.py
   → save_gics_comparison()
   → save_algorithm_comparison()
   → save_feature_importance()
   → save_temporal_stability()
   → save_plot() für alle Visualisierungen
```

---

## 📊 Visualisierungen (Vollständig)

### **Single Algorithm Mode**

Jeder Algorithmus erzeugt **3 Analyse-Stadien** × **4 Plot-Typen** = 12 Plots:

#### **Static Analysis**
```
output/{market}/{algorithm}/static/visualizations/
├── cluster_scatter.png              # 2D PCA Scatter
├── cluster_distribution.png         # Bar Chart Cluster-Größen
├── cluster_heatmap.png             # Correlation Heatmap
└── feature_distributions.png       # Violin Plots pro Feature
```

#### **Dynamic Analysis**
```
output/{market}/{algorithm}/dynamic/visualizations/
├── cluster_scatter.png              # Trend-basierte Scatter
├── cluster_distribution.png
├── cluster_heatmap.png
└── trend_comparison.png            # Line Plots Entwicklung
```

#### **Combined Analysis**
```
output/{market}/{algorithm}/combined/visualizations/
├── cluster_scatter.png
├── cluster_distribution.png
├── cluster_heatmap.png
└── static_vs_dynamic.png          # Vergleich beider Dimensionen
```

### **Comparison Mode**

#### **1. GICS Comparison** (4 Plots pro Stadium)
```
output/{market}/comparisons/01_gics_comparison/
├── summary_gics_static.png                     # Bar Chart Cramér's V
├── summary_gics_dynamic.png
├── summary_gics_combined.png
└── contingency_tables/
    ├── kmeans_vs_gsector.png                   # Heatmap 3×
    ├── hierarchical_vs_gsector.png
    └── dbscan_vs_gsector.png
```

#### **2. Algorithm Comparison** (6 Plots pro Stadium)
```
output/{market}/comparisons/02_algorithm_comparison/
├── metrics_comparison_static.png               # Bar Chart Metriken
├── metrics_comparison_dynamic.png
├── metrics_comparison_combined.png
├── algorithm_overlap_static.png                # Heatmap ARI
├── algorithm_overlap_dynamic.png
└── algorithm_overlap_combined.png
```

#### **3. Feature Importance** (10 Plots)
```
output/{market}/comparisons/03_feature_importance/
├── kmeans_importance_static.png                # Bar Chart Top 15
├── kmeans_importance_combined.png
├── hierarchical_importance_static.png
├── hierarchical_importance_combined.png
├── dbscan_importance_static.png
├── dbscan_importance_combined.png
├── combined_importance_static.png              # Vergleich alle 3
├── combined_importance_combined.png
├── importance_heatmap_static.png               # NEU: Heatmap
└── importance_heatmap_combined.png             # NEU: Heatmap
```

#### **4. Temporal Stability** (7+ Plots)
```
output/{market}/comparisons/04_temporal_stability/
├── algorithm_stability_comparison.png          # Bar Chart Consistency
├── kmeans_migration_heatmap.png                # Jahr-zu-Jahr Migration
├── hierarchical_migration_heatmap.png
├── dbscan_migration_heatmap.png
├── kmeans_yearly_stability.png                 # Line Plot über Zeit
├── hierarchical_yearly_stability.png
├── dbscan_yearly_stability.png
├── cluster_longevity.png                       # NEU: Durchschn. Verweildauer
└── migration_flow.png                          # NEU: Sankey Diagram (optional)
```

---

## 🔑 Wichtige Module (Detailliert)

### **core/config.py** (55 Zeilen)
**Zweck:** Konfiguration laden & abrufen
**Funktionen:**
- `load_config(path)` → dict
- `get_value(config, *keys, default=None)` → Any

**Verwendet von:** Alle Module

---

### **core/data_loader.py** (~370 Zeilen)
**Zweck:** Rohdaten laden, bereinigen, filtern
**Funktionen:**
- `load_data(path)` → pd.DataFrame
- `load_all_csv_from_directory(dir)` → pd.DataFrame
- `clean_data(df)` → (df_clean, report)
- `filter_relevant_columns(df)` → df_filtered

**Wichtig:**
- Fügt GICS-Spalten hinzu (gsector, gind, ggroup, gsubind)
- Bereinigt numerische Spalten
- Entfernt leere Spalten

**Verwendet von:** preprocessing.py

---

### **core/feature_engineering.py** (~650 Zeilen)
**Zweck:** Finanzkennzahlen berechnen
**Funktionen:**
- `create_all_features(df)` → df_with_features
- Einzelne Funktionen pro Kategorie:
  - Profitabilität: ROA, ROE, Margins
  - Liquidität: Current/Quick/Cash Ratio
  - Verschuldung: Debt-to-Equity, Equity Ratio
  - Effizienz: Turnover, Revenue per Employee
  - Wachstum: YoY Growth Rates

**Outlier Detection:**
- IQR-Methode
- Markiert aber entfernt NICHT

**Verwendet von:** preprocessing.py

---

### **core/preprocessing.py** (~100 Zeilen)
**Zweck:** Preprocessing orchestrieren
**Funktionen:**
- `run_preprocessing(input_dir, market)` → df_features
- `prepare_time_data(df, market)` → (df_all, df_latest)
- `load_processed_data(market)` → df_features

**Flow:**
1. Load Data (data_loader)
2. Clean Data (data_loader)
3. Filter Columns (data_loader)
4. Create Features (feature_engineering)
5. Save Processed Data

**Verwendet von:** main.py

---

### **clustering/factory.py** (~80 Zeilen)
**Zweck:** Factory Pattern für Algorithmen
**Funktion:**
- `ClustererFactory.create(algorithm, config, random_state)` → BaseClusterer

**Unterstützt:**
- `'kmeans'` → KMeansClusterer
- `'hierarchical'` → HierarchicalClusterer
- `'dbscan'` → DBSCANClusterer

**Verwendet von:** clustering_engine.py

---

### **analysis/clustering_engine.py** (~400 Zeilen)
**Zweck:** Clustering-Logik ausführen
**Funktionen:**
- `perform_clustering(df, features, n_clusters, type)` → (df, profiles, metrics)
- `compute_timeseries_features(df, min_years)` → df_aggregated
- `assign_clusters_to_timeseries(df_all, df_clustered)` → df_with_clusters
- `analyze_migration(df_static, df_dynamic, df_combined)` → migration_matrix

**Wichtig:**
- Verwendet Factory Pattern
- Berechnet Cluster-Namen basierend auf Profilen
- Speichert Scaler & Model

**Verwendet von:** pipeline.py

---

### **analysis/pipeline.py** (~280 Zeilen)
**Zweck:** 3-Stage Clustering Pipeline
**Funktionen:**
- `run_analysis(df_all, df_latest, run_static, run_dynamic)` → results
- `_run_static_analysis(df_latest)` → df_result
- `_run_dynamic_analysis(df_all)` → df_result
- `_run_combined_analysis(df_static, df_dynamic)` → df_result

**Wichtig:**
- Koordiniert alle 3 Stadien
- Speichert `df_timeseries` für Temporal Stability
- Ruft Visualizer auf (falls skip_plots=False)

**Verwendet von:** main.py, comparison_pipeline.py

---

### **comparison/comparison_pipeline.py** (~480 Zeilen)
**Zweck:** Alle Algorithmen vergleichen
**Funktionen:**
- `run_full_comparison_pipeline(df_all, df_latest)` → results
- `run_all_algorithms()` → algorithm_results
- `run_comparisons()` → comparison_results
- `_run_gics_comparison()`
- `_run_algorithm_comparison()`
- `_run_feature_importance()`
- `_run_temporal_stability()`

**Flow:**
1. Führt jeden Algorithmus aus (kmeans, hierarchical, dbscan)
2. Sammelt Ergebnisse
3. Führt 4 Vergleichs-Analysen durch
4. Speichert alle Ergebnisse & Plots

**Verwendet von:** main.py (--compare Mode)

---

### **output/output_handler.py** (~430 Zeilen)
**Zweck:** Einzelne Algorithmen-Ergebnisse speichern
**Funktionen:**
- `save_cluster_data(df, profiles, metrics, type)`
- `save_cluster_lists(df, type)`
- `save_models(scaler, model, type)`
- `create_summary_report(results)`

**Output-Struktur:**
```
output/{market}/{algorithm}/{type}/
  ├── reports/
  │   ├── data/
  │   ├── clusters/
  │   ├── models/
  │   └── analysis/
  └── visualizations/
```

**Verwendet von:** pipeline.py

---

### **output/comparison_handler.py** (~130 Zeilen)
**Zweck:** Vergleichs-Ergebnisse speichern
**Funktionen:**
- `save_gics_comparison(...)`
- `save_algorithm_comparison(...)`
- `save_feature_importance(...)`
- `save_temporal_stability(...)`
- `save_plot(fig, filename, category)`

**Output-Struktur:**
```
output/{market}/comparisons/
  ├── 01_gics_comparison/
  ├── 02_algorithm_comparison/
  ├── 03_feature_importance/
  └── 04_temporal_stability/
```

**Verwendet von:** comparison_pipeline.py

---

### **output/visualizer.py** (~400 Zeilen)
**Zweck:** Alle Visualisierungen erstellen
**Funktionen:**
- `create_all_plots(df, profiles, metrics, type, algorithm, output_dir)`
- `plot_cluster_scatter(df, algorithm)` → fig
- `plot_cluster_distribution(df)` → fig
- `plot_feature_distributions(df, features)` → fig
- `plot_cluster_heatmap(profiles)` → fig

**Wichtig:**
- Alle Plots 300 DPI (Publication Quality)
- Einheitlicher Style
- Automatische Farben

**Verwendet von:** pipeline.py

---

## 🎯 Best Practices

### **Separation of Concerns**
- **core/**: Daten laden & vorbereiten
- **clustering/**: Algorithmen-Implementierungen
- **analysis/**: Analyse-Logik & Orchestrierung
- **comparison/**: Vergleichs-Analysen
- **output/**: Ergebnisse speichern & visualisieren

### **Factory Pattern**
Alle Clustering-Algorithmen implementieren `BaseClusterer`:
```python
clusterer = ClustererFactory.create('kmeans', config, random_state=42)
```

### **Configuration**
Alle Parameter in `config.yaml`, nicht hardcoded:
```python
n_clusters = config.get_value(cfg, 'static_analysis', 'n_clusters', default=5)
```

### **Error Handling**
- Graceful Degradation bei fehlenden Features
- Warning statt Error bei Noise-Clustern
- Logging auf INFO-Level

---

## 📈 Performance

**Typische Ausführungszeiten (Germany Market, ~160 Companies):**

| Mode | Preprocessing | Single Algorithm | All 3 Algorithms | Total |
|------|--------------|------------------|------------------|-------|
| Single (skip-prep) | 0s | 0.4s | - | 0.4s |
| Single (full) | 1.5s | 0.4s | - | 1.9s |
| Compare (skip-prep) | 0s | - | 1.2s | 7.5s |
| Compare (full) | 1.5s | - | 1.2s | 9.0s |

**Bottlenecks:**
1. Feature Engineering (1.5s) - Zeitreihen-Berechnungen
2. Visualization (4s im Compare Mode) - PNG-Erzeugung
3. DBSCAN Parameter Search (optional, nicht aktiviert)

**Optimierungen:**
- Preprocessing cachen (`--skip-prep`)
- Plots überspringen (`--skip-plots`)
- Parallel Processing für Algorithmen (TODO)
