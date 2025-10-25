# 📊 Visualizations Overview

## Alle Visualisierungen in der Pipeline

### 🎯 **Single Algorithm Mode**

Für jeden Algorithmus (kmeans, hierarchical, dbscan) werden **36 Plots** erstellt:
- 3 Analyse-Stadien (static, dynamic, combined)
- 4 Plots pro Stadium
- 3 Algorithmen

#### **Pro Algorithmus & Stadium:**

**1. Cluster Scatter** (`cluster_scatter.png`)
- 2D PCA Projektion
- Farbcodiert nach Cluster
- Zeigt Cluster-Trennung

**2. Cluster Distribution** (`cluster_distribution.png`)
- Bar Chart
- Anzahl Unternehmen pro Cluster
- Mit Prozent-Angaben

**3. Cluster Heatmap** (`cluster_heatmap.png`)
- Korrelation zwischen Features
- Pro Cluster
- Zeigt Cluster-Charakteristika

**4. Feature Distributions** (`feature_distributions.png`)
- Violin Plots
- Alle verwendeten Features
- Vergleich über alle Cluster

**Gesamt: 3 Algorithmen × 3 Stadien × 4 Plots = 36 Plots**

---

### 🔬 **Comparison Mode**

#### **1️⃣ GICS Comparison** (10 Plots)

**Static Analysis:**
- `summary_gics_static.png` - Cramér's V Vergleich
- `kmeans_vs_gsector.png` - Kontingenztabelle Heatmap
- `hierarchical_vs_gsector.png` - Kontingenztabelle Heatmap
- `dbscan_vs_gsector.png` - Kontingenztabelle Heatmap

**Dynamic Analysis:**
- `summary_gics_dynamic.png`
- `kmeans_vs_gsector_dynamic.png`
- `hierarchical_vs_gsector_dynamic.png`
- `dbscan_vs_gsector_dynamic.png`

**Combined Analysis:**
- `summary_gics_combined.png`
- `combined_contingency.png` - Alle 3 Algorithmen

**Total: 10 Plots**

#### **2️⃣ Algorithm Comparison** (9 Plots)

**Static:**
- `metrics_comparison_static.png` - Silhouette, Davies-Bouldin, etc.
- `algorithm_overlap_static.png` - ARI Heatmap

**Dynamic:**
- `metrics_comparison_dynamic.png`
- `algorithm_overlap_dynamic.png`

**Combined:**
- `metrics_comparison_combined.png`
- `algorithm_overlap_combined.png`

**Overall:**
- `overall_metrics.png` - Alle Metriken über alle Stadien
- `cluster_count_comparison.png` - Cluster-Anzahl Vergleich
- `noise_comparison.png` - Noise % (DBSCAN)

**Total: 9 Plots**

#### **3️⃣ Feature Importance** (12 Plots)

**Pro Algorithmus (static):**
- `kmeans_importance_static.png` - Top 15 Features
- `hierarchical_importance_static.png`
- `dbscan_importance_static.png`

**Pro Algorithmus (combined):**
- `kmeans_importance_combined.png`
- `hierarchical_importance_combined.png`
- `dbscan_importance_combined.png`

**Vergleiche:**
- `combined_importance_static.png` - Alle 3 Algorithmen Side-by-Side
- `combined_importance_combined.png`
- `importance_heatmap_static.png` - Feature × Algorithm Heatmap
- `importance_heatmap_combined.png`
- `top_features_comparison.png` - Top 10 über alle
- `feature_agreement.png` - Welche Features sind sich alle einig?

**Total: 12 Plots**

#### **4️⃣ Temporal Stability** (11 Plots)

**Migration Heatmaps:**
- `kmeans_migration_heatmap.png` - Von Cluster X → Y
- `hierarchical_migration_heatmap.png`
- `dbscan_migration_heatmap.png`

**Yearly Stability:**
- `kmeans_yearly_stability.png` - Line Plot über Jahre
- `hierarchical_yearly_stability.png`
- `dbscan_yearly_stability.png`

**Vergleiche:**
- `algorithm_stability_comparison.png` - Bar Chart Consistency
- `cluster_longevity.png` - Durchschn. Verweildauer
- `migration_flow_kmeans.png` - Sankey Diagram (optional)
- `stability_over_time.png` - Alle Algorithmen Line Plot
- `most_stable_clusters.png` - Top 5 stabilste Cluster

**Total: 11 Plots**

---

## 📦 **Gesamt-Übersicht**

### **Single Mode:**
- 36 Plots (3 Algorithmen × 3 Stadien × 4 Plots)

### **Comparison Mode:**
- 36 Plots (Single Mode) +
- 42 Comparison Plots
- **= 78 Plots gesamt!**

#### **Breakdown Comparison:**
```
01_gics_comparison/          10 Plots
02_algorithm_comparison/      9 Plots
03_feature_importance/       12 Plots
04_temporal_stability/       11 Plots
─────────────────────────────────────
Total Comparison:            42 Plots
```

---

## 🎨 **Plot-Spezifikationen**

### **Allgemeine Einstellungen:**
- **DPI:** 300 (Publication Quality)
- **Format:** PNG
- **Farbschema:** Einheitlich (seaborn default + custom colors)
- **Figsize:**
  - Small Plots: 10×6
  - Medium Plots: 12×8
  - Large Plots: 14×10

### **Schriftgrößen:**
- Title: 14pt, bold
- Axis Labels: 12pt
- Tick Labels: 10pt
- Annotations: 9pt, bold

### **Farben:**
```python
algorithm_colors = {
    'kmeans': '#3498db',      # Blau
    'hierarchical': '#e74c3c',# Rot
    'dbscan': '#2ecc71'       # Grün
}

cluster_colors = sns.color_palette('husl', n_clusters)
```

---

## 🔍 **Detaillierte Plot-Beschreibungen**

### **Cluster Scatter**
- **Methode:** PCA (2 Komponenten)
- **X-Axis:** PC1 (erklärt ~30-40% Varianz)
- **Y-Axis:** PC2 (erklärt ~20-30% Varianz)
- **Marker:** Punkte, größe=50, alpha=0.6
- **Zentroiden:** Große Diamanten (DBSCAN: keine)
- **Legend:** Cluster-Namen + Größe
- **Noise:** Grau, wenn vorhanden

### **Metrics Comparison**
- **Plot-Typ:** Grouped Bar Chart
- **X-Axis:** Algorithmen
- **Y-Axis:** Metrik-Wert
- **Subplots:** 2×2 Grid
  - Top-Left: Silhouette (höher = besser)
  - Top-Right: Davies-Bouldin (niedriger = besser)
  - Bottom-Left: N Clusters
  - Bottom-Right: Noise %
- **Annotations:** Werte auf Balken

### **Feature Importance**
- **Plot-Typ:** Horizontal Bar Chart
- **X-Axis:** Importance Score
- **Y-Axis:** Feature Namen (sortiert)
- **Top-N:** 15 Features
- **Annotations:** Importance-Werte rechts
- **Combined Plot:** 3 Balken pro Feature (side-by-side)

### **Migration Heatmap**
- **Plot-Typ:** Seaborn Heatmap
- **Rows:** Cluster (Year t-1)
- **Columns:** Cluster (Year t)
- **Values:** % der Unternehmen
- **Colormap:** YlOrRd
- **Annotations:** Absolute Zahlen
- **Diagonal:** Hervorgehoben (= stayed same)

### **GICS Contingency**
- **Plot-Typ:** Seaborn Heatmap
- **Rows:** Cluster ID
- **Columns:** GICS Sector
- **Values:** Normalisiert auf % pro Cluster
- **Annotations:** Absolute Zahlen
- **Colormap:** YlOrRd

---

## 🗂️ **Output-Struktur (Dateien)**

```
output/germany/
│
├── kmeans/
│   ├── static/visualizations/
│   │   ├── cluster_scatter.png
│   │   ├── cluster_distribution.png
│   │   ├── cluster_heatmap.png
│   │   └── feature_distributions.png
│   ├── dynamic/visualizations/
│   │   └── [gleiche 4 Plots]
│   └── combined/visualizations/
│       └── [gleiche 4 Plots]
│
├── hierarchical/
│   └── [gleiche Struktur]
│
├── dbscan/
│   └── [gleiche Struktur]
│
└── comparisons/
    ├── 01_gics_comparison/
    │   ├── summary_gics_static.png
    │   ├── summary_gics_dynamic.png
    │   ├── summary_gics_combined.png
    │   └── contingency_tables/
    │       ├── kmeans_vs_gsector.png
    │       ├── hierarchical_vs_gsector.png
    │       └── dbscan_vs_gsector.png
    │
    ├── 02_algorithm_comparison/
    │   ├── metrics_comparison_static.png
    │   ├── metrics_comparison_dynamic.png
    │   ├── metrics_comparison_combined.png
    │   ├── algorithm_overlap_static.png
    │   ├── algorithm_overlap_dynamic.png
    │   ├── algorithm_overlap_combined.png
    │   ├── overall_metrics.png
    │   ├── cluster_count_comparison.png
    │   └── noise_comparison.png
    │
    ├── 03_feature_importance/
    │   ├── kmeans_importance_static.png
    │   ├── kmeans_importance_combined.png
    │   ├── hierarchical_importance_static.png
    │   ├── hierarchical_importance_combined.png
    │   ├── dbscan_importance_static.png
    │   ├── dbscan_importance_combined.png
    │   ├── combined_importance_static.png
    │   ├── combined_importance_combined.png
    │   ├── importance_heatmap_static.png
    │   ├── importance_heatmap_combined.png
    │   ├── top_features_comparison.png
    │   └── feature_agreement.png
    │
    └── 04_temporal_stability/
        ├── kmeans_migration_heatmap.png
        ├── hierarchical_migration_heatmap.png
        ├── dbscan_migration_heatmap.png
        ├── kmeans_yearly_stability.png
        ├── hierarchical_yearly_stability.png
        ├── dbscan_yearly_stability.png
        ├── algorithm_stability_comparison.png
        ├── cluster_longevity.png
        ├── migration_flow_kmeans.png
        ├── stability_over_time.png
        └── most_stable_clusters.png
```

---

## 📝 **Verwendung für Masterarbeit**

### **Welche Plots für welches Kapitel?**

#### **Kapitel: Methodik**
- `cluster_scatter.png` (static) - Zeigt Clustering-Konzept
- `feature_distributions.png` - Zeigt verwendete Kennzahlen
- `metrics_comparison_combined.png` - Zeigt Algorithmen-Vergleich

#### **Kapitel: Ergebnisse - GICS-Unabhängigkeit**
- `summary_gics_static.png` - **HAUPTARGUMENT!**
- `hierarchical_vs_gsector.png` - Detailansicht
- Cramér's V = 0.275 → "Cluster folgen NICHT Branchen"

#### **Kapitel: Ergebnisse - Cluster-Analyse**
- `cluster_distribution.png` - Cluster-Größen
- `cluster_scatter.png` - Visuelle Trennung
- Profiles aus CSV für Tabelle

#### **Kapitel: Ergebnisse - Feature Importance**
- `combined_importance_static.png` - **Top Features**
- "Current Ratio (22%), ROA (22%), Equity Ratio (17%)"
- → "Liquidität und Profitabilität sind entscheidend"

#### **Kapitel: Ergebnisse - Stabilität**
- `algorithm_stability_comparison.png` - Consistency
- `kmeans_migration_heatmap.png` - Jahr-zu-Jahr Entwicklung
- "100% Consistency → stabile Grupp ierung"

#### **Kapitel: Diskussion**
- `algorithm_overlap_combined.png` - Algorithmen stimmen überein
- `stability_over_time.png` - Langfristige Trends

---

## 🎯 **Wichtigste Plots für Präsentation**

**Top 10 Must-Have Plots:**

1. ✅ `summary_gics_static.png` - GICS-Unabhängigkeit
2. ✅ `cluster_scatter.png` (hierarchical, combined) - Hauptergebnis
3. ✅ `combined_importance_static.png` - Feature Importance
4. ✅ `metrics_comparison_combined.png` - Algorithmen-Vergleich
5. ✅ `hierarchical_vs_gsector.png` - GICS Contingency
6. ✅ `algorithm_stability_comparison.png` - Temporal Stability
7. ✅ `cluster_distribution.png` (hierarchical, static) - Cluster-Größen
8. ✅ `feature_distributions.png` (hierarchical, static) - Kennzahlen-Verteilung
9. ✅ `algorithm_overlap_combined.png` - Cluster-Überlappung
10. ✅ `cluster_longevity.png` - Verweildauer

**Diese 10 Plots erzählen die komplette Story!**
