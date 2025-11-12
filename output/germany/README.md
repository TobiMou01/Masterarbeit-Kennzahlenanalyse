# Output-Struktur - Masterarbeit Kennzahlenanalyse

**Projekt**: Kennzahlenbasierte Unternehmensklassifikation mittels Clusteranalyse
**Generiert**: 2024-11-12
**Markt**: Germany

---

## 📁 Verzeichnisstruktur

```
output/germany/
├── 02_algorithms/          # Clustering-Algorithmen & Analysen (29 MB)
├── 03_comparisons/         # Vergleichsanalysen (5.3 MB)
├── 04_excel_reports/       # Excel-Reports für Masterarbeit (1.8 MB)
└── 99_summary/            # Kurz-Zusammenfassung (7 KB)
```

---

## 🎯 Mapping zur Exposé-Methodik

### 02_algorithms/ - Hauptanalysen

Umsetzung der **Clusterbildung** (Exposé Kapitel 4.3):

```
02_algorithms/
├── kmeans_comparative/     # Hauptverfahren K-Means (3 Varianten)
│   ├── combined/          # Static + Dynamic Features (empfohlen)
│   ├── static/            # Momentaufnahme-Kennzahlen
│   └── dynamic/           # Wachstumsraten (CAGR)
│
├── hierarchical/          # Validierung via Hierarchisches Clustering
└── dbscan/               # Validierung via DBSCAN
```

#### Analyse-Bereiche pro Algorithmus:

**K-Means (detailliert)**:
1. `1_cluster_quality/` → **Exposé Kernfrage 1 (Homogenität)**
   - Silhouette Scores, Intra-Cluster-Varianz
   - Cluster Naming & Profiling
   - Score Evolution (nur in combined/)

2. `2_algorithm_congruence/` → **Methodische Robustheit**
   - ARI (Adjusted Rand Index) zwischen Algorithmen
   - Konfusionsmatrizen

3. `3_external_validation/` → **Exposé Kernfrage 2 (Kongruenz)**
   - Cramér's V gegen GICS (Branchen)
   - Chi²-Tests gegen Größenklassen

4. `4_company_insights/` → **Deskriptive Analysen**
   - Top/Bottom Performers
   - Outlier Detection
   - Feature-Verteilungen

5. `5_pca_analysis/` → **Exposé Kapitel 4.2 (Dimensionsreduktion)**
   - Scree Plots, Biplots
   - Component Loadings
   - Cluster Separation im PC-Raum

**Hierarchical & DBSCAN (kompakt)**:
- `master_clustering/` → Basis-Clustering
- `dynamic_enrichment/` → Mit dynamischen Features
- `combined_scores/` → Kombinierte Scores

---

### 03_comparisons/ - Vergleichsanalysen

Systematische Vergleiche über Algorithmen & Labels:

```
03_comparisons/
├── temporal/              # Exposé Kernfrage 4 (Stabilität über Zeit)
│   ├── *_migration_matrix.csv      # Übergangsmatrizen
│   ├── *_migration_heatmap.png     # Visualisierungen
│   ├── cluster_migrations.csv      # Migrationsanalysen
│   └── stability_metrics.csv       # Stabilitäts-KPIs
│
├── gics/ + gics_tables/   # Exposé Kernfrage 2 (Kongruenz mit Branchen)
│   ├── cramers_v.csv              # Assoziationsstärke
│   ├── chi_square.csv             # Signifikanz-Tests
│   ├── *_vs_gsector.csv           # Kontingenz-Tabellen
│   └── *.png                      # Visualisierungen
│
├── features/              # Exposé Kernfrage 3 (Treiber)
│   ├── *_importance.csv           # Feature Importance (Random Forest)
│   ├── *_importance.png           # Visualisierungen
│   └── combined_importance_*.png  # Vergleiche
│
└── algorithms/            # Algorithmen-Metriken
    └── metrics_comparison_*.png   # Silhouette, Calinski-Harabasz, etc.
```

---

### 04_excel_reports/ - Excel-Analysen

Aufbereitete Reports für Masterarbeit:

```
04_excel_reports/
├── research_analysis_master.xlsx  (1.2 MB, 14 Sheets)
│   └── Master-Thesis File - Strukturiert nach 4 Kernfragen
│       ├── Section 0: Config & Overview
│       ├── Section 1: Homogenität (4 Sheets)
│       ├── Section 2: Kongruenz (3 Sheets)
│       ├── Section 3: Treiber (3 Sheets)
│       └── Section 4: Stabilität (3 Sheets)
│
├── algorithm_comparison_combined.xlsx  (65 KB, 5 Sheets)
│   └── Algorithmen-Vergleich (K-Means vs. Hierarchical vs. DBSCAN)
│
└── company_cluster_analysis/  (5 Files)
    ├── kmeans_combined.xlsx      # K-Means Combined (empfohlen)
    ├── kmeans_static.xlsx        # K-Means Static
    ├── kmeans_dynamic.xlsx       # K-Means Dynamic
    ├── hierarchical.xlsx         # Hierarchical Clustering
    └── dbscan.xlsx              # DBSCAN
    └── Pro File: Overview, Pro-Cluster Sheets, Score Evolution, Summary
```

---

## 📊 Datei-Typen & Verwendung

### CSV-Dateien (126 Files)
- `assignments.csv` → Cluster-Zuweisungen pro Unternehmen
- `profiles.csv` → Cluster-Profile (Mittelwerte, Std.-Abw.)
- `company_scores.csv` → Unternehmensspezifische Scores
- `*_importance.csv` → Feature Importance (Treiber-Analyse)
- `*_matrix.csv` → Übergangsmatrizen, Kontingenz-Tabellen

### PNG-Visualisierungen (125 Files)
- `performance_dashboard.png` → Cluster-Übersicht (Scores, Größen)
- `cluster_characteristics.png` → Feature-Verteilungen
- `correlation_heatmap.png` → Feature-Korrelationen
- `*_migration_heatmap.png` → Zeitliche Stabilität
- `biplot_*.png`, `scree_plot.png` → PCA-Analysen
- `contingency_*.png` → GICS/Größen-Vergleiche

### Pickle-Dateien (.pkl)
- `kmeans_model.pkl` → Trainierte Modelle (für Re-Prediction)

---

## 🎓 Verwendung für Masterarbeit

### Für Thesis-Kapitel:

**Kapitel 4.3 - Clusterbildung**:
- `02_algorithms/kmeans_comparative/combined/`
- Nutze: `1_cluster_quality/` für Metriken & Naming

**Kapitel 4.4 - Vergleich & Bewertung**:
- `03_comparisons/gics/` → Kongruenz mit Branchen
- `03_comparisons/algorithms/` → Algorithmen-Metriken
- `02_algorithms/*/3_external_validation/` → Cramér's V

**Kapitel 4.5 - Treiberanalyse**:
- `03_comparisons/features/` → Feature Importance
- `02_algorithms/kmeans_comparative/combined/5_pca_analysis/` → PCA Loadings

**Kapitel 4.6 - Stabilität**:
- `03_comparisons/temporal/` → Migrationsmatrizen, Stabilitäts-KPIs

### Für Präsentationen:

**Master-Excel für Prüfer**:
- `04_excel_reports/research_analysis_master.xlsx`
- Enthält alle 4 Kernfragen auf separaten Sheets

**Detail-Analysen**:
- `04_excel_reports/company_cluster_analysis/kmeans_combined.xlsx`
- Für Deep-Dive in einzelne Cluster

---

## 🔧 Technische Details

### File-Naming Konventionen:
- `kmeans_*` → K-Means Algorithmus
- `hierarchical_*` → Hierarchisches Clustering
- `dbscan_*` → DBSCAN
- `*_combined` → Static + Dynamic Features
- `*_static` → Momentaufnahme-Kennzahlen
- `*_dynamic` → Wachstumsraten (CAGR)

### Reproduzierbarkeit:
- Config: Siehe `research_analysis_master.xlsx` Sheet "0_Config"
- Modelle: `02_algorithms/*/models/*.pkl`
- Daten-Pipeline: `src/_02_preprocessing/` → `src/_03_clustering/` → `src/_04_comparison/`

---

## 📝 Changelog

**2024-11-12**:
- Initiale Struktur nach Refactoring (14 → 43 Module)
- 7 Commits: Runtime-Fehler behoben
- Output-Cleanup: Duplicates entfernt, Excel-Files organisiert

---

## 💡 Hinweise

1. **Empfohlene Analyse**: Nutze `kmeans_comparative/combined/` (beste Balance zwischen Static & Dynamic)
2. **Feature-Problem**: Aktuell fehlen Features beim Clustering → Scores=0, generische Namen (wird behoben)
3. **Excel-Visualisierungen**: Werden derzeit nur teilweise eingebettet (Phase 2 TODO)

---

## 📧 Kontakt

Bei Fragen zur Output-Struktur:
- Siehe `src/_04_comparison/comparison_pipeline.py`
- Excel-Generierung: `src/_04_comparison/section_writers/`
