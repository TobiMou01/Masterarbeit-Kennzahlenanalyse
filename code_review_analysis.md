# Code-Review: Aufgabenteilung & Verbesserungspotential

## Ziel
- Klare Aufgabenteilung für jede .py-Datei
- Jupyter-Notebook-ready (kompakt, modular)
- Redundanzen eliminieren
- Bessere Übersichtlichkeit

---

## Dateistruktur-Analyse

### 📂 src/_01_setup/
```
config_loader.py        - Config laden & parsen
environment.py          - venv Check & Setup
interactive_menu.py     - Interaktives Terminal-Menu
output_handler.py       - Output-Verzeichnisse & Speichern
```

### 📂 src/_02_preprocessing/
```
data_loader.py          - Daten laden, bereinigen, imputieren
feature_engineer.py     - Features berechnen, CAGR-Glättung
data_cleaner.py         - ORCHESTRATOR: run_preprocessing()
pca_transformer.py      - PCA-Transformation (Dimensionsreduktion)
```

### 📂 src/_03_clustering/
```
cluster_engine.py       - Clustering durchführen (K-Means, etc.)
k_selector.py           - Automatische k-Bestimmung (Elbow, etc.)
pipeline.py             - ORCHESTRATOR: Gesamte Analyse-Pipeline
hierarchical_pipeline.py - Alternative Pipeline (Master-Labeling)
```

### 📂 src/_04_comparison/
```
comparison_engine.py    - ??? (muss ich anschauen)
comparison_pipeline.py  - ??? (muss ich anschauen)
feature_analyzer.py     - Feature Importance (RF + SHAP)
temporal_analyzer.py    - Temporale Stabilität & Migration
```

### 📂 src/_05_visualization/
```
cluster_visualizer.py   - Basis-Plots (Distribution, Heatmap, etc.)
comparison_visualizer.py - Vergleichs-Plots
plot_engine.py          - ORCHESTRATOR: create_all_plots()
plot_engine_pca.py      - PCA-spezifische Plots
plot_engine_scores.py   - Scoring-Plots
plot_engine_validation.py - Validation-Plots
summary_generator.py    - ??? (muss ich anschauen)
```

### 📂 src/_06_scoring/
```
??? (muss ich anschauen)
```

### 📂 src/_07_naming/
```
??? (muss ich anschauen)
```

---

## Zu untersuchende Fragen

1. **Redundanz zwischen pipeline.py und hierarchical_pipeline.py?**
   - Beide erben von ClusteringPipeline?
   - Kann man das vereinfachen?

2. **Redundanz zwischen comparison_engine.py und comparison_pipeline.py?**
   - Was ist der Unterschied?
   - Braucht man beide?

3. **Ist summary_generator.py leer/redundant?**

4. **Gibt es Overlap zwischen den plot_engine_*.py Files?**

5. **Können manche Orchestratoren zusammengelegt werden?**

---

## ANALYSE STARTET...

## ⚠️ KRITISCHE BEFUNDE

### 🔴 SEHR GROSSE DATEIEN (>800 Zeilen)
```
pipeline.py                    1,846 Zeilen  ⚠️⚠️⚠️ KRITISCH!
research_excel_writer.py       2,022 Zeilen  ⚠️⚠️⚠️ KRITISCH!
plot_engine_pca.py              948 Zeilen  ⚠️⚠️
plot_engine_scores.py         1,027 Zeilen  ⚠️⚠️
plot_engine_validation.py       996 Zeilen  ⚠️⚠️
feature_engineer.py             908 Zeilen  ⚠️⚠️
output_handler.py               891 Zeilen  ⚠️⚠️
```

**Jupyter-Problem:** Diese Dateien sind zu groß für übersichtliche Notebooks!

### 🟡 POTENTIELLE REDUNDANZEN

1. **comparison_engine.py (477) vs comparison_pipeline.py (579)**
   - Beide ~500 Zeilen
   - Klare Aufgabenteilung?

2. **summary_generator.py (20 Zeilen)**
   - Fast leer - kann integriert werden?

3. **config_loader.py vs feature_config_loader.py vs feature_selector.py**
   - 3 Dateien für Config-Handling?
   - Kann man vereinfachen?

---

## NÄCHSTE SCHRITTE

1. ✅ Größte Datei analysieren: `pipeline.py` (1846 Zeilen)
2. ✅ `summary_generator.py` prüfen (evtl. leer?)
3. ✅ Redundanzen zwischen comparison_*.py finden
4. ✅ Config-Loader konsolidieren
