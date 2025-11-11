# Visualisierungs-Analyse & Dashboard-Konsolidierung

## Aktuelle Visualisierungen

### 1. Basis-Plots (cluster_visualizer.py)
- `plot_cluster_distribution` - Verteilung der Cluster
- `plot_cluster_characteristics` - Kennzahlen pro Cluster
- `plot_correlation_heatmap` - Korrelationsmatrix
- `plot_pca_clusters` - PCA-Visualisierung
- `create_performance_dashboard` - Performance-Dashboard (bereits konsolidiert!)

### 2. Erweiterte Plots (Pipeline)
- `_create_pca_plots` - Detaillierte PCA-Analyse
- `_create_algorithm_congruence_plots` - Robustness Check
- `_create_company_insights_plots` - Company Insights

### 3. Vergleichs-Plots (comparison_visualizer.py)
- `plot_gics_comparison` - GICS-Sektor-Vergleich
- `plot_algorithm_comparison` - Algorithmen-Vergleich
- `plot_feature_importance` - Feature Importance
- `plot_temporal_stability` - Zeitliche Stabilität

### 4. Spezial-Plots
- `plot_engine_scores.py` - Scoring-System-Plots
- `plot_engine_validation.py` - Validation-Plots

---

## Identifizierte Redundanzen

### ⚠️ Potenzielle Redundanz 1: PCA-Plots
- `plot_pca_clusters` (in Basis-Plots)
- `_create_pca_plots` (erweitert, nur wenn PCA enabled)

**Status:** NICHT redundant
- Basis-Plot: Einfache PCA-Visualisierung (immer)
- Erweiterte Plots: Detaillierte Analyse (nur wenn pca.enabled=true)

### ⚠️ Potenzielle Redundanz 2: Correlation Heatmap
- `plot_correlation_heatmap` - Korrelationsmatrix der Features

**Status:** BEHALTEN
- Wichtig für Feature-Auswahl und Multikollinearitäts-Check
- Nicht redundant zu Feature Importance

---

## Dashboard-Konsolidierungsvorschlag

### Option 1: Minimale Änderung (EMPFOHLEN)
**Keine Löschungen, nur Optimierung:**
- ✅ Alle bestehenden Plots behalten
- ✅ `create_performance_dashboard` ist bereits konsolidiert
- ✅ Nur Config-Parameter hinzufügen für optionale Plots

**Neue Config-Parameter:**
```yaml
visualization:
  enabled: true
  plots:
    cluster_distribution: true
    cluster_characteristics: true
    correlation_heatmap: true
    pca_clusters: true
    performance_dashboard: true
    algorithm_congruence: true  # Robustness
    company_insights: true
```

### Option 2: Konservative Konsolidierung
**Minimale Reduktion:**
- ✅ Basis-Plots bleiben alle
- ⚠️ `plot_correlation_heatmap` → Optional machen
- ✅ Alle anderen bleiben

---

## Empfehlung

**Vorgehen: Option 1 - Minimale Änderung**

Begründung:
1. User sagte: "nicht zu viel löschen"
2. Keine echten Redundanzen gefunden
3. Alle Plots haben spezifischen Nutzen
4. `create_performance_dashboard` ist bereits ein Dashboard

**Umsetzung:**
1. Config-Parameter für optionale Plots hinzufügen
2. Alle Plots standardmäßig aktiviert lassen
3. User kann selbst entscheiden, was er deaktiviert

---

## Ergebnis

❌ KEINE Plots werden gelöscht
✅ Alle Informationen bleiben erhalten
✅ Flexibilität durch Config-Parameter
