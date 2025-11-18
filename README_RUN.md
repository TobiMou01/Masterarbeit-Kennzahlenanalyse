# 🚀 Clustering Analysis - Quick Start Guide

## **Einfaches Python-Programm für reproduzierbare Analysen**

Dieses Tool ermöglicht dir, Clustering-Analysen **komplett über die Kommandozeile** zu steuern, ohne das Jupyter Notebook zu nutzen.

---

## 📋 **VORAUSSETZUNGEN**

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Daten vorbereiten
# Stelle sicher, dass deine CSVs in data/raw/germany/ liegen:
data/raw/germany/
├── dax40_proxy.csv
├── mdax_proxy.csv
└── sdax_proxy.csv
```

---

## ⚡ **QUICK START - 3 Schritte**

### **Schritt 1: Minimal-Test (Empfohlen für Start!)**

```bash
python run.py --config configs/minimal_quick.yaml --analyses static
```

**Was passiert:**
- Lädt DAX/MDAX/SDAX Daten
- Nutzt nur 6-8 wichtigste Features
- Erstellt 4-7 Cluster (automatisch optimiert)
- Output in: `output/germany/02_algorithms/kmeans_comparative/static/`

**Dauer:** ~2-5 Minuten

---

### **Schritt 2: Ergebnisse prüfen**

```bash
# Silhouette Score (Cluster-Qualität)
cat output/germany/02_algorithms/kmeans_comparative/static/1_cluster_quality/silhouette_score.txt

# Cluster-Größen
cat output/germany/02_algorithms/kmeans_comparative/static/1_cluster_quality/cluster_sizes.csv

# Plots anschauen
open output/germany/02_algorithms/kmeans_comparative/static/1_cluster_quality/plots/
```

**Gute Ergebnisse:**
- ✅ Silhouette Score >0.4
- ✅ Cluster-Größen relativ ausgewogen
- ✅ Klare visuelle Trennung in Plots

**Schlechte Ergebnisse:**
- ❌ Silhouette Score <0.3
- ❌ Ein Cluster >70% der Daten
- ❌ Cluster überlappen stark

→ **Wenn schlecht:** Lies `CLUSTERING_TUNING_GUIDE.md` für Verbesserungen!

---

### **Schritt 3: Wenn gut → Erweitern**

```bash
# Alle Analysen (static + dynamic + combined)
python run.py --config configs/minimal_quick.yaml

# Mit PCA (wenn viele Features)
python run.py --config configs/pca_optimized.yaml

# Algorithmen vergleichen
python run.py --algorithms kmeans hierarchical --analyses static
```

---

## 🎛️ **USAGE - Alle Optionen**

### **Basic Usage**

```bash
# Standard (nutzt config.yaml)
python run.py

# Custom Config
python run.py --config my_config.yaml

# Nur Static Analyse
python run.py --analyses static

# Mehrere Algorithmen
python run.py --algorithms kmeans hierarchical

# Custom Feature-Preset
python run.py --preset minimal
```

---

### **Advanced Usage**

```bash
# Nur DAX (nicht MDAX/SDAX)
python run.py --files dax40_proxy.csv

# Fixed K (statt Auto-Selektion)
python run.py --n-clusters 5

# Dry-Run (zeigt Config, führt nicht aus)
python run.py --dry-run

# Ohne Algorithmen-Vergleich (schneller)
python run.py --algorithms kmeans hierarchical --no-comparison
```

---

### **Vollständige Optionen**

```
--config, -c        Config-Datei (default: config.yaml)
--market, -m        Market (überschreibt config)
--files, -f         Spezifische CSV-Files
--analyses, -a      Welche Analysen: static, dynamic, combined
--algorithms, -A    Welche Algorithmen: kmeans, hierarchical, dbscan
--preset, -p        Feature-Preset (minimal, standard, extended)
--n-clusters, -k    Fixe Anzahl Cluster
--no-comparison     Skip Algorithmen-Vergleich
--dry-run           Nur Config anzeigen
```

---

## 📂 **OUTPUT-STRUKTUR**

Nach dem Run findest du alle Ergebnisse in:

```
output/germany/02_algorithms/kmeans_comparative/static/
├── 1_cluster_quality/
│   ├── silhouette_score.txt         # Hauptmetrik!
│   ├── cluster_sizes.csv            # Größen pro Cluster
│   ├── inertia.txt                  # Within-cluster sum of squares
│   └── plots/
│       ├── cluster_distribution.png # Verteilung
│       └── silhouette_plot.png      # Per-sample Silhouette
│
├── 2_algorithm_congruence/          # (nur wenn mehrere Algorithmen)
│   └── ari_matrix.csv               # Algorithmen-Übereinstimmung
│
├── 3_external_validation/
│   ├── cramers_v.csv                # GICS-Cluster Assoziation
│   └── contingency_tables/          # Cluster × GICS
│
├── 4_company_insights/
│   ├── data/
│   │   ├── cluster_profiles.csv     # Kennzahlen pro Cluster
│   │   ├── company_assignments.csv  # Welches Unternehmen → welcher Cluster
│   │   └── cluster_statistics.csv   # Mean, Std pro Cluster
│   ├── plots/
│   │   └── cluster_characteristics.png  # Feature-Profile
│   └── rankings/
│       └── top_bottom_companies.csv
│
└── summary/
    └── analysis_report.md           # Zusammenfassung
```

---

## 🔧 **CONFIGS ANPASSEN**

### **1. Nutze vorhandene Configs (einfachste Methode)**

```bash
# Minimal (wenige Features, schnell)
python run.py --config configs/minimal_quick.yaml

# PCA-optimiert (viele Features)
python run.py --config configs/pca_optimized.yaml
```

---

### **2. Eigene Config erstellen**

```bash
# Kopiere Vorlage
cp config.yaml my_analysis.yaml

# Bearbeite my_analysis.yaml
nano my_analysis.yaml  # oder dein Editor

# Die wichtigsten Stellschrauben:
```

```yaml
# In my_analysis.yaml:

# Feature-Auswahl (WICHTIGSTE Stellschraube!)
feature_selection:
  preset: 'minimal'  # Optionen: minimal, standard, extended

# Cluster-Anzahl
cluster_selection:
  mode: 'auto'     # Oder 'manual' mit fixer k
  k_range: [4, 7]  # Teste 4-7 Cluster

# PCA (bei vielen Features)
pca:
  enabled: true    # Aktivieren bei >12 Features
  n_components: 0.85

# Outlier
preprocessing:
  outlier_detection:
    enabled: true
    iqr_multiplier: 1.5  # 1.5 = Standard, 2.0 = weniger streng

# Welche CSV-Dateien?
data:
  file_selection:
    germany:
      - 'dax40_proxy.csv'  # Nur DAX
      # - 'mdax_proxy.csv'  # Auskommentiert = nicht laden
```

```bash
# Nutzen
python run.py --config my_analysis.yaml
```

---

### **3. Features anpassen (Advanced)**

Bearbeite `features_config.yaml`:

```yaml
# In features_config.yaml:

feature_presets:
  my_custom_preset:
    description: "Mein optimiertes Feature-Set"
    static_features:
      - roa
      - ebit_margin
      - debt_to_equity
      - current_ratio
      - fcf_margin
      # NUR die wichtigsten!

    dynamic_features:
      - roa_trend
      - revenue_growth
      - margin_trend
```

Dann in config:
```yaml
feature_selection:
  preset: 'my_custom_preset'
```

---

## 🎯 **TYPISCHE WORKFLOWS**

### **Workflow 1: Schneller Test**
```bash
# Minimal Config, nur Static
python run.py --config configs/minimal_quick.yaml --analyses static

# Prüfe Silhouette Score
cat output/germany/.../1_cluster_quality/silhouette_score.txt

# Wenn >0.4: Gut!
# Wenn <0.3: Passe Features an (siehe TUNING_GUIDE.md)
```

---

### **Workflow 2: Iterative Optimierung**

```bash
# Test 1: Minimal
python run.py --preset minimal --analyses static

# Test 2: Standard (mehr Features)
python run.py --preset standard --analyses static

# Test 3: Mit PCA
python run.py --config configs/pca_optimized.yaml

# Vergleiche Silhouette Scores
# Nutze das beste Setup für Full Run
```

---

### **Workflow 3: Multi-Algorithm Comparison**

```bash
# Alle 3 Algorithmen vergleichen
python run.py \
  --algorithms kmeans hierarchical dbscan \
  --analyses static

# Schaue ARI-Matrix an
cat output/germany/03_comparisons/algorithms/ari_matrix.csv

# Nutze den besten Algorithmus für weitere Analysen
```

---

## 📊 **ERGEBNISSE INTERPRETIEREN**

### **Silhouette Score** (Hauptmetrik)

- **>0.5:** Exzellente Cluster-Trennung ⭐⭐⭐
- **0.4-0.5:** Gute Cluster ✅
- **0.3-0.4:** Akzeptable Cluster ⚠️
- **<0.3:** Schlechte Trennung ❌ → Anpassung nötig!

### **Cluster-Größen**

Ideal: Alle Cluster zwischen 10-40% der Daten

**Beispiel Gut:**
```
Cluster 0: 25 companies (15.6%)
Cluster 1: 32 companies (20.0%)
Cluster 2: 28 companies (17.5%)
Cluster 3: 35 companies (21.9%)
Cluster 4: 40 companies (25.0%)
```

**Beispiel Schlecht:**
```
Cluster 0: 120 companies (75%)  ← Zu groß!
Cluster 1: 15 companies (9.4%)
Cluster 2: 10 companies (6.3%)
Cluster 3: 8 companies (5.0%)
Cluster 4: 7 companies (4.4%)
```

---

## ❗ **TROUBLESHOOTING**

### Problem: `FileNotFoundError: dax40_proxy.csv`
**Lösung:** Stelle sicher, dass CSVs in `data/raw/germany/` liegen

---

### Problem: `ModuleNotFoundError: src._02_preprocessing`
**Lösung:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run.py
```

---

### Problem: Silhouette Score <0.3
**Lösung:** Siehe `CLUSTERING_TUNING_GUIDE.md` - Die 6 Stellschrauben!

Kurzfassung:
1. WENIGER Features nutzen (`preset: minimal`)
2. Outliers entfernen (`outlier_detection.enabled: true`)
3. PCA aktivieren (`pca.enabled: true`)
4. k-Range anpassen (`k_range: [4, 7]`)

---

### Problem: "Zu lange Laufzeit"
**Lösung:**
```bash
# Nur Static (schnellst)
python run.py --analyses static --no-comparison

# Weniger n_init
# In config.yaml: kmeans.n_init: 20 (statt 50)
```

---

## 📚 **WEITERE RESSOURCEN**

- **`CLUSTERING_TUNING_GUIDE.md`** - Detaillierte Optimierungs-Tipps
- **`config.yaml`** - Haupt-Konfiguration mit allen Optionen
- **`features_config.yaml`** - Feature-Sets und Definitionen
- **`notebooks/MASTER_Workflow.ipynb`** - Interaktive Exploration

---

## 🎓 **BEST PRACTICES**

1. **Starte mit Minimal Config** (`configs/minimal_quick.yaml`)
2. **Prüfe Silhouette Score** nach jedem Run
3. **Iteriere:** Test → Prüfen → Anpassen → Wiederholen
4. **Erst Static perfektionieren**, dann Dynamic/Combined
5. **Dokumentiere** welche Config am besten funktioniert

---

## ✅ **QUICK CHECKLIST**

- [ ] Dependencies installiert (`pip install -r requirements.txt`)
- [ ] Daten in `data/raw/germany/` vorhanden
- [ ] Ersten Test-Run gemacht (`configs/minimal_quick.yaml`)
- [ ] Silhouette Score geprüft (Ziel: >0.4)
- [ ] Bei Bedarf: Config angepasst (siehe TUNING_GUIDE.md)
- [ ] Full Run mit optimaler Config
- [ ] Ergebnisse in `output/germany/` validiert

---

**Happy Clustering! 🎉**

Bei Fragen: Schau dir die generierten Plots und Reports in `output/{market}/` an.
