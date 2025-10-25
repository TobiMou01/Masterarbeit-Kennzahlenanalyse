# 🎯 Financial Clustering Analysis Pipeline

Clustering-basierte Unternehmensanalyse mit **3 Algorithmen** (K-Means, Hierarchical, DBSCAN) und **3 Analyse-Stadien** (Static, Dynamic, Combined).

## 🚀 Schnellstart

### Vergleichs-Analyse (EMPFOHLEN für Masterarbeit)
```bash
# Führt ALLE Algorithmen aus + umfassende Vergleiche
python src/main.py --market germany --compare
```

**Output:**
- ✅ GICS-Vergleich (Branchen-Unabhängigkeit!)
- ✅ Algorithmen-Vergleich (Welcher ist besser?)
- ✅ Metriken-Vergleich (Silhouette, Davies-Bouldin)
- ✅ Cluster-Überlappung zwischen Algorithmen

### Einzelner Algorithmus
```bash
# Standard (in config.yaml definiert)
python src/main.py --market germany

# Preprocessing überspringen (schneller)
python src/main.py --market germany --skip-prep

# Ohne Visualisierungen
python src/main.py --market germany --skip-plots
```

---

## 📁 Projektstruktur

```
masterarbeit-kennzahlenanalyse/
├── config.yaml                 # ⚙️ HAUPTKONFIGURATION
├── CONFIG_GUIDE.md            # 📘 Ausführliche Dokumentation
│
├── data/
│   ├── raw/germany/           # 📥 Input: DAX, MDAX, SDAX CSVs
│   └── processed/germany/     # 🔄 Verarbeitete Features
│
├── output/germany/
│   ├── algorithms/            # 📊 Individuelle Algorithmen-Ergebnisse
│   │   ├── kmeans/
│   │   ├── hierarchical/
│   │   └── dbscan/
│   └── comparisons/           # 🔬 VERGLEICHE (NEU!)
│       ├── 01_gics_comparison/
│       ├── 02_algorithm_comparison/
│       ├── 03_feature_importance/
│       └── 04_temporal_stability/
│
└── src/
    ├── main.py                # 🎬 Entry Point
    ├── pipeline.py            # 🔄 Haupt-Pipeline
    ├── comparison_pipeline.py # 🔬 Vergleichs-Pipeline
    ├── clustering_engine.py   # 🧮 Clustering-Logik
    └── comparison/            # 📊 Vergleichs-Module
```

---

## ⚙️ Konfiguration

### Algorithm wählen (für Single Mode)
```yaml
# config.yaml
classification:
  algorithm: 'kmeans'  # Optionen: kmeans, hierarchical, dbscan
```

**⚠️ Im Comparison Mode (`--compare`) wird diese Einstellung ignoriert - alle Algorithmen laufen!**

### Features anpassen
```yaml
static_analysis:
  features:
    - roa                    # Return on Assets
    - roe                    # Return on Equity
    - ebit_margin            # EBIT Margin
    - debt_to_equity         # Verschuldung
    - equity_ratio           # Eigenkapitalquote
```

**Verfügbare Features:**
- **Profitabilität**: `roa`, `roe`, `ebit_margin`, `ebitda_margin`, `net_profit_margin`
- **Liquidität**: `current_ratio`, `quick_ratio`, `cash_ratio`
- **Verschuldung**: `debt_to_equity`, `total_debt_to_equity`, `equity_ratio`, `debt_ratio`
- **Effizienz**: `asset_turnover`, `revenue_per_employee`, `receivables_turnover`
- **Wachstum**: `revenue_growth`, `asset_growth`, `employee_growth`

👉 **Detaillierte Anleitung:** Siehe [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

### Cluster-Anzahl anpassen
```yaml
static_analysis:
  n_clusters: 5      # Empfohlen: 3-7

dynamic_analysis:
  n_clusters: 5

combined_analysis:
  n_clusters: 6
  weights:
    static: 0.4      # 40% aktueller Zustand
    dynamic: 0.6     # 60% Entwicklung
```

---

## 🔬 Comparison Mode (NEU!)

### Was macht der Comparison Mode?

Führt **4 umfassende Analysen** durch:

#### 1️⃣ **GICS Comparison** - Branchen-Unabhängigkeit
```
Testet ob Cluster der Branchen-Logik folgen oder eigene Muster finden

Metriken:
- Cramér's V (< 0.3 = gut, unabhängig von Branchen)
- Adjusted Rand Index
- Chi²-Test

Output:
- cramers_v.csv
- Contingency Tables (Heatmaps)
```

**Beispiel-Ergebnis:**
```
K-Means:      Cramér's V = 0.350 (moderate Korrelation)
Hierarchical: Cramér's V = 0.275 (schwache Korrelation ✓)
DBSCAN:       Cramér's V = 0.704 (starke Korrelation)

→ Hierarchical zeigt beste Branchen-Unabhängigkeit!
```

#### 2️⃣ **Algorithm Comparison** - Algorithmen-Vergleich
```
Vergleicht K-Means, Hierarchical & DBSCAN direkt

Metriken:
- Silhouette Score (höher = besser)
- Davies-Bouldin Index (niedriger = besser)
- Cluster-Überlappung (Adjusted Rand Index)
```

#### 3️⃣ **Feature Importance** - Wichtigste Kennzahlen
```
Zeigt welche Features die Cluster am besten trennen

Methode: Random Forest Classifier
Output: Top 15 Features pro Algorithmus
```

#### 4️⃣ **Temporal Stability** - Zeitliche Stabilität
```
Analysiert wie stabil Cluster über Jahre sind

Metriken:
- Cluster Consistency Rate
- Migration Matrices (Jahr-zu-Jahr)
```

### Usage
```bash
# Alle Algorithmen + alle Vergleiche
python src/main.py --market germany --compare

# Nur 2 Algorithmen vergleichen
python src/main.py --market germany --compare --algorithms kmeans hierarchical

# Ohne Plots (schneller)
python src/main.py --market germany --compare --skip-plots
```

---

## 📊 Output-Struktur

### Einzelner Algorithmus
```
output/germany/kmeans/
├── static/
│   ├── reports/
│   │   ├── data/
│   │   │   ├── assignments.csv      # Cluster-Zuordnungen
│   │   │   ├── profiles.csv         # Cluster-Profile
│   │   │   └── metrics.json         # Metriken
│   │   ├── clusters/                # CSV pro Cluster
│   │   └── models/                  # Gespeicherte Modelle
│   └── visualizations/              # Plots
├── dynamic/
└── combined/
```

### Comparison Mode Output
```
output/germany/comparisons/
├── 01_gics_comparison/
│   ├── cramers_v.csv
│   ├── contingency_tables/*.png
│   └── summary_gics_static.png
├── 02_algorithm_comparison/
│   ├── metrics_comparison.csv
│   └── *.png
├── 03_feature_importance/
└── 04_temporal_stability/
```

---

## 🛠️ CLI Options

```bash
# Grundlegende Options
python src/main.py --market germany              # Standard
python src/main.py --skip-prep                   # Skip Preprocessing
python src/main.py --skip-plots                  # Keine Visualisierungen

# Analysis Type
python src/main.py --only-static                 # Nur Static Analysis
python src/main.py --only-dynamic                # Nur Dynamic Analysis

# Comparison Mode
python src/main.py --compare                     # Alle Algorithmen + Vergleiche
python src/main.py --compare --algorithms kmeans hierarchical
```

---

## ❓ FAQ

### F: Kann ich mehrere Algorithmen gleichzeitig in config.yaml angeben?

**A: NEIN** - für Single Mode nur ein Algorithmus.

```yaml
# ❌ FALSCH
classification:
  algorithm: 'kmeans, hierarchical, dbscan'

# ✅ RICHTIG (Single Mode)
classification:
  algorithm: 'kmeans'

# ✅ RICHTIG (Comparison Mode)
python src/main.py --compare  # Ignoriert config, führt alle aus
```

### F: Wie wähle ich die beste Cluster-Anzahl?

**A:** 3 Methoden:

1. **Elbow-Methode** - Schaue dir den Plot an
2. **Silhouette Score** - Probiere 3-7 Cluster, wähle höchsten Score
3. **Domain Knowledge** - Bei Finanzkennzahlen sind 4-6 Cluster typisch

### F: DBSCAN findet nur Noise - was tun?

**A:** `eps` Parameter erhöhen:
```yaml
dbscan:
  eps: 1.5  # War 0.5, probiere größere Werte
```

---

## 📖 Weiterführende Dokumentation

- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Ausführliche Config-Dokumentation mit allen Details
- **[config.yaml](config.yaml)** - Hauptkonfiguration mit Kommentaren

---

## 🏆 Empfohlener Workflow für Masterarbeit

```bash
# 1. GICS-Vergleich durchführen (Hauptanalyse)
python src/main.py --market germany --compare

# 2. Ergebnisse prüfen
cat output/germany/comparisons/01_gics_comparison/cramers_v.csv

# 3. Besten Algorithmus wählen (z.B. Hierarchical)
vim config.yaml  # Setze algorithm: 'hierarchical'
python src/main.py --market germany

# 4. Cluster interpretieren
cat output/germany/hierarchical/static/reports/data/profiles.csv
```

**Happy Clustering! 🎯**
