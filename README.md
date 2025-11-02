# 🎯 Financial Clustering Analysis Pipeline

Clustering-basierte Unternehmensanalyse mit **3 Algorithmen** (K-Means, Hierarchical, DBSCAN) und **3 Analyse-Stadien** (Static, Dynamic, Combined).

---

## 🚀 Schnellstart

### Methode 1: Automatisches Setup (EMPFOHLEN)

```bash
# 1. Repository klonen
git clone <your-repo-url>
cd Masterarbeit-Kennzahlenanalyse

# 2. Setup starten (erstellt automatisch venv + installiert dependencies)
python3 src/main.py --market germany

# 3. Folge den Anweisungen im Menü:
#    → Wähle Option [1] "Automatisch installieren"
#    → Warte bis Installation fertig ist

# 4. Aktiviere venv und starte Pipeline:
source venv/bin/activate
python src/main.py --market germany --compare
```

### Methode 2: Manuelles Setup

```bash
# 1. Repository klonen
git clone <your-repo-url>
cd Masterarbeit-Kennzahlenanalyse

# 2. Virtuelle Umgebung erstellen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Pipeline starten
python src/main.py --market germany --compare
```

### Methode 3: One-Click Setup Script

```bash
# 1. Repository klonen und setup.sh ausführen
git clone <your-repo-url>
cd Masterarbeit-Kennzahlenanalyse
bash setup.sh

# 2. Pipeline starten (venv ist bereits aktiv!)
python src/main.py --market germany --compare
```

### 🎯 Pipeline-Befehle

**⚠️ WICHTIG: Virtuelle Umgebung MUSS aktiviert sein!**

```bash
# JEDES Mal wenn du ein neues Terminal öffnest:
source venv/bin/activate

# KOMPLETT-DURCHLAUF: Alle Algorithmen mit allen Visualisierungen (EMPFOHLEN)
python src/main.py --market germany --compare

# Einzelner Algorithmus
python src/main.py --market germany

# Schneller Durchlauf (ohne Preprocessing & Plots)
python src/main.py --market germany --compare --skip-prep --skip-plots
```

**Output:**
- ✅ Cluster-Analysen (Static, Dynamic, Combined)
- ✅ GICS-Vergleich (Branchen-Unabhängigkeit)
- ✅ Algorithmen-Vergleich (K-Means vs Hierarchical vs DBSCAN)
- ✅ 5 Visualisierungen pro Analyse (Verteilung, Charakteristika, PCA, Correlation Heatmap, Performance Dashboard)

---

## 📁 Output-Struktur

Die Pipeline erstellt zwei verschiedene Strukturen je nach Algorithmus:

### K-Means: Comparative Mode (3 unabhängige Clusterings)
```
output/germany/
└── 02_algorithms/
    └── kmeans_comparative/
        ├── static/          # Static-Clustering
        ├── dynamic/         # Dynamic-Clustering (unabhängig)
        └── combined/        # Combined-Clustering (unabhängig)
```

### Hierarchical/DBSCAN: Hierarchical Mode (1 Master-Clustering)
```
output/germany/
└── 02_algorithms/
    └── hierarchical/
        ├── master_clustering/     # Static → Master-Labels
        ├── dynamic_enrichment/    # Dynamic → Gleiche Labels + Scores
        └── combined_scores/       # Combined → Gleiche Labels + Combined Score
```

### Vergleiche (bei --compare)
```
output/germany/
└── 03_comparisons/
    ├── algorithms/        # Algorithmen-Vergleich
    ├── gics/             # GICS-Branchen-Analyse
    ├── features/         # Feature Importance
    └── temporal/         # Zeitliche Stabilität
```

Jede Analyse enthält:
- `data/` - CSVs (assignments, profiles, metrics)
- `plots/` - 5 Visualisierungen
- `reports/clusters/` - CSVs pro Cluster
- `models/` - Gespeicherte Modelle (Scaler, KMeans)

---

## ⚙️ Konfiguration

### Features auswählen

**2 Modi verfügbar in `config.yaml`:**

```yaml
# MODE 1: Preset (empfohlen)
feature_selection:
  mode: 'preset'
  preset: 'pca_optimized'  # minimal, standard, comprehensive, pca_optimized

# MODE 2: Manual
feature_selection:
  mode: 'manual'

static_analysis:
  features:
    - roa, roe, ebit_margin        # Profitabilität
    - current_ratio, equity_ratio  # Liquidität & Leverage
    - debt_to_equity              # Verschuldung
```

**Verfügbare Features (46 gesamt):**

**Static (37):**
- **Profitabilität**: roa, roe, ebit_margin, operating_margin, gross_margin, roc
- **Liquidität**: current_ratio, quick_ratio, cash_ratio, working_capital_ratio
- **Leverage**: debt_to_equity, debt_to_assets, equity_ratio, interest_coverage
- **Effizienz**: asset_turnover, capital_intensity, inventory_turnover
- **Cashflow**: fcf_margin, capex_to_revenue, cash_conversion
- **Struktur**: financial_leverage, rnd_intensity, dividend_payout_ratio

**Dynamic (14):**
- **Growth**: revenue_cagr, roa_cagr, fcf_growth, capex_growth
- **Trends**: margin_trend, leverage_trend, capex_trend, fcf_trend
- **Volatility**: margin_volatility, leverage_volatility, cashflow_volatility
- **Quality**: margin_consistency, growth_quality

Siehe `features_config.yaml` für Formeln und Details.

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

### Algorithm wählen

```yaml
classification:
  algorithm: 'kmeans'  # Optionen: kmeans, hierarchical, dbscan
```

**⚠️ Im Comparison Mode (`--compare`) werden alle Algorithmen ausgeführt!**

---

## 🔬 Comparison Mode

### Was macht der Comparison Mode?

Führt **4 umfassende Analysen** durch:

#### 1️⃣ GICS Comparison - Branchen-Unabhängigkeit
Testet ob Cluster der Branchen-Logik folgen oder eigene Muster finden.

**Metriken:**
- Cramér's V (< 0.3 = gut, unabhängig von Branchen)
- Adjusted Rand Index
- Chi²-Test

**Output:** `output/germany/03_comparisons/gics/`

#### 2️⃣ Algorithm Comparison - Algorithmen-Vergleich
Vergleicht K-Means, Hierarchical & DBSCAN direkt.

**Metriken:**
- Silhouette Score (höher = besser)
- Davies-Bouldin Index (niedriger = besser)
- Cluster-Überlappung (Adjusted Rand Index)

#### 3️⃣ Feature Importance - Wichtigste Kennzahlen
Zeigt welche Features die Cluster am besten trennen (Random Forest).

#### 4️⃣ Temporal Stability - Zeitliche Stabilität
Analysiert wie stabil Cluster über Jahre sind.

### Usage

```bash
# Alle Algorithmen + alle Vergleiche
python src/main.py --market germany --compare

# Nur 2 Algorithmen vergleichen
python src/main.py --compare --algorithms kmeans hierarchical

# Ohne Plots (schneller)
python src/main.py --compare --skip-plots
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

## 📊 Visualisierungen

Pro Analyse (static/dynamic/combined) werden 5 Plots erstellt:

1. **Cluster Distribution** - Balkendiagramm der Cluster-Größen
2. **Cluster Characteristics** - Grouped Bar Chart der Top-Kennzahlen
3. **PCA Clusters** - 2D Projektion der Cluster
4. **Correlation Heatmap** - Korrelationen zwischen Top-15 Features
5. **Performance Dashboard** - 6-Panel Dashboard mit Überblick

Alle Plots: `output/germany/02_algorithms/{algorithm}/{analysis}/plots/`

---

## ❓ FAQ

### F: Welcher Algorithmus ist der beste?

**A:** Kommt auf den Use-Case an:

- **K-Means:** Schnell, gut für klar getrennte Cluster, vergleicht 3 unabhängige Sichtweisen
- **Hierarchical:** Konsistente Labels über alle Analysen, gut für Unternehmensbewertung mit Scores
- **DBSCAN:** Findet Ausreißer, aber eps-Parameter schwierig zu tunen

→ **Empfehlung:** Nutze `--compare` und vergleiche die Cramér's V Werte!

### F: Preprocessing dauert lange - wie skippen?

**A:** Nach dem ersten Durchlauf:
```bash
python src/main.py --market germany --compare --skip-prep
```

Processed Features werden gespeichert in: `data/processed/germany/features.csv`

### F: DBSCAN findet nur Noise - was tun?

**A:** `eps` Parameter erhöhen:
```yaml
dbscan:
  eps: 1.5  # Standard: 0.5, probiere 1.0-2.0
```

### F: Wie viele Features soll ich nutzen?

**A:** Für PCA-Analyse:
- **Minimum:** 15-20 static, 8-10 dynamic
- **Empfohlen:** Nutze `preset: 'pca_optimized'` (21 static, 10 dynamic)
- **Maximum:** `preset: 'comprehensive'` (51 features)

---

## 🏆 Empfohlener Workflow für Masterarbeit

```bash
# 1. KOMPLETT-DURCHLAUF: Alle Algorithmen + GICS-Vergleich
python src/main.py --market germany --compare

# 2. Ergebnisse prüfen
cat output/germany/03_comparisons/gics/cramers_v.csv

# 3. Besten Algorithmus wählen (z.B. Hierarchical)
# → Setze in config.yaml: algorithm: 'hierarchical'

# 4. Detaillierte Analyse mit bestem Algorithmus
python src/main.py --market germany

# 5. Cluster interpretieren
cat output/germany/02_algorithms/hierarchical/master_clustering/reports/data/profiles.csv

# 6. Visualisierungen prüfen
open output/germany/02_algorithms/hierarchical/master_clustering/plots/
```

---

## 📁 Projektstruktur

```
masterarbeit-kennzahlenanalyse/
├── config.yaml                 # ⚙️ HAUPTKONFIGURATION
├── features_config.yaml        # 📋 Feature-Definitionen (46 Kennzahlen)
│
├── data/
│   ├── raw/germany/           # 📥 Input: DAX, MDAX, SDAX CSVs
│   └── processed/germany/     # 🔄 Verarbeitete Features (149 Spalten)
│
├── output/germany/
│   ├── 01_data/               # Processed Features
│   ├── 02_algorithms/         # Pro Algorithmus
│   │   ├── kmeans_comparative/
│   │   ├── hierarchical/
│   │   └── dbscan/
│   ├── 03_comparisons/        # Cross-Analysen
│   └── 99_summary/            # Executive Reports
│
└── src/
    ├── main.py                        # 🎬 Entry Point
    ├── _01_setup/
    │   ├── config_loader.py           # Config & Feature Selection
    │   ├── feature_config_loader.py   # Feature Preset Loader
    │   └── output_handler.py          # Output Management (Option B)
    ├── _02_processing/
    │   ├── data_loader.py             # WRDS Compustat Loader
    │   └── feature_engineer.py        # 46 Kennzahlen berechnen
    ├── _03_clustering/
    │   ├── pipeline.py                # K-Means (Comparative Mode)
    │   ├── hierarchical_pipeline.py   # Hierarchical/DBSCAN (Label-Consistency)
    │   └── cluster_engine.py          # Clustering-Algorithmen
    ├── _04_comparison/
    │   └── comparison_pipeline.py     # GICS + Algorithm Comparison
    └── _05_visualization/
        ├── cluster_visualizer.py      # 5 Cluster-Plots
        └── comparison_visualizer.py   # Comparison-Plots
```

---

## 🔧 VS Code Setup

Falls du in VS Code auf "Play" drücken möchtest:

1. **Cmd + Shift + P**
2. Tippe: **"Python: Select Interpreter"**
3. Wähle: **`venv/bin/python`**

Dann kannst du [src/main.py](src/main.py) öffnen und auf ▶️ Play drücken!

---

## ⚠️ Troubleshooting

### Setup-Menü: Welche Option wählen?

Das Setup-Menü passt sich automatisch an deine Situation an:

#### Szenario 1: venv existiert NICHT (frisch geklontes Repo)
```
Optionen:
[1] Automatisch installieren (EMPFOHLEN)
[2] Manuelle Anweisungen anzeigen
[3] Dependencies prüfen
[4] Abbrechen
```
→ **Wähle [1]** für automatische Installation

#### Szenario 2: venv existiert, ist aber NICHT aktiviert
```
✓ venv/ gefunden im Projekt-Ordner!
💡 Du hast vergessen die venv zu aktivieren.

Optionen:
[1] venv aktivieren (EMPFOHLEN)
[2] Neu installieren
[3] Dependencies prüfen
[4] Abbrechen
```
→ **Wähle [1]** für Aktivierungsanleitung
→ **Wähle [2]** nur bei Problemen (löscht venv/ komplett!)

### "Keine venv aktiv" Meldung erscheint immer wieder

**Problem:** Du hast vergessen die venv zu aktivieren!

**Lösung:** JEDES Mal wenn du ein neues Terminal öffnest:
```bash
source venv/bin/activate
```

**Tipp:** Prüfe ob venv aktiv ist - dein Prompt sollte `(venv)` zeigen:
```bash
(venv) user@mac Masterarbeit-Kennzahlenanalyse %
```

### VS Code zeigt Setup-Menü obwohl venv aktiv ist

**Problem:** VS Code nutzt falschen Python-Interpreter

**Lösung:**
1. **Cmd + Shift + P**
2. Tippe: **"Python: Select Interpreter"**
3. Wähle: **`venv/bin/python`** (im Projekt-Ordner)

### Installation schlägt fehl: "No module named 'pip'"

**Problem:** Python venv ohne pip installiert

**Lösung:**
```bash
# Lösche alte venv
rm -rf venv

# Erstelle neue mit --upgrade-deps
python3 -m venv venv --upgrade-deps
source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied" beim Setup

**Problem:** setup.sh hat keine Ausführungsrechte

**Lösung:**
```bash
chmod +x setup.sh
bash setup.sh
```

### Dependencies installiert, aber Import-Fehler

**Problem:** Falscher Python-Interpreter wird verwendet

**Lösung:** Prüfe welcher Python aktiv ist:
```bash
which python
# Sollte zeigen: /pfad/zu/projekt/venv/bin/python

# Falls nicht:
source venv/bin/activate
which python  # Erneut prüfen
```

---

**Happy Clustering! 🎯**
