# 📘 Configuration Guide - Clustering Analysis Pipeline

## Inhaltsverzeichnis
1. [Schnellstart](#schnellstart)
2. [Pipeline Modi](#pipeline-modi)
3. [Config-Datei Struktur](#config-datei-struktur)
4. [Algorithmus-Auswahl](#algorithmus-auswahl)
5. [Feature-Anpassung](#feature-anpassung)
6. [Cluster-Anzahl & Parameter](#cluster-anzahl--parameter)
7. [Häufige Anwendungsfälle](#häufige-anwendungsfälle)
8. [Troubleshooting](#troubleshooting)

---

## Schnellstart

### Einzelner Algorithmus
```bash
# K-Means (Standard)
python src/main.py --market germany

# Hierarchical Clustering
# Ändere in config.yaml: algorithm: 'hierarchical'
python src/main.py --market germany

# DBSCAN
# Ändere in config.yaml: algorithm: 'dbscan'
python src/main.py --market germany
```

### Alle Algorithmen vergleichen (EMPFOHLEN!)
```bash
# Führt K-Means, Hierarchical & DBSCAN aus + Vergleiche
python src/main.py --market germany --compare

# Nur bestimmte Algorithmen
python src/main.py --market germany --compare --algorithms kmeans hierarchical
```

### Weitere Optionen
```bash
# Skip Preprocessing (schneller bei wiederholten Läufen)
python src/main.py --market germany --skip-prep

# Keine Plots (nur Daten)
python src/main.py --market germany --skip-plots

# Nur Static Analysis
python src/main.py --market germany --only-static

# Nur Dynamic Analysis
python src/main.py --market germany --only-dynamic
```

---

## Pipeline Modi

### 🔬 **Comparison Mode** (HAUPTMODUS für deine Masterarbeit!)

Der Comparison Mode führt **alle Algorithmen** aus und erstellt umfassende Vergleiche:

```bash
python src/main.py --market germany --compare
```

**Output:**
```
output/germany/
├── algorithms/              # Individuelle Ergebnisse
│   ├── kmeans/
│   ├── hierarchical/
│   └── dbscan/
├── comparisons/             # VERGLEICHE (NEU!)
│   ├── 01_gics_comparison/       # Branchen-Unabhängigkeit
│   ├── 02_algorithm_comparison/  # Algorithmen-Vergleich
│   ├── 03_feature_importance/    # Wichtigste Kennzahlen
│   └── 04_temporal_stability/    # Zeitliche Stabilität
```

**Wann verwenden?**
- ✅ Für wissenschaftliche Analyse (Masterarbeit)
- ✅ Um besten Algorithmus zu finden
- ✅ Um GICS-Unabhängigkeit zu beweisen
- ✅ Um Feature Importance zu verstehen

### 📊 **Single Mode** (Einzelner Algorithmus)

Führt nur einen Algorithmus aus:

```bash
python src/main.py --market germany
# Algorithmus in config.yaml festgelegt
```

**Wann verwenden?**
- ✅ Schnelle Tests
- ✅ Fokus auf einen Algorithmus
- ✅ Produktivbetrieb

---

## Config-Datei Struktur

Die `config.yaml` ist in **6 Bereiche** unterteilt:

### 1️⃣ **Algorithm Selection**

```yaml
classification:
  algorithm: 'kmeans'  # Welcher Algorithmus für Single Mode?
```

**Verfügbare Werte:**
- `'kmeans'` - K-Means Clustering (Standard)
- `'hierarchical'` - Hierarchisches Clustering (Ward's Methode)
- `'dbscan'` - Density-Based Clustering

**⚠️ WICHTIG für Comparison Mode:**
Diese Einstellung wird **ignoriert** im `--compare` Mode!
Im Comparison Mode laufen **alle** Algorithmen automatisch.

---

### 2️⃣ **Algorithm-Specific Parameters**

Jeder Algorithmus hat eigene Parameter:

#### **K-Means**
```yaml
kmeans:
  n_init: 20          # Wie oft neu initialisieren? (höher = stabiler, langsamer)
  max_iter: 300       # Max Iterationen
  algorithm: 'lloyd'  # Algorithmus-Variante
```

**Wann anpassen?**
- `n_init: 50` - Bei instabilen Ergebnissen
- `max_iter: 500` - Bei Konvergenz-Warnungen

#### **Hierarchical**
```yaml
hierarchical:
  linkage: 'ward'              # Linkage-Methode
  distance_metric: 'euclidean' # Distanzmetrik
```

**Linkage-Optionen:**
- `'ward'` - Minimiert Varianz (EMPFOHLEN, Standard)
- `'complete'` - Maximum Distance
- `'average'` - Average Distance
- `'single'` - Minimum Distance

**⚠️ Bei Ward MUSS `distance_metric: 'euclidean'` sein!**

#### **DBSCAN**
```yaml
dbscan:
  eps: 0.5           # Maximaler Abstand zwischen Punkten
  min_samples: 5     # Min. Punkte für Core-Point
  metric: 'euclidean'
```

**Kritischste Parameter:**
- `eps`: **MUSS angepasst werden!**
  - Zu klein → Viele Noise-Punkte
  - Zu groß → Nur 1 großer Cluster
  - **Empfehlung**: 0.3 - 2.0, starte mit 0.5
  - Verwende Elbow-Plot zur Bestimmung

- `min_samples`:
  - Klein (3-5): Mehr kleine Cluster
  - Groß (10+): Nur dichte Cluster

**Tipp:** DBSCAN findet Cluster-Anzahl automatisch!

---

### 3️⃣ **Cluster Count (n_clusters)**

```yaml
static_analysis:
  n_clusters: 5      # Anzahl Cluster für Static Analysis

dynamic_analysis:
  n_clusters: 5      # Anzahl Cluster für Dynamic Analysis
  min_years_required: 5  # Min. Jahre pro Unternehmen

combined_analysis:
  n_clusters: 6      # Anzahl Cluster für Combined
  weights:
    static: 0.4      # 40% Gewicht auf aktuellen Zustand
    dynamic: 0.6     # 60% Gewicht auf Entwicklung
```

**Wie wähle ich n_clusters?**

1. **Elbow-Methode** (sieh dir Plots an):
   - 3 Cluster: Sehr grob (High/Mid/Low)
   - 5 Cluster: Ausgewogen (EMPFOHLEN)
   - 7+ Cluster: Sehr fein

2. **Domain Knowledge**:
   - Finanzkennzahlen: 4-6 ist typisch
   - Zu viele → schwer interpretierbar
   - Zu wenige → verliert Details

3. **Silhouette Score** (in Output):
   - Höher = besser
   - Vergleiche verschiedene n_clusters

**Combined Analysis Weights:**
```yaml
weights:
  static: 0.4    # Aktuelle Performance
  dynamic: 0.6   # Entwicklung über Zeit
```

- **0.5 / 0.5** - Gleichgewichtet
- **0.7 / 0.3** - Fokus auf aktuellen Zustand
- **0.3 / 0.7** - Fokus auf Trends (EMPFOHLEN für Wachstumsanalyse)

---

### 4️⃣ **Features (Die wichtigste Einstellung!)**

Features bestimmen **welche Kennzahlen** für Clustering verwendet werden.

#### **Static Analysis Features**
```yaml
static_analysis:
  features:
    - roa                    # Return on Assets
    - roe                    # Return on Equity
    - ebit_margin            # EBIT Margin
    - net_profit_margin      # Net Profit Margin
    - current_ratio          # Liquidität
    - debt_to_equity         # Verschuldung
    - equity_ratio           # Eigenkapitalquote
```

**Verfügbare Features (Profitabilität):**
- `roa` - Return on Assets
- `roe` - Return on Equity
- `ebit_margin` - EBIT Margin
- `ebitda_margin` - EBITDA Margin
- `net_profit_margin` - Net Profit Margin

**Liquidität:**
- `current_ratio` - Current Ratio
- `quick_ratio` - Quick Ratio
- `cash_ratio` - Cash Ratio

**Verschuldung:**
- `debt_to_equity` - Debt-to-Equity Ratio
- `total_debt_to_equity` - Total Debt-to-Equity
- `equity_ratio` - Equity Ratio
- `debt_ratio` - Debt Ratio

**Effizienz:**
- `asset_turnover` - Asset Turnover
- `revenue_per_employee` - Revenue per Employee
- `receivables_turnover` - Receivables Turnover
- `days_sales_outstanding` - Days Sales Outstanding

**Wachstum:**
- `revenue_growth` - Revenue Growth
- `asset_growth` - Asset Growth
- `employee_growth` - Employee Growth

#### **Dynamic Analysis Features**

```yaml
dynamic_analysis:
  features:
    - roa          # Wird zu: roa_trend, roa_volatility, roa_cagr
    - roe
    - ebit_margin
    - revt         # Revenue
    - debt_to_equity
```

**⚠️ WICHTIG:**
Bei Dynamic werden **automatisch** 3 Varianten erstellt:
- `roa` → `roa_trend`, `roa_volatility`, `roa_cagr`
- `roe` → `roe_trend`, `roe_volatility`, `roe_cagr`
- etc.

Du musst **nur die Basis-Kennzahl** angeben!

#### **Combined Analysis Features**

```yaml
combined_analysis:
  features_static:     # Aktuelle Werte
    - roa
    - roe
    - ebit_margin

  features_dynamic:    # Entwicklung
    - roa_trend
    - roa_volatility
    - roe_trend
    - revt_cagr
```

Hier kannst du **genau steuern** welche Kennzahlen kombiniert werden.

---

### 5️⃣ **Data & Market Settings**

```yaml
data:
  market: 'germany'      # Market identifier
  input_dir: 'data/raw'  # Wo liegen die Raw-Daten?
  output_dir: 'output'   # Wo speichern?
```

**Markets:**
- `'germany'` - Deutsche Unternehmen (DAX, MDAX, SDAX)
- `'usa'` - US Unternehmen
- Beliebige andere (Ordner in `data/raw/`)

**Ordnerstruktur:**
```
data/raw/
├── germany/
│   ├── dax40_proxy.csv
│   ├── mdax_proxy.csv
│   └── sdax_proxy.csv
└── usa/
    └── sp500.csv
```

---

### 6️⃣ **Global Settings**

```yaml
global:
  random_state: 42    # Für reproduzierbare Ergebnisse

output:
  create_plots: true           # Visualisierungen erstellen?
  create_company_lists: true   # CSV pro Cluster?
```

---

## Häufige Anwendungsfälle

### 🎯 **Use Case 1: Masterarbeit - GICS-Unabhängigkeit beweisen**

**Ziel:** Zeigen, dass Cluster **nicht** Branchen folgen

```bash
# 1. Alle Algorithmen laufen lassen
python src/main.py --market germany --compare

# 2. Ergebnisse prüfen
cat output/germany/comparisons/01_gics_comparison/cramers_v.csv
```

**Was suchen?**
- **Niedrige Cramér's V Werte** (< 0.3)
- **Hierarchical** hat typischerweise niedrigste Werte

**Config anpassen:**
```yaml
# Keine Anpassung nötig!
# Vergleich läuft automatisch auf gsector
```

---

### 🎯 **Use Case 2: Profitabilität vs. Verschuldung clustern**

**Ziel:** Unternehmen nach Rentabilität UND Schuldenstand gruppieren

**Config:**
```yaml
static_analysis:
  n_clusters: 5
  features:
    - roa                # Profitabilität
    - roe                # Profitabilität
    - ebit_margin        # Profitabilität
    - debt_to_equity     # Verschuldung
    - equity_ratio       # Verschuldung
```

```bash
python src/main.py --market germany
```

**Erwartete Cluster:**
- Cluster 0: High Profit, Low Debt (💎 Top Tier)
- Cluster 1: High Profit, High Debt (⚠️ Aggressive Growth)
- Cluster 2: Mid Profit, Mid Debt (📊 Balanced)
- Cluster 3: Low Profit, High Debt (🔴 Challenged)
- Cluster 4: Low Profit, Low Debt (😴 Underperformers)

---

### 🎯 **Use Case 3: Wachstums-Champions finden**

**Ziel:** Unternehmen mit starkem Wachstum identifizieren

**Config:**
```yaml
dynamic_analysis:
  n_clusters: 4
  features:
    - roa          # Profitabilität Trend
    - revt         # Revenue Trend
    - at           # Asset Growth

combined_analysis:
  n_clusters: 5
  weights:
    static: 0.3   # Weniger Gewicht auf aktuellen Zustand
    dynamic: 0.7  # MEHR Gewicht auf Entwicklung!

  features_dynamic:
    - roa_trend         # Profitabilität steigt?
    - revt_cagr         # Umsatz wächst stark?
    - roa_volatility    # Stabil oder volatil?
```

```bash
python src/main.py --market germany
```

**Cluster-Interpretation:**
- Hoher `revt_cagr` + Hoher `roa_trend` = **Wachstums-Champions** ⭐
- Hoher `revt_cagr` + Negativer `roa_trend` = **Aggressive Expander** (Risiko!)

---

### 🎯 **Use Case 4: Konservative vs. Aggressive Strategie**

**Ziel:** Unterscheide zwischen konservativen und risikofreudigen Unternehmen

**Config:**
```yaml
static_analysis:
  features:
    - debt_to_equity          # Verschuldung
    - current_ratio           # Liquidität
    - cash_ratio              # Cash-Position
    - equity_ratio            # Eigenkapital
    - receivables_turnover    # Effizienz

dynamic_analysis:
  features:
    - debt_to_equity    # Verschuldung ändert sich?
    - revt              # Wachstum
```

**Erwartete Cluster:**
- Niedriger `debt_to_equity` + Hoher `current_ratio` = **Konservativ** 🛡️
- Hoher `debt_to_equity` + Hoher `revt_cagr` = **Aggressiv/Wachstum** 🚀

---

### 🎯 **Use Case 5: Nur große Unternehmen (Filter)**

**Problem:** DBSCAN klassifiziert kleine Unternehmen als Noise

**Lösung 1: DBSCAN Parameter anpassen**
```yaml
dbscan:
  eps: 1.0              # GRÖßER = mehr Unternehmen in Clustern
  min_samples: 3        # KLEINER = weniger Noise
```

**Lösung 2: Pre-Processing Filter (TODO)**
Aktuell nicht in Pipeline, aber könnte hinzugefügt werden:
- Filter nach `at` (Total Assets) > 1000
- Filter nach `emp` (Employees) > 100

---

## Troubleshooting

### ❌ **Problem: "Keine Features verfügbar"**

**Ursache:** Feature-Namen in config.yaml stimmen nicht mit berechneten Features überein

**Lösung:**
```bash
# Prüfe welche Features berechnet wurden
head -1 data/processed/germany/features.csv

# Passe config.yaml an
vim config.yaml
```

**Häufige Tippfehler:**
- ❌ `return_on_assets` → ✅ `roa`
- ❌ `debt_equity_ratio` → ✅ `debt_to_equity`
- ❌ `EBIT_margin` → ✅ `ebit_margin` (lowercase!)

---

### ❌ **Problem: "DBSCAN findet 0 Cluster / alles Noise"**

**Ursache:** `eps` zu klein

**Lösung:**
```yaml
dbscan:
  eps: 2.0      # Größer machen (war 0.5)
  min_samples: 3  # Oder kleiner machen
```

**Empirisches Vorgehen:**
1. Starte mit `eps: 0.5`
2. Wenn > 50% Noise → erhöhe auf `eps: 1.0`
3. Wenn > 50% Noise → erhöhe auf `eps: 2.0`
4. Wiederhole bis < 20% Noise

---

### ❌ **Problem: "Zu viele Cluster (10+) bei DBSCAN"**

**Ursache:** `eps` zu klein ODER `min_samples` zu klein

**Lösung:**
```yaml
dbscan:
  eps: 0.8           # GRÖßER
  min_samples: 10    # GRÖßER
```

---

### ❌ **Problem: "Silhouette Score sehr niedrig (< 0.2)"**

**Ursache:** Schlechte Feature-Wahl ODER falsche Cluster-Anzahl

**Lösung 1: Weniger Features**
```yaml
# Vorher: 10 Features → Nachher: 5 Features
static_analysis:
  features:
    - roa
    - roe
    - debt_to_equity
    - current_ratio
    - ebit_margin
```

**Lösung 2: Andere Cluster-Anzahl**
```yaml
static_analysis:
  n_clusters: 4  # War 5, probiere 4 oder 6
```

**Lösung 3: Anderer Algorithmus**
```bash
# Hierarchical hat oft bessere Silhouette Scores
python src/main.py --market germany --compare
# Prüfe metrics_comparison.csv
```

---

### ❌ **Problem: "Feature Importance zeigt nichts"**

**Status:** Bekanntes Problem - Features werden nicht im Output gespeichert

**Workaround:**
Aktuell nicht verfügbar. Framework ist implementiert, aber Daten-Pipeline muss angepasst werden.

**Alternative:**
Schaue dir die Cluster-Profile an:
```bash
cat output/germany/kmeans/static/reports/data/profiles.csv
```

Die Unterschiede zwischen Clustern zeigen welche Features wichtig sind!

---

### ❌ **Problem: "Temporal Stability zeigt nichts"**

**Status:** Bekanntes Problem - `fyear` nicht im dynamic DataFrame

**Workaround:**
Migration-Analyse funktioniert bereits in der normalen Pipeline:
```bash
python src/main.py --market germany
cat output/germany/kmeans/static/reports/analysis/migration_matrix.csv
```

---

## Zusammenfassung

### ✅ **Quick Reference**

| Was willst du? | Command | Config Change |
|----------------|---------|---------------|
| Alle Algorithmen vergleichen | `--compare` | - |
| Nur K-Means | Normal | `algorithm: 'kmeans'` |
| Nur Hierarchical | Normal | `algorithm: 'hierarchical'` |
| GICS-Vergleich | `--compare` | Automatisch |
| Mehr Cluster | Normal | `n_clusters: 7` |
| Andere Features | Normal | Ändere `features:` Liste |
| Fokus auf Trends | Normal | `dynamic: 0.7` in weights |
| Schneller (skip prep) | `--skip-prep` | - |
| Keine Plots | `--skip-plots` | - |

### 📊 **Empfohlene Workflows**

**Für Masterarbeit:**
```bash
# 1. Einmal vollständig
python src/main.py --market germany --compare

# 2. GICS-Ergebnisse prüfen
cat output/germany/comparisons/01_gics_comparison/cramers_v.csv

# 3. Besten Algorithmus wählen (Hierarchical?)
# Ändere config.yaml: algorithm: 'hierarchical'

# 4. Finale Analyse
python src/main.py --market germany
```

**Für schnelle Experimente:**
```bash
# Features ändern in config.yaml
vim config.yaml

# Schnell testen
python src/main.py --market germany --skip-prep --skip-plots
```

**Für Production:**
```bash
# Voller Lauf mit allen Plots
python src/main.py --market germany
