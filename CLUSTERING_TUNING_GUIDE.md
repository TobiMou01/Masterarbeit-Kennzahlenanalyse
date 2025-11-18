# 🎯 Clustering Tuning Guide - Bessere Ergebnisse erzielen

## 🔴 PROBLEM: Schlechte Clustering-Ergebnisse bei DAX/MDAX/SDAX

**Typische Symptome:**
- Cluster zu groß/zu klein
- Unklare Trennung zwischen Clustern
- Unternehmen landen in "falschen" Clustern
- Niedriger Silhouette Score (<0.3)
- Hohe Intra-Cluster Varianz

---

## ✅ LÖSUNGSANSÄTZE - Die 6 wichtigsten Stellschrauben

### 1️⃣ **Feature Selection** (WICHTIGSTE Stellschraube!)

#### Problem: Zu viele oder falsche Features
```yaml
# ❌ SCHLECHT: Alle 21 Features
feature_selection:
  preset: 'extended'  # Zu viele Features → schlechte Cluster
```

```yaml
# ✅ GUT: Fokus auf die wichtigsten Features
feature_selection:
  preset: 'minimal'  # Nur 6-10 Features → klarere Cluster
```

#### **Erstelle eigene Presets in `features_config.yaml`:**

```yaml
# Beispiel: DAX-optimiertes Preset
feature_presets:
  dax_optimized:
    description: "Optimiert für DAX/MDAX/SDAX Analyse"
    static_features:
      # Profitabilität (wichtigste Dimension)
      - roa
      - ebit_margin
      - roe

      # Leverage (Verschuldung)
      - debt_to_equity
      - equity_ratio

      # Liquidität
      - current_ratio

      # Effizienz
      - asset_turnover

      # Cashflow
      - fcf_margin

    dynamic_features:
      - roa_trend
      - revenue_growth
      - margin_trend
      - leverage_trend
```

#### **Faustregel:**
- **Weniger ist mehr!** Start mit 6-8 Features
- Fokus auf **unkorrelierten** Features
- Teste mit `correlation_threshold: 0.7` in config.yaml

---

### 2️⃣ **Outlier Behandlung**

#### Problem: Extreme Werte verzerren Cluster

```yaml
# config.yaml
preprocessing:
  outlier_detection:
    enabled: true
    method: 'iqr'           # 'iqr' ist robuster als 'zscore'
    iqr_multiplier: 1.5     # Standard (streng)
    # Für mehr Daten: iqr_multiplier: 2.0
```

#### **Tipp:** Nach Outlier-Removal prüfen
```python
# Im Jupyter Notebook (Cell 4)
print(f"Outliers entfernt: {outliers_removed} ({pct:.1f}%)")
# Ziel: 5-15% entfernt ist normal
```

---

### 3️⃣ **PCA - Dimensionsreduktion**

#### Problem: Zu viele Features → "Curse of Dimensionality"

```yaml
# ❌ SCHLECHT: PCA deaktiviert bei vielen Features
pca:
  enabled: false  # Bei >15 Features problematisch!
```

```yaml
# ✅ GUT: PCA aktiviert
pca:
  enabled: true
  n_components: 0.85  # Behalte 85% Varianz
  run_before_clustering: true
```

#### **Wann PCA nutzen?**
- ✅ Bei >12 Features
- ✅ Wenn Features korreliert sind
- ❌ Bei <8 Features (meist nicht nötig)

---

### 4️⃣ **Cluster-Anzahl** (K-Selektion)

#### Problem: Falsche Anzahl Cluster

```yaml
# ❌ SCHLECHT: Fixe Anzahl
cluster_selection:
  mode: 'manual'
  n_clusters: 5  # Nicht optimal für deine Daten!
```

```yaml
# ✅ GUT: Auto-Selektion
cluster_selection:
  mode: 'auto'
  k_range: [3, 8]  # Teste 3-8 Cluster
  methods: ['silhouette', 'elbow']  # Nutze beide Methoden
```

#### **Empfehlung für DAX/MDAX/SDAX (160 Unternehmen):**
- **Zu wenig:** k=2-3 (zu grob)
- **Optimal:** k=4-7 (gute Balance)
- **Zu viel:** k>10 (zu fragmentiert)

---

### 5️⃣ **Preprocessing - Scaling**

#### Problem: Features auf verschiedenen Skalen

```yaml
# ✅ IMMER aktivieren!
preprocessing:
  scaling:
    method: 'standard'  # StandardScaler (mean=0, std=1)
    # Alternative: 'robust' (besser bei Outliern)
```

---

### 6️⃣ **Algorithmus-Wahl**

#### Für verschiedene Cluster-Formen:

```yaml
# K-Means: Gut für kompakte, ähnlich große Cluster
classification:
  algorithm: 'kmeans'
  kmeans:
    n_init: 50  # Mehr Initialisierungen = stabiler (default: 20)

# Hierarchical: Gut für hierarchische Strukturen
classification:
  algorithm: 'hierarchical'
  hierarchical:
    linkage: 'ward'  # Beste Methode für Financial Data

# DBSCAN: Gut für ungleiche Cluster-Größen
classification:
  algorithm: 'dbscan'
  dbscan:
    eps: 1.2  # Passe an! (zu hoch → wenige Cluster, zu niedrig → viel Noise)
    min_samples: 3
```

---

## 📊 **WORKFLOW FÜR BESSERE ERGEBNISSE**

### Schritt 1: Minimal-Config testen

```bash
# Kopiere config.yaml → config_minimal.yaml
cp config.yaml config_minimal.yaml
```

Ändere in `config_minimal.yaml`:
```yaml
feature_selection:
  preset: 'minimal'  # Nur 6-8 Features

pca:
  enabled: false  # Bei wenigen Features nicht nötig

cluster_selection:
  mode: 'auto'
  k_range: [4, 7]  # Fokus auf 4-7 Cluster

preprocessing:
  outlier_detection:
    enabled: true
    iqr_multiplier: 1.5
```

```bash
# Teste
python run.py --config config_minimal.yaml --analyses static
```

---

### Schritt 2: Ergebnisse prüfen

Schaue in `output/germany/02_algorithms/kmeans_comparative/static/`:

```
1_cluster_quality/
├── silhouette_score.txt   # Ziel: >0.4
├── cluster_sizes.csv      # Alle Cluster >10 Unternehmen?
└── plots/
    └── cluster_distribution.png

4_company_insights/
├── data/
│   └── cluster_profiles.csv  # Prüfe: Klare Unterschiede?
```

#### **Gute Indikatoren:**
- ✅ Silhouette Score >0.4
- ✅ Cluster-Größen relativ ausgewogen (nicht 120 + 5 + 5 + ...)
- ✅ Cluster haben klare "Charaktere" (z.B. "High Margin, Low Leverage")

#### **Schlechte Indikatoren:**
- ❌ Silhouette Score <0.3
- ❌ Ein Cluster hat >70% aller Unternehmen
- ❌ Cluster unterscheiden sich kaum in den Kennzahlen

---

### Schritt 3: Iterativ verbessern

#### Wenn Silhouette Score zu niedrig (<0.3):
1. **Reduziere Features** (weniger ist oft mehr!)
2. **Aktiviere PCA** (bei >10 Features)
3. **Passe k-Range an** (teste k=3-8 statt 2-11)

#### Wenn Cluster zu unausgeglichen:
1. **Versuche anderen Algorithmus** (Hierarchical statt K-Means)
2. **Passe outlier_detection an** (mehr/weniger streng)

#### Wenn Cluster inhaltlich unklar:
1. **Weniger Features** verwenden
2. **Fokus auf eine Dimension** (erst nur Profitability, dann erweitern)

---

## 🎛️ **QUICK CONFIGS - Fertige Szenarien**

### Scenario A: "Minimal & Clean" (Empfohlen für Start!)
```bash
python run.py \
  --preset minimal \
  --analyses static \
  --algorithms kmeans
```

### Scenario B: "PCA-optimiert" (Viele Features)
```yaml
# In config.yaml setzen:
feature_selection:
  preset: 'standard'  # 10-15 Features

pca:
  enabled: true
  n_components: 0.90  # Mehr Varianz behalten
```

```bash
python run.py --analyses static dynamic
```

### Scenario C: "Comparison Mode" (Algorithmen vergleichen)
```bash
python run.py \
  --algorithms kmeans hierarchical \
  --analyses static
```

---

## 🔧 **FEATURES CONFIG ANPASSEN**

### Erstelle Custom Preset in `features_config.yaml`:

```yaml
feature_presets:
  # Dein neues Preset
  my_optimized:
    description: "Mein optimiertes Feature-Set für DAX"
    category: custom

    static_features:
      # Wähle NUR die wichtigsten 6-8!
      - roa              # Profitability
      - ebit_margin      # Profitability
      - debt_to_equity   # Leverage
      - equity_ratio     # Leverage
      - current_ratio    # Liquidity
      - asset_turnover   # Efficiency
      - fcf_margin       # Cashflow

    dynamic_features:
      # Trends der wichtigsten Kennzahlen
      - roa_trend
      - revenue_growth
      - margin_trend
      - leverage_trend
```

Dann in `config.yaml`:
```yaml
feature_selection:
  mode: 'preset'
  preset: 'my_optimized'  # Dein Custom Preset
```

---

## 📈 **TYPISCHE PROBLEME & LÖSUNGEN**

### Problem 1: "Ein Cluster hat 80% der Unternehmen"
**Ursache:** Zu viele ähnliche Unternehmen, Features trennen nicht gut
**Lösung:**
1. Andere Features wählen (mehr Varianz)
2. PCA aktivieren
3. k erhöhen (mehr Cluster)

### Problem 2: "Silhouette Score = 0.25"
**Ursache:** Schlechte Trennung zwischen Clustern
**Lösung:**
1. WENIGER Features (z.B. nur 5-7 statt 15)
2. Korrelierte Features entfernen (`correlation_threshold: 0.7`)
3. Outlier strenger behandeln (`iqr_multiplier: 1.2`)

### Problem 3: "DBSCAN findet nur Noise"
**Ursache:** `eps` zu klein
**Lösung:**
```yaml
dbscan:
  eps: 1.5  # Erhöhe von 0.5 auf 1.5
  min_samples: 3
```

### Problem 4: "Verschiedene Runs → verschiedene Cluster"
**Ursache:** K-Means instabil
**Lösung:**
```yaml
kmeans:
  n_init: 100  # Mehr Initialisierungen (default: 20)
```

Oder nutze Hierarchical (deterministisch):
```bash
python run.py --algorithms hierarchical
```

---

## ✅ **CHECKLISTE FÜR OPTIMIERUNG**

- [ ] **Features reduziert** auf 6-10 wichtigste
- [ ] **Korrelation geprüft** (`correlation_threshold: 0.7-0.8`)
- [ ] **Outliers behandelt** (5-15% entfernt)
- [ ] **PCA aktiviert** (bei >12 Features)
- [ ] **Auto k-Selektion** genutzt (`mode: 'auto'`)
- [ ] **Mehrere k getestet** (k_range: [3, 8])
- [ ] **Silhouette Score geprüft** (Ziel: >0.4)
- [ ] **Cluster-Größen validiert** (alle >5-10%)
- [ ] **Cluster-Profile interpretierbar** (klare Charaktere)

---

## 🚀 **NEXT STEPS**

1. **Start mit Minimal Config:**
   ```bash
   python run.py --preset minimal --analyses static
   ```

2. **Prüfe Output:**
   ```bash
   cat output/germany/02_algorithms/kmeans_comparative/static/1_cluster_quality/silhouette_score.txt
   ```

3. **Iteriere:**
   - Silhouette <0.4? → Reduziere Features
   - Cluster unausgewogen? → Andere k-Range
   - Features unklar? → Prüfe Korrelation

4. **Wenn gut (Silhouette >0.4):**
   - Teste `dynamic` und `combined`
   - Teste andere Algorithmen
   - Nutze Jupyter Notebook für interaktive Exploration

---

**Viel Erfolg! Bei Fragen: Schau dir die generierten Plots in `output/{market}/` an** 🎉
