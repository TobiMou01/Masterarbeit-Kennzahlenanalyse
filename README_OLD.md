# Masterarbeit: Clustering-Analyse von Finanzkennzahlen

3-stufige Clustering-Analyse deutscher Unternehmen: **Static** → **Dynamic** → **Combined**

---

## 🚀 Schnellstart

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Vollständige Analyse durchführen
python main.py --market germany

# Fertig! Ergebnisse in: output/germany/
```

---

## 📋 Was macht das Projekt?

Das Projekt analysiert Unternehmen in **3 Stufen**:

1. **Static Analysis** - Querschnitt (aktuelles Jahr)
   - Wie profitabel ist das Unternehmen JETZT?
   - Features: ROA, ROE, EBIT Margin, Debt/Equity, etc.

2. **Dynamic Analysis** - Zeitreihen (Entwicklung)
   - Wie hat sich das Unternehmen ENTWICKELT?
   - Features: Trends, Volatilität, CAGR

3. **Combined Analysis** - Beides kombiniert
   - Welche Unternehmen sind profitabel UND wachsen stabil?
   - Gewichtung: 40% aktuell, 60% Entwicklung (anpassbar)

**Plus:** Cross-Analysis zeigt Cluster-Migration (wer verbessert/verschlechtert sich)

---

## 📁 Output-Struktur

Nach der Analyse:

```
output/germany/
├── data/
│   ├── static_assignments.csv      # Alle Unternehmen mit Clustern
│   ├── static_profiles.csv         # Cluster-Mittelwerte
│   ├── dynamic_assignments.csv
│   ├── dynamic_profiles.csv
│   ├── combined_assignments.csv
│   └── combined_profiles.csv
│
├── clusters/                        # ← WICHTIG: Pro Cluster eine CSV!
│   ├── 0_high_performers.csv       #   Sortiert nach Performance
│   ├── 1_upper_mid.csv
│   ├── 2_mid.csv
│   ├── 3_lower_mid.csv
│   └── 4_low_performers.csv
│
├── analysis/
│   ├── static_results.csv
│   ├── dynamic_results.csv
│   ├── combined_results.csv
│   └── migration_matrix.csv        # Cluster-Wechsel zwischen Analysen
│
├── reports/
│   └── summary.txt                  # Zusammenfassung aller Ergebnisse
│
├── plots/
│   └── *.png                        # Visualisierungen
│
└── models/
    ├── combined_kmeans.pkl
    └── combined_scaler.pkl
```

---

## ⚙️ Konfiguration

**Datei:** `config/clustering_config.yaml`

### Wichtigste Einstellungen:

```yaml
# Anzahl Cluster ändern
static_analysis:
  n_clusters: 5  # Ändere auf 3-7

dynamic_analysis:
  n_clusters: 5

combined_analysis:
  n_clusters: 6

# Gewichtung (Combined Analysis)
combined_analysis:
  weights:
    static: 0.4   # 40% aktueller Status
    dynamic: 0.6  # 60% Entwicklung/Trends

# Minimum Jahre (Dynamic)
dynamic_analysis:
  min_years_required: 5  # Mindestens 5 Jahre Daten
```

**Mehr Optionen:** Siehe `config/clustering_config.yaml` (vollständig kommentiert)

---

## 🎯 Verwendung

### Standard (alle 3 Analysen)

```bash
python main.py --market germany
```

### Nur bestimmte Analysen

```bash
# Nur statische Analyse
python main.py --market germany --only-static

# Nur dynamische Analyse
python main.py --market germany --only-dynamic
```

### Preprocessing überspringen

```bash
# Falls bereits durchgeführt (schneller!)
python main.py --market germany --skip-prep
```

### Andere Märkte

```bash
python main.py --market usa
python main.py --market europe
```

---

## 📊 Ergebnisse verstehen

### 1. Cluster-Listen (`clusters/`)

**Wichtigste Dateien!** Pro Cluster gibt es eine CSV mit allen Unternehmen:

```csv
# 0_high_performers.csv
gvkey,company_name,cluster,cluster_name,roa,roe,ebit_margin,...
001234,BMW AG,0,High Performers,8.5,15.2,10.3,...
005678,Siemens AG,0,High Performers,7.8,14.1,9.5,...
```

**Sortiert** nach Performance (beste zuerst)!

**Use Case:** Finde Peer-Unternehmen
- Dein Unternehmen in Cluster 2? → Öffne `2_*.csv`
- Alle Unternehmen in dieser Liste = Peers

### 2. Summary Report (`reports/summary.txt`)

Komplette Übersicht aller Analysen:
- Anzahl Cluster
- Anzahl Unternehmen pro Cluster
- Qualitätsmetriken (Silhouette Score, etc.)
- Migration-Patterns

### 3. Migration Matrix (`analysis/migration_matrix.csv`)

Zeigt wie Unternehmen zwischen Analysen wechseln:

| gvkey | company_name | static_cluster | dynamic_cluster | combined_cluster | pattern | flag |
|-------|--------------|----------------|-----------------|------------------|---------|------|
| 001234 | BMW AG | 0 | 0 | 0 | Consistent | ✓ |
| 005678 | Company X | 0 | 3 | 2 | Declining | ⚠ |

**Flags:**
- ✓ **Consistent** - Stabil gut
- ⭐ **Improving** - Rising Star (wird besser!)
- ⚠ **Declining** - Warnung (Verschlechterung)
- 🔥 **Critical** - Problematisch

---

## 🔍 Häufige Aufgaben

### Anzahl Cluster ändern

Editiere `config/clustering_config.yaml`:

```yaml
static_analysis:
  n_clusters: 7  # Von 5 auf 7 ändern
```

Dann erneut ausführen:

```bash
python main.py --market germany --skip-prep
```

### Peer-Unternehmen finden

**Methode 1 (einfach):**
1. Öffne `output/germany/clusters/`
2. Suche dein Unternehmen in den CSV-Dateien
3. Alle Unternehmen in derselben Datei = Peers!

**Methode 2 (programmatisch):**

```python
import pandas as pd

# Lade Cluster-Zuordnungen
df = pd.read_csv('output/germany/data/combined_assignments.csv')

# Finde Cluster eines Unternehmens
company = "BMW"
my_cluster = df[df['company_name'].str.contains(company)]['cluster'].iloc[0]

# Alle Unternehmen im gleichen Cluster
peers = df[df['cluster'] == my_cluster]
print(peers[['company_name', 'roa', 'roe']].head(20))
```

### Gewichtung anpassen

Mehr Gewicht auf Entwicklung statt aktuellen Status?

```yaml
combined_analysis:
  weights:
    static: 0.3   # Nur 30% aktuell
    dynamic: 0.7  # 70% Entwicklung
```

---

## 📈 Analyse-Tools (erweitert)

Für tiefere Analysen siehe `example_analysis.py`:

```bash
python example_analysis.py
```

Zeigt:
- Cluster-Vergleiche
- Ähnliche Unternehmen finden
- What-If Analysen
- Feature-Wichtigkeit

---

## 🔧 Troubleshooting

### Problem: "Zu wenig Daten für X Cluster"

**Lösung:** Reduziere `n_clusters` in Config

```yaml
static_analysis:
  n_clusters: 3  # Statt 5
```

### Problem: "Pipeline bricht ab bei Dynamic Analysis"

**Ursache:** Zu wenig Unternehmen mit ≥5 Jahren Daten

**Lösung:** Reduziere Minimum Jahre

```yaml
dynamic_analysis:
  min_years_required: 3  # Statt 5
```

### Problem: "Cluster ergeben keinen Sinn"

**Lösung 1:** Andere Features verwenden

```yaml
static_analysis:
  features:
    - roa
    - roe
    # Entferne korrelierte Features
```

**Lösung 2:** Andere Cluster-Anzahl testen (3, 4, 5, 6, 7)

### Problem: "Schlechter Silhouette Score"

**Normal!** Finanz-Daten haben oft Scores um 0.3-0.5.

**Was tun:**
- Prüfe trotzdem ob Cluster Sinn ergeben (manuelle Validierung)
- Teste andere Cluster-Anzahl
- Entferne unwichtige Features

---

## 📚 Projektstruktur

```
masterarbeit-kennzahlenanalyse/
├── config/
│   └── clustering_config.yaml       # ← ZENTRALE KONFIGURATION
│
├── data/
│   ├── raw/                          # Original CSV-Dateien hier ablegen
│   └── processed/{market}/           # Aufbereitete Daten
│
├── src/
│   ├── data_loader.py                # Daten laden & bereinigen
│   ├── feature_engineering.py        # Kennzahlen berechnen
│   ├── clustering_engine.py          # ← HAUPT-ENGINE (alle 3 Analysen)
│   ├── output_handler.py             # Output-Verwaltung
│   ├── config_loader.py              # Config-Management
│   ├── analysis_tools.py             # Interaktive Analysen
│   └── visualizer.py                 # Plots
│
├── output/{market}/                  # Ergebnisse (siehe oben)
│
├── main.py                           # ← HAUPTPROGRAMM
├── example_analysis.py               # Beispiele für erweiterte Analysen
├── requirements.txt                  # Dependencies
└── README.md                         # Diese Datei
```

---

## 🎓 Für die Masterarbeit

### Verwendung in Thesis

1. **Cluster-Listen für Tabellen**
   - `clusters/*.csv` → Direkt in LaTeX/Word einfügen
   - Bereits sortiert nach Performance

2. **Summary Report für Beschreibungen**
   - `reports/summary.txt` → Statistiken für Thesis

3. **Migration Matrix für Diskussion**
   - `analysis/migration_matrix.csv` → Interessant für "Entwicklung im Zeitverlauf"

4. **Plots für Abbildungen**
   - `plots/*.png` → Direkt in Thesis einfügen
   - Hochauflösend (300 DPI)

### Reproduzierbarkeit

**Wichtig für wissenschaftliche Arbeit:**

1. Config-Datei in Thesis dokumentieren
2. Verwendete Parameter klar angeben
3. Silhouette Scores mit angeben (Qualitätsmetrik)
4. Random State gesetzt (42) → Reproduzierbar

---

## 💡 Best Practices

1. **Start mit Default-Config** - Erst schauen was rauskommt
2. **Manuelle Validierung** - Öffne `clusters/*.csv` und prüfe ob sinnvoll
3. **Iterativ anpassen** - Config ändern → Erneut ausführen → Vergleichen
4. **Combined Analysis bevorzugen** - Beste Ergebnisse (statisch + dynamisch)
5. **Migration beachten** - Zeigt wer sich verbessert/verschlechtert

---

## 📝 Beispiel-Workflow

```bash
# 1. Erste Analyse
python main.py --market germany

# 2. Ergebnisse prüfen
cat output/germany/reports/summary.txt
ls output/germany/clusters/

# 3. Cluster-Anzahl anpassen (falls nötig)
nano config/clustering_config.yaml
# Ändere n_clusters: 7

# 4. Erneut ausführen (schneller mit --skip-prep)
python main.py --market germany --skip-prep

# 5. Vergleichen & beste Version wählen

# 6. Für Thesis verwenden
cp output/germany/clusters/*.csv ~/thesis/tables/
cp output/germany/plots/*.png ~/thesis/figures/
```

---

## 🆘 Support

Bei Problemen:

1. **Logs prüfen:** `output/logs/pipeline_*.log`
2. **Config prüfen:** Ist YAML-Syntax korrekt?
3. **Test mit weniger Clustern:** `n_clusters: 3`
4. **Preprocessing neu:** Ohne `--skip-prep`

---

## 👨‍🎓 Autor

Erstellt für Masterarbeit an der TH Wildau

## 📜 Lizenz

MIT

---

**Happy Clustering! 🎉**
