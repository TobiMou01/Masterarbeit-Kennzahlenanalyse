# Konsolidierte Excel-Datei: Algorithm Comparison

Die konsolidierte Excel-Datei bietet einen **umfassenden Vergleich aller Clustering-Algorithmen** in einer einzigen, übersichtlichen Datei.

## Dateiname

```
output/germany/03_comparisons/algorithm_comparison_combined.xlsx
```

Diese Datei wird automatisch erstellt, wenn Sie die Comparison Pipeline mit mehreren Algorithmen laufen lassen.

---

## Sheet-Struktur

Die Excel-Datei enthält **5 Sheets**:

### 1. Overview (Haupttabelle)

**Beschreibung:** Zentrale Vergleichstabelle mit allen Unternehmen und allen Algorithmen nebeneinander.

**Spalten-Struktur:**

```
├── Basisinformationen
│   ├── gvkey (Unternehmens-ID)
│   ├── company_name
│   ├── gsector, ggroup, gind, gsubind (GICS Klassifikation)
│   └── revt, at (Größen-Metriken)
│
├── K-Means Ergebnisse
│   ├── kmeans_cluster
│   ├── kmeans_cluster_name
│   ├── kmeans_proximity_score
│   ├── kmeans_profitability_score
│   ├── kmeans_leverage_score
│   ├── kmeans_efficiency_score
│   ├── kmeans_growth_score
│   ├── kmeans_relative_score
│   └── kmeans_overall_score
│
├── Hierarchical Ergebnisse
│   ├── hierarchical_cluster
│   ├── hierarchical_cluster_name
│   ├── hierarchical_proximity_score
│   ├── hierarchical_profitability_score
│   ├── hierarchical_leverage_score
│   ├── hierarchical_efficiency_score
│   ├── hierarchical_growth_score
│   ├── hierarchical_relative_score
│   └── hierarchical_overall_score
│
├── DBSCAN Ergebnisse
│   ├── dbscan_cluster
│   ├── dbscan_cluster_name
│   ├── dbscan_proximity_score
│   ├── dbscan_profitability_score
│   ├── dbscan_leverage_score
│   ├── dbscan_efficiency_score
│   ├── dbscan_growth_score
│   ├── dbscan_relative_score
│   └── dbscan_overall_score
│
└── Consensus-Metriken
    ├── consensus_overall_score (Durchschnitt über alle Algorithmen)
    ├── score_std_dev (Standardabweichung der Scores)
    ├── algorithms_with_data (Anzahl Algorithmen mit Daten)
    ├── unique_cluster_assignments (Anzahl unterschiedlicher Cluster)
    └── cluster_agreement_score (1.0 = perfekte Übereinstimmung)
```

**Features:**
- ✅ **Excel-Filter** auf allen Spalten aktiviert
- ✅ **Conditional Formatting** auf allen Score-Spalten (Rot → Gelb → Grün)
- ✅ Freeze Panes bei Spalte C (gvkey, company_name sichtbar beim Scrollen)
- ✅ Sortierbar und filterbar

**Verwendung:**
```
Beispiel-Filter-Kombinationen:

1. Top-Performer über alle Algorithmen:
   - consensus_overall_score >= 80

2. Unternehmen mit hoher Algorithmen-Übereinstimmung:
   - cluster_agreement_score >= 0.9

3. Algorithmus-sensitive Unternehmen:
   - score_std_dev > 15 (große Varianz zwischen Algorithmen)

4. Sektor-spezifische Analyse:
   - gsector = "Financials"
   - Dann Scores vergleichen
```

---

### 2. Summary (Zusammenfassung)

**Beschreibung:** Aggregierte Statistiken über alle Algorithmen.

**Inhalt:**

| Category | Metric | Value |
|----------|--------|-------|
| General | Total Companies | 150 |
| Kmeans | Number of Clusters | 5 |
| Kmeans | Companies Clustered | 150 |
| Kmeans | Average Overall Score | 62.45 |
| Hierarchical | Number of Clusters | 5 |
| Hierarchical | Average Overall Score | 58.92 |
| DBSCAN | Number of Clusters | 7 |
| DBSCAN | Companies Clustered | 142 (Noise = 8) |
| DBSCAN | Average Overall Score | 55.31 |
| Consensus | Average Consensus Score | 60.23 |
| Consensus | Average Cluster Agreement | 0.76 |
| Consensus | High Agreement Companies | 98 |

**Verwendung:**
- Schneller Überblick über Algorithmen-Performance
- Identifikation von Algorithmen mit ähnlichen/unterschiedlichen Ergebnissen
- Vergleich der Cluster-Anzahl

---

### 3. Cluster_Analysis (Cluster-Details)

**Beschreibung:** Detaillierte Statistiken pro Cluster und Algorithmus.

**Spalten:**
- `Algorithm` - Algorithmus-Name
- `Cluster_ID` - Cluster-Nummer
- `Company_Count` - Anzahl Unternehmen im Cluster
- `Avg_Overall_Score` - Durchschnittlicher Overall Score
- `Min_Score` - Minimaler Score
- `Max_Score` - Maximaler Score
- `Std_Score` - Standardabweichung der Scores

**Beispiel:**

| Algorithm | Cluster_ID | Company_Count | Avg_Overall_Score | Min_Score | Max_Score | Std_Score |
|-----------|------------|---------------|-------------------|-----------|-----------|-----------|
| kmeans | 0 | 35 | 75.2 | 55.1 | 92.3 | 8.4 |
| kmeans | 1 | 28 | 45.8 | 25.3 | 68.9 | 12.1 |
| hierarchical | 0 | 32 | 72.5 | 52.8 | 88.7 | 9.2 |

**Verwendung:**
- Vergleich von Cluster-Qualität zwischen Algorithmen
- Identifikation von heterogenen Clustern (hohe Std_Score)
- Cluster-Größen-Vergleich

---

### 4. Score_Distributions (Boxplot-Statistiken)

**Beschreibung:** Statistische Verteilungen der Scores für Boxplot-Erstellung.

**Inhalt:**
- Statistiken für **Overall Score**, **Proximity Score**, **Profitability Score**
- Pro Algorithmus: Mean, Median, Std Dev, Min, Max

**Verwendung:**
```
1. Daten aus diesem Sheet kopieren
2. In Excel: Insert → Charts → Box and Whisker Plot
3. Vergleich der Score-Verteilungen zwischen Algorithmen visuell darstellen
```

**Interpretation:**
- **Mean > Median**: Rechtsschief (viele niedrige Scores, wenige hohe)
- **Mean < Median**: Linksschief (viele hohe Scores, wenige niedrige)
- **Hohe Std Dev**: Große Streuung innerhalb des Algorithmus
- **Min/Max**: Outlier-Identifikation

---

### 5. Score_Comparisons (Scatter-Plot-Daten)

**Beschreibung:** Paarweise Score-Vergleiche zwischen Algorithmen für Scatter-Plots.

**Vergleiche:**
1. **K-Means vs Hierarchical** (Overall Score)
2. **K-Means vs DBSCAN** (Overall Score)
3. **Hierarchical vs DBSCAN** (Overall Score)

**Verwendung:**
```
1. Daten aus diesem Sheet kopieren
2. In Excel: Insert → Charts → Scatter Plot
3. X-Achse: Algorithmus 1 Score
4. Y-Achse: Algorithmus 2 Score
5. Diagonale hinzufügen (y=x) für perfekte Übereinstimmung
```

**Interpretation:**

```
Scatter Plot: K-Means (X) vs Hierarchical (Y)

  100│           •
     │         •   •
     │       •   •   •
  Y  │     •   •       •
     │   •   •
  50 │ •   •
     │ •
     │•
   0 └─────────────────────
     0        50         100
              X

- Punkte nahe der Diagonalen: Hohe Übereinstimmung
- Punkte über Diagonale: Hierarchical bewertet höher
- Punkte unter Diagonale: K-Means bewertet höher
- Weit von Diagonale: Algorithmen uneinig über Unternehmen
```

---

## Conditional Formatting

Alle Score-Spalten haben **automatische Farbcodierung**:

### Farbskala (3-Color Scale)

| Score-Bereich | Farbe | Bedeutung |
|---------------|-------|-----------|
| 0-40 | 🔴 Rot | Schwache Performance |
| 40-70 | 🟡 Gelb | Durchschnittliche Performance |
| 70-100 | 🟢 Grün | Starke Performance |

**Gradient:** Fließender Übergang zwischen den Farben für intuitive Visualisierung.

---

## Anwendungsfälle

### 1. Top-Performer identifizieren

**Ziel:** Unternehmen finden, die über alle Algorithmen hinweg gut abschneiden.

**Vorgehen:**
1. Sheet: **Overview**
2. Filter: `consensus_overall_score >= 75`
3. Sortieren: `consensus_overall_score` absteigend
4. Zusätzlich: `cluster_agreement_score >= 0.8` (konsistent geclustert)

**Interpretation:** Diese Unternehmen sind robuste Top-Performer unabhängig vom Algorithmus.

---

### 2. Algorithmus-sensitive Unternehmen

**Ziel:** Unternehmen finden, bei denen Algorithmen stark unterschiedliche Bewertungen haben.

**Vorgehen:**
1. Sheet: **Overview**
2. Filter: `score_std_dev > 20`
3. Manuell prüfen: `kmeans_overall_score` vs `hierarchical_overall_score` vs `dbscan_overall_score`

**Interpretation:** Diese Unternehmen haben besondere Eigenschaften, die Algorithmen unterschiedlich bewerten. Weitere Analyse empfohlen!

**Beispiel:**
```
Company X:
- K-Means Overall Score: 85 (Top-Performer in sphärischen Clustern)
- Hierarchical Overall Score: 82 (Gut in hierarchischen Strukturen)
- DBSCAN Overall Score: 45 (Randpunkt in dichtebasierten Clustern)

→ Interpretation: Company X ist mainstream-typisch, aber nicht Teil
  einer dichten Gruppe → Möglicherweise Nischen-Player.
```

---

### 3. Cluster-Konsistenz analysieren

**Ziel:** Herausfinden, welche Unternehmen von allen Algorithmen gleich geclustert werden.

**Vorgehen:**
1. Sheet: **Overview**
2. Filtern: `unique_cluster_assignments = 1`
3. Zusätzlich: `cluster_agreement_score = 1.0`

**Interpretation:** Diese Unternehmen sind eindeutig einem Muster zuordenbar - unabhängig von der Clustering-Methode.

---

### 4. Sektor-spezifische Score-Vergleiche

**Ziel:** Scores innerhalb eines Sektors vergleichen.

**Vorgehen:**
1. Sheet: **Overview**
2. Filter: `gsector = "Information Technology"`
3. Sortieren: `consensus_overall_score` absteigend
4. Vergleichen: Dimensional Scores (Profitability, Growth, etc.)

**Verwendung:**
- Best-in-Class Identifikation pro Sektor
- Sektor-spezifische Benchmarks erstellen
- Relative Performance innerhalb des Sektors

---

### 5. Outlier-Detection über Algorithmen

**Ziel:** Unternehmen finden, die bei allen Algorithmen niedrige Scores haben.

**Vorgehen:**
1. Sheet: **Overview**
2. Filter kombinieren:
   - `kmeans_overall_score < 40`
   - `hierarchical_overall_score < 40`
   - `dbscan_cluster = -1` (Noise bei DBSCAN)

**Interpretation:** Diese Unternehmen sind echte Outliers über alle Methoden hinweg → Risiko-Kandidaten oder Sonderfälle.

---

## Excel-Tipps für erweiterte Analyse

### 1. Pivot-Tabellen erstellen

```
1. Overview Sheet komplett markieren
2. Insert → PivotTable
3. Rows: gsector
4. Values: Average of consensus_overall_score
5. Filter: cluster_agreement_score >= 0.8

→ Ergebnis: Durchschnittliche Scores pro Sektor (nur konsistente Unternehmen)
```

### 2. Eigene Conditional Formatting Regeln

```
Beispiel: Algorithmen-Disagreement hervorheben

1. Spalte auswählen: score_std_dev
2. Home → Conditional Formatting → New Rule
3. Format cells that contain: Cell Value > 15
4. Format: Gelbe Füllung

→ Alle Unternehmen mit hoher Score-Varianz werden gelb markiert
```

### 3. Slicers für interaktive Filterung

```
1. Overview Sheet auswählen
2. Insert → Slicer
3. Slicers hinzufügen für:
   - gsector
   - kmeans_cluster
   - Consensus Score Ranges (custom)

→ Interaktives Dashboard zur Datenexploration
```

### 4. Charts direkt in Excel erstellen

**Boxplot-Erstellung:**
```
1. Score_Distributions Sheet öffnen
2. Daten markieren (z.B. Overall Score Statistics)
3. Insert → Chart → Box and Whisker
4. Titel: "Overall Score Distribution by Algorithm"
```

**Scatter Plot:**
```
1. Score_Comparisons Sheet öffnen
2. Zwei Spalten markieren (z.B. K-Means vs Hierarchical)
3. Insert → Chart → Scatter
4. Diagonale manuell hinzufügen:
   - Add Trendline → Linear → Display Equation
```

---

## Vergleich: Konsolidierte vs. Einzelne Excel-Dateien

| Aspekt | Konsolidierte Datei | Einzelne Dateien |
|--------|---------------------|------------------|
| **Vergleichbarkeit** | ✅ Exzellent (alle Daten nebeneinander) | ⚠️ Mühsam (VLOOKUP/manuell) |
| **Übersicht** | ✅ Zentrale Anlaufstelle | ⚠️ Mehrere Dateien öffnen |
| **Filter/Pivot** | ✅ Über alle Algorithmen | ❌ Pro Algorithmus separat |
| **Consensus Metrics** | ✅ Vorhanden | ❌ Nicht verfügbar |
| **Detail-Tiefe** | ⚠️ Kompakt | ✅ Sehr detailliert |
| **Dateigröße** | ⚠️ Größer | ✅ Kleiner |

**Empfehlung:**
- **Konsolidierte Datei** für: Vergleiche, High-level Analyse, Dashboards
- **Einzelne Dateien** für: Detaillierte Cluster-Analysen pro Algorithmus

---

## Technische Details

### Erstellung

Die Datei wird automatisch durch die **ComparisonPipeline** erstellt:

```python
# In src/_04_comparison/comparison_pipeline.py
consolidated_excel_path = create_consolidated_comparison_excel(
    algorithm_results=self.algorithm_results,
    market=self.market,
    output_dir=str(self.base_dir)
)
```

### Datenquellen

Für jeden Algorithmus wird das **Combined-Stage** verwendet (wenn verfügbar), sonst **Static-Stage**:

- **K-Means**: `combined` (Static + Dynamic Features)
- **Hierarchical**: `combined` (wenn vorhanden) oder `static`
- **DBSCAN**: `combined` (wenn vorhanden) oder `static`

### Conditional Formatting (openpyxl)

Implementiert als 3-Color Scale:
```python
ColorScaleRule(
    start_type='num', start_value=0, start_color='F8696B',    # Rot
    mid_type='num', mid_value=50, mid_color='FFEB84',         # Gelb
    end_type='num', end_value=100, end_color='63BE7B'         # Grün
)
```

---

## Troubleshooting

### Problem: Leere Spalten für einen Algorithmus

**Ursache:** Algorithmus wurde nicht ausgeführt oder hat keine Combined/Static Daten.

**Lösung:**
1. Prüfen Sie `output/germany/02_algorithms/<algorithmus>/`
2. Stellen Sie sicher, dass Combined oder Static erfolgreich durchlief
3. Re-run der ComparisonPipeline

---

### Problem: Consensus-Metriken sind NaN

**Ursache:** Nur ein Algorithmus hat Daten.

**Lösung:** Mehrere Algorithmen laufen lassen für sinnvolle Consensus-Metriken.

---

### Problem: Conditional Formatting fehlt

**Ursache:** Excel-Version unterstützt 3-Color Scale nicht.

**Lösung:** Manuell Conditional Formatting in Excel hinzufügen (Home → Conditional Formatting → Color Scales).

---

## Zusammenfassung

Die **konsolidierte Excel-Datei** bietet:

✅ **Einen zentralen Vergleichspunkt** für alle Algorithmen
✅ **Consensus-Metriken** für robuste Bewertungen
✅ **Interaktive Filterung** für flexible Analysen
✅ **Visuelle Aufbereitung** durch Conditional Formatting
✅ **Statistische Grundlagen** für Boxplots und Scatter Plots
✅ **Detaillierte Cluster-Analysen** pro Algorithmus

**Empfohlener Workflow:**
1. Öffnen Sie `algorithm_comparison_combined.xlsx`
2. Starten Sie im **Overview Sheet** mit Filtern
3. Nutzen Sie **Summary** für Gesamtüberblick
4. **Cluster_Analysis** für Details pro Cluster
5. **Score_Distributions** und **Score_Comparisons** für visuelle Analysen

---

**Version:** 1.0
**Stand:** November 2025
**Autor:** Masterarbeit Kennzahlenanalyse
