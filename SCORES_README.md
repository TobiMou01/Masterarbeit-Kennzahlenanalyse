# Clustering Score-Berechnungen

Umfassende Dokumentation aller Score-Berechnungen in der Cluster-Analyse.

## Übersicht

Das System berechnet **7 verschiedene Score-Typen** für jedes Unternehmen in allen drei Algorithmen (K-Means, Hierarchical, DBSCAN):

| Score | Wertebereich | Interpretation | Beschreibung |
|-------|--------------|----------------|--------------|
| **Proximity Score** | 0-100 | Höher = Besser | Distanz zum Cluster-Zentrum |
| **Profitability Score** | 0-100 | Höher = Besser | Rentabilitätskennzahlen |
| **Leverage Score** | 0-100 | Höher = Besser | Verschuldungskennzahlen |
| **Efficiency Score** | 0-100 | Höher = Besser | Effizienzkennzahlen |
| **Growth Score** | 0-100 | Höher = Besser | Wachstumskennzahlen |
| **Relative Score** | 0-100 | Höher = Besser | Performance vs. Cluster-Durchschnitt |
| **Overall Score** | 0-100 | Höher = Besser | Gewichtete Gesamtbewertung |

**Alle Scores sind standardisiert auf 0-100:**
- **100** = Perfekt/Optimal
- **50** = Durchschnitt
- **0** = Schlechteste Performance

---

## 1. Proximity Score (Cluster-Zugehörigkeit)

### Was misst dieser Score?
Wie **typisch** ein Unternehmen für seinen Cluster ist - basierend auf der Distanz zum Cluster-Zentrum.

### Berechnung

```
1. Features standardisieren (Z-Transformation)
2. Cluster-Zentrum berechnen (Mittelwert aller Unternehmen im Cluster)
3. Euklidische Distanz: d = ||Unternehmen - Zentrum||
4. Score-Transformation: score = 100 × exp(-d / 2)
```

### Interpretation

| Score | Bedeutung | Interpretation |
|-------|-----------|----------------|
| 90-100 | Kern-Mitglied | Sehr typisches Exemplar des Clusters |
| 70-89 | Stabiles Mitglied | Gut zum Cluster passend |
| 50-69 | Peripheres Mitglied | Am Rand des Clusters |
| 30-49 | Grenzfall | Könnte auch zu anderem Cluster gehören |
| 0-29 | Ausreißer | Sehr atypisch für den Cluster |

### Eigenschaften
- **Algorithmusabhängig**: Bei K-Means sind Cluster sphärisch, bei DBSCAN beliebig geformt
- **Feature-sensitiv**: Hängt von den verwendeten Features ab
- **Noise-Punkte** (DBSCAN): Erhalten automatisch Score = 0

---

## 2. Dimensional Scores (Kategorie-spezifisch)

Diese Scores messen die Proximity **nur für Features einer bestimmten Kategorie**.

### 2.1 Profitability Score

**Features:**
- Return on Assets (ROA)
- Return on Equity (ROE)
- EBIT Margin
- Net Profit Margin
- Operating Margin
- Free Cash Flow Margin

**Interpretation:** Wie gut passt die **Rentabilität** des Unternehmens zu seinem Cluster?

### 2.2 Leverage Score

**Features:**
- Debt-to-Equity Ratio
- Interest Coverage
- Current Ratio
- Quick Ratio
- Cash Ratio

**Interpretation:** Wie gut passt die **Verschuldungsstruktur** zum Cluster?

### 2.3 Efficiency Score

**Features:**
- Asset Turnover
- Inventory Turnover
- Receivables Turnover
- Working Capital Turnover

**Interpretation:** Wie effizient wirtschaftet das Unternehmen **im Vergleich zu seinen Cluster-Peers**?

### 2.4 Growth Score

**Features:**
- Revenue Growth (CAGR)
- Asset Growth
- Earnings Growth
- Employee Growth
- (Bei Dynamic: Trend-Features)

**Interpretation:** Wie gut passt das **Wachstumsprofil** zum Cluster?

### Verwendung
Diese Scores helfen zu verstehen, **welche Dimension** ein Unternehmen stark/schwach zum Cluster beiträgt:

**Beispiel:**
- Proximity Score: 65 (mittelmäßig)
- Profitability Score: 85 (gut)
- Growth Score: 40 (schwach)
→ **Interpretation:** Unternehmen passt rentabilitätsmäßig gut, aber Wachstum ist unterdurchschnittlich für den Cluster

---

## 3. Relative Score (Cluster-Vergleich)

### Was misst dieser Score?
Wie gut ein Unternehmen **im Vergleich zu anderen Unternehmen im selben Cluster** performt.

### Berechnung

```
1. Für jedes Feature: Z-Score innerhalb des Clusters berechnen
   z_i = (x_i - μ_cluster) / σ_cluster

2. Durchschnittlicher Z-Score über alle Features: z_avg

3. Transformation auf 0-100 Skala:
   score = z_avg × 10 + 50
```

### Interpretation

| Score | Z-Score | Bedeutung |
|-------|---------|-----------|
| 70+ | +2σ oder mehr | Top-Performer im Cluster |
| 60-70 | +1σ bis +2σ | Überdurchschnittlich |
| 40-60 | -1σ bis +1σ | Durchschnitt |
| 30-40 | -2σ bis -1σ | Unterdurchschnittlich |
| <30 | -2σ oder weniger | Schwache Performance |

### Unterschied zu Proximity Score

| Metric | Was wird gemessen? | Beispiel |
|--------|-------------------|----------|
| **Proximity** | Wie typisch für Cluster | Unternehmen nah am Durchschnitt → Hoher Score |
| **Relative** | Wie gut im Cluster | Unternehmen besser als Durchschnitt → Hoher Score |

**Wichtig:** Ein Unternehmen kann gleichzeitig:
- **Hohen Proximity Score** haben (typisch für Cluster)
- **Niedrigen Relative Score** haben (schwächstes im Cluster)

---

## 4. Overall Score (Gesamtbewertung)

### Was misst dieser Score?
Gewichtete Kombination aller anderen Scores zu einer **Gesamtbewertung**.

### Berechnung

```
Overall Score = Σ (Score_i × Gewicht_i)

Standardgewichte:
- Proximity Score:      40%
- Profitability Score:  20%
- Leverage Score:       10%
- Efficiency Score:     15%
- Growth Score:         15%
```

### Interpretation

| Score | Bewertung | Business-Bedeutung |
|-------|-----------|-------------------|
| 80-100 | Champions | Exzellente Unternehmen, Best-in-Class |
| 65-79 | Strong | Solide, überdurchschnittliche Performance |
| 50-64 | Solid | Durchschnittlich für den Cluster |
| 35-49 | Challenged | Unterdurchschnittlich, Handlungsbedarf |
| 0-34 | Weak | Kritische Situation, hoher Handlungsdruck |

### Verwendung
- **Investment-Entscheidungen**: Unternehmen mit hohem Overall Score bevorzugen
- **Portfolio-Management**: Diversifikation über verschiedene Score-Bereiche
- **Risk-Assessment**: Niedrige Scores = höheres Risiko

---

## Anwendungsbeispiele

### Beispiel 1: Typischer Fall

**Unternehmen A:**
- Proximity Score: 75 → Gut ins Cluster passend
- Profitability Score: 80 → Hohe Rentabilität für den Cluster
- Growth Score: 70 → Gutes Wachstum
- Relative Score: 65 → Überdurchschnittlich im Cluster
- **Overall Score: 72** → **Strong**

**Interpretation:** Solides, überdurchschnittliches Unternehmen, gute Investment-Kandidat.

---

### Beispiel 2: Ausreißer im Cluster

**Unternehmen B:**
- Proximity Score: 45 → Peripheres Cluster-Mitglied
- Profitability Score: 90 → Sehr hohe Rentabilität
- Growth Score: 40 → Schwaches Wachstum
- Relative Score: 75 → Deutlich besser als Cluster-Durchschnitt
- **Overall Score: 62** → **Solid**

**Interpretation:** Unternehmen performt besser als sein Cluster, könnte bei Rebalancing in anderen Cluster wandern.

---

### Beispiel 3: Cluster-Kern, aber schwach

**Unternehmen C:**
- Proximity Score: 85 → Sehr typisch für Cluster
- Profitability Score: 30 → Schwache Rentabilität
- Growth Score: 35 → Schwaches Wachstum
- Relative Score: 45 → Leicht unterdurchschnittlich
- **Overall Score: 52** → **Solid** (aber grenzwertig)

**Interpretation:** Typisches Mitglied eines schwachen Clusters. Cluster selbst möglicherweise problematisch.

---

## Algorithmische Unterschiede

Die Scores werden **für alle drei Algorithmen** berechnet, aber haben unterschiedliche Eigenschaften:

### K-Means
- **Proximity**: Basiert auf Zentroiden (explizit berechnet)
- **Cluster-Form**: Sphärisch
- **Typisch**: Gleichmäßigere Score-Verteilung

### Hierarchical Clustering
- **Proximity**: Basiert auf Cluster-Mittelwert (Zentroide werden berechnet)
- **Cluster-Form**: Hierarchisch verschachtelt
- **Typisch**: Variablere Score-Verteilungen

### DBSCAN
- **Proximity**: Basiert auf Cluster-Mittelwert
- **Cluster-Form**: Beliebig (dichtebasiert)
- **Besonderheit**: Noise-Punkte (Cluster = -1) erhalten Score = 0
- **Typisch**: Höhere Varianz, da Ausreißer identifiziert werden

### Vergleichbarkeit

| Aspekt | Vergleichbar? | Anmerkung |
|--------|---------------|-----------|
| **Absolute Werte** | ⚠️ Bedingt | Scores hängen von Cluster-Definition ab |
| **Relative Rankings** | ✅ Ja | Rankings innerhalb Algorithmus sind aussagekräftig |
| **Konsistenz-Analyse** | ✅ Sinnvoll | Unternehmen mit hohen Scores über alle Algorithmen = robuste Bewertung |

**Empfehlung:** Nutze Scores **innerhalb eines Algorithmus** für Ranking, aber **über Algorithmen hinweg** für Konsistenz-Checks.

---

## Analyse-Phasen

Scores werden in **3 Analyse-Phasen** berechnet:

### 1. Static Analysis
**Features:** Aktuelle Kennzahlen (ROA, ROE, Debt-to-Equity, ...)
**Interpretation:** Wie gut ist das Unternehmen **aktuell** positioniert?

### 2. Dynamic Analysis
**Features:** Trend-Kennzahlen (ROA_trend, Revenue_CAGR, Volatility, ...)
**Interpretation:** Wie entwickelt sich das Unternehmen **über Zeit**?

### 3. Combined Analysis
**Features:** Static + Dynamic zusammen
**Interpretation:** Gesamtbild aus **aktueller Situation** und **Entwicklung**

### Score Evolution Tracking
Das System trackt, wie sich Scores von Static → Dynamic → Combined entwickeln:

**Patterns:**
- **Consistent Excellence**: Hoch in Static UND Dynamic
- **Eroding Position**: Hoch Static, niedrig Dynamic (Verschlechterung)
- **Improving Trend**: Niedrig Static, hoch Dynamic (Verbesserung)
- **Challenged**: Generell niedrige Scores
- **Mixed**: Unterschiedliche Scores je nach Phase

---

## Verwendung in Excel-Reports

Die Scores werden in den **Company Cluster Analysis Excel-Dateien** als eigene Spalten ausgegeben:

### Sheet-Struktur
1. **Overview**: Alle Unternehmen mit allen Scores
2. **Cluster_0, Cluster_1, ...**: Separate Sheets pro Cluster, sortiert nach Overall Score
3. **Score_Evolution** (nur bei Combined): Vergleich Static → Dynamic → Combined
4. **Summary**: Aggregierte Statistiken

### Spalten in Excel

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| gvkey | ID | Unternehmens-ID |
| company_name | Text | Unternehmensname |
| cluster | Integer | Cluster-Zuordnung (0, 1, 2, ...) |
| proximity_score | 0-100 | Proximity Score |
| profitability_score | 0-100 | Profitability Score |
| leverage_score | 0-100 | Leverage Score |
| efficiency_score | 0-100 | Efficiency Score |
| growth_score | 0-100 | Growth Score |
| relative_score | 0-100 | Relative Score |
| overall_score | 0-100 | Overall Score |
| cluster_rank | Integer | Rang innerhalb des Clusters (1 = Bester) |

---

## Technische Details

### Score-Normalisierung
Alle Scores werden auf [0, 100] geclippt:
```python
score = np.clip(score, 0, 100)
```

### Umgang mit Missing Values
- Features mit Missing Values werden mit Median imputiert
- Bei zu vielen Missing Values wird Feature übersprungen

### Feature-Standardisierung
Vor Distanzberechnung werden alle Features standardisiert (Z-Transformation):
```python
X_scaled = StandardScaler().fit_transform(X)
```

---

## FAQ

### F: Warum haben manche Unternehmen einen Score von 0?
**A:** Typischerweise Noise-Punkte bei DBSCAN (Cluster = -1) oder extreme Ausreißer.

### F: Kann ein Unternehmen hohen Proximity aber niedrigen Relative Score haben?
**A:** Ja! Proximity misst "Typikalität", Relative misst "Performance im Cluster".

### F: Welcher Score ist am wichtigsten?
**A:** **Overall Score** für Gesamtbewertung, aber **Dimensional Scores** für detaillierte Analyse.

### F: Sind Scores über Algorithmen vergleichbar?
**A:** Absolute Werte NEIN, aber **Konsistenz** (hoher Score über alle Algorithmen) IST aussagekräftig.

### F: Was bedeutet ein Overall Score von 50?
**A:** Genau durchschnittlich für den Cluster - weder besonders gut noch schlecht.

### F: Wie oft sollten Scores neu berechnet werden?
**A:** Bei jeder neuen Datengrundlage oder wenn sich Cluster-Zuordnungen ändern.

---

## Weiterführende Ressourcen

- `src/_04_scoring/score_calculator.py` - Implementierung der Score-Berechnungen
- `src/_04_scoring/score_evolution.py` - Score-Evolution Tracking
- `src/_04_scoring/score_analyzer.py` - Statistische Score-Analysen
- `src/_04_comparison/company_analysis.py` - Excel-Export mit Scores

---

## Zusammenfassung

**Scores bieten:**
- ✅ Quantifizierung der Cluster-Qualität auf Unternehmensebene
- ✅ Vergleichbarkeit innerhalb von Clustern
- ✅ Mehrdimensionale Bewertung (Proximity, Dimensional, Relative, Overall)
- ✅ Zeitliche Entwicklung (Static → Dynamic → Combined)
- ✅ Entscheidungsunterstützung für Investment und Portfolio-Management

**Best Practices:**
1. Nutze **Overall Score** für erste Bewertung
2. Analysiere **Dimensional Scores** für Details
3. Vergleiche **über Algorithmen** für Robustheit
4. Tracke **Score Evolution** für Trend-Erkennung
5. Beachte **Cluster-Kontext** (schwacher Cluster = generell niedrigere Scores)

---

**Version:** 1.0
**Stand:** November 2025
**Autor:** Masterarbeit Kennzahlenanalyse
