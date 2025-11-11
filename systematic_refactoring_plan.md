# Systematischer Refactoring-Plan

## 🎯 Ziel
Klare Verantwortlichkeiten: **1 Modul = 1 Aufgabe**

---

## 📊 GEFUNDENE VIOLATIONS

### 🔴 KRITISCH: pipeline.py (7 Methoden fehlplatziert)

#### → _05_visualization/ verschieben:
```python
_create_score_visualizations()       76 Zeilen
_create_company_insights_plots()    248 Zeilen  
_create_algorithm_congruence_plots() 146 Zeilen
_create_pca_plots()                 117 Zeilen
```
**Ziel:** `src/_05_visualization/plot_engine_insights.py` (neu)

#### → _06_validation/ verschieben:
```python
_perform_validation()               101 Zeilen
_add_external_labels()               77 Zeilen
_create_algorithm_congruence_plots() 146 Zeilen (overlap mit viz)
run_with_pca_validation()            50 Zeilen
```
**Ziel:** `src/_06_validation/validation_runner.py` (neu)

#### → _04_scoring/ verschieben:
```python
_apply_scoring()                     66 Zeilen
_track_score_evolution()             63 Zeilen
```
**Ziel:** `src/_04_scoring/score_integrator.py` (neu)

**TOTAL: ~1.090 Zeilen aus pipeline.py verschieben**
**RESULT: pipeline.py: 1846 → ~750 Zeilen**

---

### 🟡 MODERAT: hierarchical_pipeline.py (4 Methoden)

#### → _04_scoring/ verschieben:
```python
_apply_scoring()
_enhance_cluster_names_with_scores()
_analyze_score_evolution()
_print_score_statistics()
```
**Ziel:** `src/_04_scoring/score_integrator.py` (erweitern)

**RESULT: hierarchical_pipeline.py: 563 → ~400 Zeilen**

---

### 🟢 OK: Andere Module

✅ **research_excel_writer.py** (2022) - spezialisierte Excel-Logik
✅ **feature_engineer.py** (908) - reine Feature-Berechnung
✅ **output_handler.py** (891) - Output-Management (2 Helper OK)
✅ **plot_engine_*.py** (900-1000) - spezialisierte Visualisierungen

---

## 📋 REFACTORING STEPS

### Step 1: Redundanzen löschen ✅
- ❌ `comparison_engine.py` (nicht verwendet)
- ❌ `summary_generator.py` (nur TODO)

**Benefit:** -500 Zeilen

---

### Step 2: Neue Module erstellen

#### A) `src/_05_visualization/plot_engine_insights.py` (NEU)
```python
"""
Company Insights & Advanced Visualizations
"""

def create_company_insights_plots(...):
    """Top/Bottom companies per cluster"""
    pass

def create_algorithm_congruence_plots(...):
    """Robustness check visualization"""
    pass

def create_pca_detailed_plots(...):
    """Detailed PCA analysis plots"""
    pass

def create_score_visualizations(...):
    """Score distribution plots"""
    pass
```

#### B) `src/_06_validation/validation_runner.py` (NEU)
```python
"""
Validation Orchestrator
"""

def perform_validation(df, features, analysis_type):
    """Run all validation steps"""
    # 1. Alternative algorithms
    # 2. External labels
    # 3. Metrics
    pass

def add_external_labels(df, label_cols):
    """Add GICS, size, etc."""
    pass

def run_pca_validation(...):
    """PCA space validation"""
    pass
```

#### C) `src/_04_scoring/score_integrator.py` (NEU)
```python
"""
Score Integration for Pipelines
"""

def apply_scoring(df, features, cluster_col, profiles, analysis_type):
    """Calculate all scores"""
    from .score_calculator import ScoreCalculator
    calculator = ScoreCalculator(...)
    return calculator.calculate_all_scores(df, ...)

def track_score_evolution(static_df, dynamic_df, combined_df):
    """Track score changes across analyses"""
    from .score_evolution import ScoreEvolution
    tracker = ScoreEvolution()
    return tracker.analyze_evolution(...)
```

---

### Step 3: Pipeline refactoren

#### pipeline.py - VORHER (1846 Zeilen)
```python
class ClusteringPipeline:
    def _run_static_analysis(...):
        # Clustering
        df_result, profiles, metrics = self.engine.perform_clustering(...)
        
        # Scoring (direkt implementiert - 66 Zeilen)
        df_result = self._apply_scoring(...)
        
        # Naming (direkt implementiert - 72 Zeilen)
        cluster_names = self._apply_cluster_naming(...)
        
        # Validation (direkt implementiert - 101 Zeilen)
        validation_results = self._perform_validation(...)
        
        # Plots (direkt implementiert - 587 Zeilen)
        self._create_company_insights_plots(...)
        self._create_algorithm_congruence_plots(...)
        self._create_pca_plots(...)
```

#### pipeline.py - NACHHER (~750 Zeilen)
```python
class ClusteringPipeline:
    def __init__(...):
        # Import specialized modules
        from src._04_scoring.score_integrator import apply_scoring, track_score_evolution
        from src._06_validation.validation_runner import perform_validation
        from src._05_visualization.plot_engine_insights import (
            create_company_insights_plots,
            create_algorithm_congruence_plots
        )
        
        self.apply_scoring = apply_scoring
        self.perform_validation = perform_validation
        self.create_insights_plots = create_company_insights_plots
        # ...
    
    def _run_static_analysis(...):
        # Clustering
        df_result, profiles, metrics = self.engine.perform_clustering(...)
        
        # DELEGIEREN statt implementieren
        df_result = self.apply_scoring(df_result, features, ...)        # 1 Zeile
        cluster_names = self.cluster_namer.generate_names(...)          # 1 Zeile
        validation = self.perform_validation(df_result, features, ...)  # 1 Zeile
        
        # Plots
        self.create_insights_plots(df_result, profiles, ...)            # 1 Zeile
```

---

### Step 4: hierarchical_pipeline.py refactoren

Gleiche Logik wie pipeline.py:
```python
# Import score_integrator statt selbst implementieren
from src._04_scoring.score_integrator import apply_scoring

def _run_static_analysis(...):
    df_result = apply_scoring(df_result, ...)  # Delegieren
```

---

## 📐 VORHER/NACHHER

### Dateigrößen:
```
pipeline.py:              1846 → ~750 Zeilen  (-60%)
hierarchical_pipeline.py:  563 → ~400 Zeilen  (-29%)

NEU:
plot_engine_insights.py:   ~600 Zeilen
validation_runner.py:      ~350 Zeilen
score_integrator.py:       ~150 Zeilen
```

### Verantwortlichkeiten:
```
VORHER:
pipeline.py: Clustering + Scoring + Validation + Visualisierung + Reporting

NACHHER:
pipeline.py:             NUR Orchestration
score_integrator.py:     NUR Scoring
validation_runner.py:    NUR Validation
plot_engine_insights.py: NUR Visualisierung
```

---

## ✅ BENEFITS

1. **Klare Aufgabenteilung** - 1 Datei = 1 Verantwortung
2. **Jupyter-ready** - Alle Module <800 Zeilen
3. **Bessere Testbarkeit** - Isolierte Funktionen
4. **Wiederverwendbarkeit** - Module unabhängig nutzbar
5. **Wartbarkeit** - Änderungen lokal begrenzt

---

## 🎯 UMSETZUNGS-REIHENFOLGE

1. ✅ **Redundanzen löschen** (5 min)
   - comparison_engine.py
   - summary_generator.py

2. ✅ **Neue Module erstellen** (30 min)
   - plot_engine_insights.py
   - validation_runner.py  
   - score_integrator.py

3. ✅ **Funktionen verschieben** (60 min)
   - Code aus pipeline.py kopieren
   - In neue Module einfügen
   - Imports anpassen

4. ✅ **Pipelines refactoren** (45 min)
   - pipeline.py: Imports + Delegation
   - hierarchical_pipeline.py: Imports + Delegation

5. ✅ **Tests** (30 min)
   - Syntax checks
   - Import validation
   - Quick smoke test

**TOTAL: ~3 Stunden**

---

## ❓ FRAGEN VOR START

1. Soll ich mit den Redundanzen anfangen? (5 min)
2. Danach die neuen Module? (Step by Step)
3. Oder alles in einem Rutsch?

