# Jupyter Notebooks - Masterarbeit Kennzahlenanalyse

Dieses Verzeichnis enthält Jupyter Notebooks für interaktive Analysen und On-the-fly Config-Anpassungen.

## 📓 Verfügbare Notebooks

### 01_quick_start.ipynb
- Schneller Einstieg in die Analyse
- Laden von Checkpoints
- Basis-Visualisierungen

### 02_config_tuning.ipynb
- Interaktive Config-Anpassungen
- Parameter-Tuning für Clustering
- Vergleich verschiedener Konfigurationen

### 03_cluster_exploration.ipynb
- Deep-Dive in einzelne Cluster
- Company-Level Analysen
- Feature-Importance Visualisierungen

### 04_algorithm_comparison.ipynb
- Vergleich K-Means vs. Hierarchical vs. DBSCAN
- Robustheitstests
- Kongruenz-Analysen

### 99_full_pipeline.ipynb
- Komplette Pipeline von Preprocessing bis Excel-Export
- Mit Checkpoints für schnelles Resume
- Ideal für finale Thesis-Analysen

## 🚀 Schnellstart

```python
# In Jupyter Notebook
import sys
sys.path.append('..')

from src._01_setup import config_loader, checkpoint_manager
from src._02_preprocessing import data_cleaner
from src._03_clustering.pipeline import ClusteringPipeline

# 1. Load Config
config = config_loader.load()

# 2. Quick Adjustments (optional)
config['clustering']['kmeans']['n_clusters'] = 7

# 3. Load Checkpoint (if available)
manager = checkpoint_manager.CheckpointManager(market='germany')
results = manager.load_checkpoint('kmeans_complete')

# 4. Or run fresh
pipeline = ClusteringPipeline(config)
results = pipeline.run()

# 5. Save checkpoint for later
manager.save_checkpoint(results, 'kmeans_complete', metadata={'n_clusters': 7})
```

## 🎯 Workflow

### Für iterative Analysen:
1. Run pipeline einmal → speichere Checkpoint
2. Im Notebook: Lade Checkpoint → analysiere
3. Ändere Config-Parameter → re-run nur betroffene Teile
4. Speichere neue Checkpoints mit beschreibenden Namen

### Für Thesis-Finalisierung:
1. Use `99_full_pipeline.ipynb`
2. Führe komplette Analyse durch
3. Exportiere alle Excel-Reports
4. Speichere finale Config in `output/germany/00_config/`

## 📂 Datenzugriff

### Von Notebooks aus:
```python
import pandas as pd

# Consolidated Data
df = pd.read_csv('../output/germany/01_data/assignments.csv')

# From Algorithm Results
df_kmeans = pd.read_csv('../output/germany/02_algorithms/kmeans_comparative/combined/data/assignments.csv')

# From Checkpoints
from src._01_setup.checkpoint_manager import load_checkpoint
results = load_checkpoint('preprocessing_complete', market='germany')
```

## 🔧 Dependencies

Alle Dependencies aus `requirements.txt` werden benötigt.
Virtual Environment aktivieren:
```bash
source venv_masterarbeit/bin/activate
```

## 💡 Tips

1. **Config-Tuning**: Ändere immer nur einen Parameter, speichere Checkpoint
2. **Checkpoints**: Nutze beschreibende Namen (z.B. `kmeans_k7`, `hierarchical_complete`)
3. **Plots**: Inline-Plots sind schneller, speichere nur finale Versionen
4. **Excel-Export**: Führe nur am Ende aus (langsam!)

## 🐛 Troubleshooting

**Import Error**:
```python
import sys
sys.path.append('..')  # Füge Projekt-Root zum Path hinzu
```

**Checkpoint not found**:
```python
manager.list_checkpoints()  # Zeige verfügbare Checkpoints
```

**Config-Fehler**:
```python
config = config_loader.load()  # Lädt default config.yaml
config_loader.validate(config)  # Validiere Config
```
