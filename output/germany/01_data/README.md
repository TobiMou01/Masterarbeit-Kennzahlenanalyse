# 01_data/ - Consolidated Analysis Data

This directory contains consolidated CSVs for quick analysis and Jupyter notebook workflows.

## 📊 Files

### Cluster Assignments
- `assignments.csv` - Company-level cluster assignments from K-Means Combined analysis
  - Columns: gvkey, company_name, cluster, cluster_name, scores, features
  - Use this for: Company lookups, cluster membership analysis

### Cluster Profiles
- `profiles.csv` - Cluster-level statistics and profiles
  - Columns: cluster_id, cluster_name, size, feature_means, feature_stds
  - Use this for: Cluster characterization, comparison

### Checkpoints
- `checkpoints/` - Intermediate pipeline results for Jupyter notebooks
  - Format: Pickle files with metadata JSON
  - Use this for: Resuming analysis, iterative development

## 🔄 Reproducibility

These files are **generated** from the main pipeline and should be treated as:
- ✅ Source of truth for consolidated data
- ✅ Safe to use in Jupyter notebooks
- ⚠️ Will be overwritten on next pipeline run

To regenerate:
```python
python src/main.py
```

## 📓 Jupyter Notebook Usage

```python
import pandas as pd

# Load assignments
df_assignments = pd.read_csv('output/germany/01_data/assignments.csv')

# Load profiles
df_profiles = pd.read_csv('output/germany/01_data/profiles.csv')

# Load checkpoint (if available)
from src._01_setup.checkpoint_manager import load_checkpoint
results = load_checkpoint('kmeans_complete', market='germany')
```

## 🔗 Related

- Full algorithm results: `../02_algorithms/`
- Excel reports: `../04_excel_reports/`
- Config snapshot: `../00_config/analysis_config.yaml`
