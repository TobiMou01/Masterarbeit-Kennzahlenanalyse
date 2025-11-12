# Interactive Jupyter Notebooks - Clustering Analysis

## 📖 Overview

This directory contains **interactive Jupyter notebooks** for step-by-step clustering analysis. Unlike running the entire pipeline at once via `main.py`, these notebooks allow you to:

- ✅ **Execute analyses step-by-step** with immediate visual feedback
- ✅ **Adjust parameters on-the-fly** and see results instantly
- ✅ **Visualize data interactively** at each stage
- ✅ **Save intermediate states** for resuming work later
- ✅ **Compare different approaches** side-by-side

---

## 🚀 Quick Start

### 1. Install Jupyter (if not already installed)

```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install Jupyter
pip install jupyter notebook ipywidgets
```

### 2. Launch Jupyter Notebook

```bash
cd notebooks/
jupyter notebook
```

This will open Jupyter in your browser. Navigate to the notebooks directory.

### 3. Execute Notebooks in Order

Start with `00_Setup_and_Config.ipynb` and work your way through:

```
00_Setup_and_Config.ipynb
  ↓
01_Data_Preprocessing.ipynb
  ↓
02_KMeans_Clustering.ipynb  (or parallel with 03 & 04)
03_Hierarchical_Clustering.ipynb
04_DBSCAN_Clustering.ipynb
  ↓
05_Algorithm_Comparison.ipynb
```

---

## 📓 Notebook Descriptions

### **00_Setup_and_Config.ipynb**
**Purpose**: Environment setup and configuration
**Duration**: ~2 minutes
**Outputs**:
- Validated environment
- Loaded configuration
- Raw data preview
- Saved state for next notebooks

**Key Features**:
- Environment check
- Configuration display
- Data quality check
- Missing value overview

---

### **01_Data_Preprocessing.ipynb**
**Purpose**: Clean and prepare data for clustering
**Duration**: ~5 minutes
**Outputs**:
- `df_features`: Cleaned features
- `df_latest`: Latest snapshot
- `df_all`: Time-series data

**Key Features**:
- 🔧 **Configurable preprocessing** (imputation, CAGR calculation)
- 📊 **Interactive visualizations** of missing values
- 📈 **Feature correlation analysis**
- 🎨 **Distribution plots** for understanding data
- 💾 **State persistence** for next notebooks

**Adjustable Parameters**:
- Imputation method (mean, median, zero)
- Imputation threshold
- CAGR years
- Smoothing options

---

### **02_KMeans_Clustering.ipynb**
**Purpose**: K-Means clustering with parameter tuning
**Duration**: ~10 minutes
**Outputs**:
- Cluster assignments
- Cluster profiles
- Quality metrics (Silhouette, Calinski-Harabasz)

**Key Features**:
- 📊 **Elbow method** visualization
- 🎯 **Silhouette analysis** for K selection
- 🗺️ **PCA visualization** (2D & 3D)
- 📈 **Cluster profiling** with heatmaps
- 🔍 **Feature distributions** across clusters

**Adjustable Parameters**:
- Number of clusters (K)
- Feature set (static, dynamic, combined)
- Random seed for reproducibility
- Number of initializations

---

### **03_Hierarchical_Clustering.ipynb**
**Purpose**: Hierarchical clustering with dendrograms
**Duration**: ~8 minutes
**Outputs**:
- Cluster assignments
- Dendrogram visualization
- Comparison with K-Means

**Key Features**:
- 🌳 **Dendrogram** visualization
- 🔗 **Linkage methods** (ward, complete, average)
- 📊 **Cluster profiles**
- 🔄 **ARI comparison** with K-Means

**Adjustable Parameters**:
- Linkage method
- Number of clusters
- Dendrogram sample size

---

### **04_DBSCAN_Clustering.ipynb**
**Purpose**: Density-based clustering with outlier detection
**Duration**: ~10 minutes
**Outputs**:
- Cluster assignments
- Outlier detection
- Parameter recommendations

**Key Features**:
- 📏 **K-distance plot** for eps selection
- 🎛️ **Parameter grid search**
- 🚨 **Automatic outlier detection**
- 📊 **Cluster density analysis**

**Adjustable Parameters**:
- Epsilon (eps)
- Minimum samples (min_samples)
- Distance metric

---

### **05_Algorithm_Comparison.ipynb**
**Purpose**: Compare all clustering algorithms
**Duration**: ~7 minutes
**Outputs**:
- Quality metrics comparison
- Agreement analysis (ARI matrix)
- Visual comparisons

**Key Features**:
- 📊 **Side-by-side quality metrics**
- 🔗 **ARI matrix** (algorithm agreement)
- 🗺️ **PCA visualizations** for all algorithms
- 📈 **Cluster size distributions**
- 💡 **Recommendations** for algorithm selection

---

## 🔄 Workflow & State Management

### State Persistence

All notebooks use a **shared state system** that saves intermediate results:

```python
# Save results
state.save('df_latest', df_latest)

# Load in next notebook
df_latest = state.load('df_latest')
```

**Saved states** are stored in `notebooks/state/` and include:
- Preprocessed data
- Cluster labels
- Trained models
- Quality metrics

### Benefits

- ✅ **Resume work** from any point
- ✅ **No need to re-run** previous notebooks
- ✅ **Share states** between different analyses
- ✅ **Fast iteration** on parameters

---

## 🎨 Visualization Features

All notebooks include **rich visualizations**:

### Data Quality
- Missing value heatmaps
- Feature distributions
- Correlation matrices

### Clustering
- Cluster size distributions (bar + pie charts)
- PCA scatter plots (2D & 3D)
- Silhouette plots
- Dendrograms (hierarchical)
- K-distance plots (DBSCAN)

### Comparison
- Side-by-side algorithm comparisons
- ARI heatmaps
- Quality metric bar charts

---

## 🛠️ Customization

### Adjust Parameters

Each notebook has a **configuration cell** where you can adjust parameters:

```python
# Example from 02_KMeans_Clustering.ipynb
OPTIMAL_K = 5                 # Number of clusters
USE_FEATURE_SET = 'combined'  # static, dynamic, or combined
RANDOM_STATE = 42             # Reproducibility
```

### Add Custom Analyses

The `notebook_utils.py` module provides helper functions:

```python
from notebook_utils import (
    plot_cluster_distribution,
    plot_feature_distributions,
    plot_cluster_profiles,
    save_results
)
```

Add your own visualizations or analyses by extending these utilities.

---

## 📁 Output Structure

Results are saved to:

```
output/germany/notebooks/
├── kmeans/
│   ├── cluster_assignments.csv
│   ├── cluster_assignments.xlsx
│   ├── cluster_profiles.csv
│   └── cluster_profiles.xlsx
├── hierarchical/
│   └── ...
└── dbscan/
    └── ...
```

Additionally, **state files** are saved in:

```
notebooks/state/
├── germany_state.pkl       # Binary state data
└── germany_metadata.json   # Metadata about saved states
```

---

## 🔍 Troubleshooting

### "Data not found" Error

**Solution**: Run the previous notebooks first. The notebooks depend on each other:
- `01` depends on `00`
- `02`, `03`, `04` depend on `01`
- `05` depends on `02`, `03`, `04`

### Clear State and Start Fresh

```python
# In any notebook
state.clear()
```

This removes all saved states and forces you to re-run from the beginning.

### Kernel Issues

If the kernel crashes or becomes unresponsive:

1. **Restart kernel**: `Kernel → Restart & Clear Output`
2. **Re-run cells**: Execute all cells from the beginning

---

## 📊 Comparison: Notebooks vs. main.py

| Feature | Notebooks | main.py |
|---------|-----------|---------|
| **Interactivity** | ✅ High - adjust parameters on-the-fly | ❌ Low - config file only |
| **Visualizations** | ✅ Inline, immediate feedback | ⚠️ Saved to files |
| **State Management** | ✅ Resume from any point | ❌ Full re-run required |
| **Learning** | ✅ Great for exploration | ⚠️ Better for production |
| **Speed** | ⚠️ Manual execution | ✅ Automated pipeline |
| **Customization** | ✅ Easy to modify | ⚠️ Requires code changes |

**Recommendation**:
- Use **notebooks** for: Exploration, parameter tuning, learning, presentations
- Use **main.py** for: Production runs, batch processing, automated workflows

---

## 💡 Tips & Best Practices

1. **Always run in order** (at least the first time)
2. **Save your work**: State is persisted automatically, but notebook outputs are not
3. **Experiment freely**: You can always reload previous states
4. **Clear outputs** before committing to Git (reduces file size)
5. **Document your changes**: Add markdown cells to explain your analysis

---

## 🤝 Contributing

To add a new notebook:

1. Create a new `.ipynb` file
2. Import `notebook_utils` for common functions
3. Use `setup_notebook()` at the start
4. Save results with `state.save()`
5. Update this README with description

---

## 📚 Resources

- **Project Documentation**: `../README.md`
- **Configuration Guide**: `../config.yaml`
- **Source Code**: `../src/`
- **Output Structure**: `../output/germany/README.md`

---

## 🎓 For Your Thesis

### Using Notebooks for Research

The notebooks are designed to support your Master's thesis:

1. **Exploration**: Use notebooks to explore different parameter settings
2. **Visualization**: Generate publication-ready plots inline
3. **Documentation**: Markdown cells document your analytical process
4. **Reproducibility**: Share notebooks + state files for reproducible results

### Exporting Results

- **Plots**: Right-click on any plot → Save Image
- **Data**: Use `save_results()` to export to CSV/Excel
- **Notebook**: `File → Download as → PDF` (requires LaTeX)

---

**Happy Clustering! 🚀**

*If you have questions or issues, check the main project README or open an issue on GitHub.*
