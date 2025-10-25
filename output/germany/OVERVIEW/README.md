# Output Navigation - GERMANY Market

Generated: 2025-10-25 12:14:57

## Quick Start

1. **Executive Summary**: [`executive_summary.txt`](executive_summary.txt)
   - Key findings and best algorithm recommendation

2. **Cluster Lists**: [`cluster_lists/`](cluster_lists/)
   - Company assignments per algorithm
   - Excel files with separate sheets per cluster

3. **Metrics Comparison**: [`metrics_comparison.csv`](metrics_comparison.csv)
   - Algorithm performance comparison

## Folder Structure

### OVERVIEW/ (You are here)
Quick access to key results and navigation

### ALGORITHMS/
Detailed clustering results for each algorithm:
- `kmeans/combined/` - K-Means results
- `hierarchical/combined/` - Hierarchical clustering results
- `dbscan/combined/` - DBSCAN results

Each folder contains:
- `cluster_assignments.csv` - Company cluster assignments
- `cluster_profiles.csv` - Cluster characteristics
- `plots/` - Visualizations

### COMPARISONS/
Comparative analyses:

#### 01_vs_gics/
Clustering results vs GICS sector classification
- Cramér's V correlation analysis
- Contingency tables
- Comparison plots per algorithm

#### 02_algorithms/
Algorithm performance comparison
- Metrics comparison (Silhouette, Davies-Bouldin, etc.)
- Best algorithm recommendation

#### 03_feature_importance/
Feature importance analysis
- Combined plot across all algorithms
- Detailed CSV files per algorithm

#### 04_temporal/
Temporal stability analysis
- Cluster migration patterns
- Stability metrics
- Migration plots per algorithm



## File Naming Conventions

- `*_combined.*` - Analysis using combined static + dynamic features
- `*_static.*` - Analysis using only static features
- `*_dynamic.*` - Analysis using only dynamic features

## Tools Used

- K-Means Clustering
- Hierarchical Clustering (Ward's method)
- DBSCAN (Density-based clustering)

---
*Generated automatically by OutputReorganizer*
