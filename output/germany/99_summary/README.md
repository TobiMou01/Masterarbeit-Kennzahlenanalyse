# Clustering Analysis Results - GERMANY

Generated: 2025-10-31 20:42:41

## Structure

### 01_data/
Processed feature data used for clustering

### 02_algorithms/
Clustering results per algorithm:
- `dbscan/` - DBSCAN results
  - `master_clustering/` - Static features analysis
  - `dynamic_enrichment/` - Dynamic features analysis
  - `combined_scores/` - Combined analysis

### 03_comparisons/
Cross-algorithm comparisons:
- `algorithms/` - Performance metrics comparison
- `gics/` - Independence from GICS sectors
- `features/` - Feature importance analysis
- `temporal/` - Temporal stability

### 99_summary/
Executive summaries and interpretation reports

## Mode

This analysis uses **HIERARCHICAL Mode**:

- Static creates **master cluster labels**
- Dynamic and Combined **reuse same labels**
- Only scores change, not cluster assignments
- Allows tracking companies across feature dimensions
