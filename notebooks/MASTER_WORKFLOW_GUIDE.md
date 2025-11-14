# 🎯 MASTER Workflow Notebook - Complete Guide

## Overview

The **MASTER_Workflow.ipynb** notebook provides a comprehensive, interactive clustering pipeline that gives you complete control over every step of the analysis process. This is your one-stop solution for exploring, tuning, and comparing clustering algorithms for your Masterarbeit.

---

## 🚀 Quick Start

1. **Launch Jupyter**:
   ```bash
   cd notebooks
   jupyter notebook MASTER_Workflow.ipynb
   ```

2. **Run cells sequentially** from top to bottom, or jump to specific sections

3. **Adjust parameters** in configuration cells (marked with 🎛️)

4. **Re-run sections** with different settings to compare results

---

## 📋 What This Notebook Does

### Complete Pipeline Coverage

The notebook covers the **entire clustering workflow**:

```
Data Import → Preprocessing → Outlier Detection → Feature Selection
     ↓              ↓                ↓                   ↓
  PCA Toggle  →  Scaling  →  K-Means  →  Hierarchical  →  DBSCAN
                                ↓              ↓             ↓
                         Algorithm Comparison & Export
```

### Key Features

✅ **Full Control**: Adjust every parameter at every step
✅ **Visual Feedback**: See results immediately after each operation
✅ **Flexible**: Jump to any section and re-run with new settings
✅ **Interactive**: Toggle PCA on/off, switch feature sets, tune parameters
✅ **Comparative**: Run all algorithms and compare side-by-side
✅ **Reproducible**: All settings clearly documented in configuration cells

---

## 🎛️ Configuration Cells

Throughout the notebook, you'll find **configuration cells** marked with 🎛️. These are where you **adjust parameters** for each step.

### Section 3: Preprocessing Configuration

```python
IMPUTE_ENABLED = True           # Enable/disable imputation
IMPUTE_METHOD = 'median'        # Options: 'mean', 'median', 'zero'
IMPUTE_THRESHOLD = 0.5          # Drop columns with >50% missing
SMOOTH_STATIC = False           # Apply smoothing to static features
CAGR_YEARS = 3                  # Years for CAGR calculation
```

**When to adjust:**
- Change `IMPUTE_METHOD` if you want different missing value handling
- Adjust `IMPUTE_THRESHOLD` to be more/less strict about missing data
- Enable `SMOOTH_STATIC` to reduce noise in financial ratios

### Section 5: Feature Selection

```python
FEATURE_SET = 'combined'  # Options: 'static', 'dynamic', 'combined'
```

**Options:**
- `'static'`: Current financial ratios only (ROE, debt ratios, margins, etc.)
- `'dynamic'`: Time-series features (trends, volatility, growth rates)
- `'combined'`: All features together

**When to use which:**
- Use **static** for snapshot-based clustering (current financial health)
- Use **dynamic** for trajectory-based clustering (growth patterns)
- Use **combined** for comprehensive analysis (recommended)

### Section 6: PCA Configuration

```python
USE_PCA = True                  # Toggle PCA on/off
PCA_VARIANCE_THRESHOLD = 0.85   # Keep components explaining 85% variance
```

**When to enable PCA:**
- Many features (>10)
- High feature correlation
- Curse of dimensionality issues
- Better clustering quality needed

**When to disable PCA:**
- Few features (<10)
- Need feature interpretability
- Features already well-separated

**💡 Pro Tip**: Run the notebook once with PCA enabled, then go back to Section 6, set `USE_PCA = False`, and re-run from there to see the difference!

### Section 7: Scaling Configuration

```python
SCALER_TYPE = 'standard'  # Options: 'standard', 'robust'
```

**Options:**
- `'standard'`: Z-score normalization (mean=0, std=1) - **Default**
- `'robust'`: Median/IQR scaling - Better for outliers

### Section 8: K-Means Configuration

```python
KMEANS_K = best_sil_k        # Number of clusters
KMEANS_RANDOM_STATE = 42     # Random seed
KMEANS_N_INIT = 10           # Number of initializations
KMEANS_MAX_ITER = 300        # Maximum iterations
```

**How to choose K:**
1. Run the "Optimal K Selection" cells first
2. Look at the visualizations (Elbow, Silhouette, Calinski-Harabasz)
3. The notebook automatically suggests `best_sil_k` based on Silhouette score
4. You can override with any value from the k_range

### Section 9: Hierarchical Clustering Configuration

```python
HIER_LINKAGE = 'ward'        # Options: 'ward', 'complete', 'average', 'single'
HIER_N_CLUSTERS = KMEANS_K   # Use same K for comparison
```

**Linkage methods:**
- `'ward'`: Minimizes within-cluster variance (recommended)
- `'complete'`: Maximum distance between clusters
- `'average'`: Average distance between clusters
- `'single'`: Minimum distance between clusters

### Section 10: DBSCAN Configuration

```python
DBSCAN_EPS = suggested_eps   # Epsilon (neighborhood radius)
DBSCAN_MIN_SAMPLES = 5       # Minimum samples per cluster
```

**Parameter guidance:**
- The notebook automatically suggests `eps` based on k-distance plot
- Look for the "elbow" in the k-distance plot
- Increase `eps` to get fewer, larger clusters
- Decrease `eps` to get more, tighter clusters
- Increase `MIN_SAMPLES` to reduce noise sensitivity

---

## 🔄 Common Workflows

### Workflow 1: Quick Exploration

**Goal**: Get a quick overview of clustering results

```
1. Run all cells sequentially (Kernel → Restart & Run All)
2. Review results in Sections 8-11
3. Examine algorithm comparison in Section 11
```

**Time**: ~5-10 minutes

---

### Workflow 2: PCA Impact Analysis

**Goal**: Understand how PCA affects clustering quality

```
1. Run all cells with USE_PCA = True (Section 6)
2. Note the clustering quality metrics
3. Go back to Section 6, set USE_PCA = False
4. Re-run from Section 6 onwards (Cell → Run All Below)
5. Compare metrics before/after in Section 8
```

**What to compare:**
- Silhouette scores
- Number of PCA components vs original features
- Cluster separation in visualizations
- Computational speed

---

### Workflow 3: Feature Set Comparison

**Goal**: Compare static vs dynamic vs combined features

```
1. Set FEATURE_SET = 'static' (Section 5)
2. Run from Section 5 onwards
3. Note clustering quality and characteristics
4. Go back to Section 5, set FEATURE_SET = 'dynamic'
5. Re-run from Section 5 onwards
6. Compare with static results
7. Finally, set FEATURE_SET = 'combined' and compare
```

**What to look for:**
- Which feature set produces the best Silhouette scores?
- Do static and dynamic features produce similar clusters?
- Does combining them improve or degrade quality?

---

### Workflow 4: Optimal K Tuning

**Goal**: Find the best number of clusters

```
1. Run Section 8 "Optimal K Selection" cells
2. Examine all 4 metrics (Inertia, Silhouette, Calinski-H, Davies-Bouldin)
3. Look for consensus across metrics
4. If metrics disagree, try different k values:
   - Set KMEANS_K = 4, run clustering
   - Set KMEANS_K = 5, run clustering
   - Set KMEANS_K = 6, run clustering
5. Compare cluster distributions and profiles
```

**Decision criteria:**
- Silhouette score maximized
- Elbow in inertia curve
- Calinski-Harabasz maximized
- Davies-Bouldin minimized
- Business/domain sense (e.g., need at least 4 clusters for your analysis)

---

### Workflow 5: DBSCAN Parameter Tuning

**Goal**: Find optimal DBSCAN parameters

```
1. Run Section 10 k-distance plot
2. Visually identify the "elbow"
3. Try suggested_eps first
4. If results are poor (too many outliers or only 1 cluster):
   - Increase eps by 20-30% and re-run
   - Or decrease MIN_SAMPLES
5. Aim for:
   - 2-8 clusters
   - <30% outliers
   - Silhouette > 0.3
```

**Common issues:**
- **Only 1 cluster**: Increase `eps`
- **Too many outliers (>50%)**: Increase `eps` or decrease `MIN_SAMPLES`
- **Too many small clusters**: Increase `MIN_SAMPLES` or increase `eps`

---

### Workflow 6: Full Comparison

**Goal**: Systematically compare all algorithms and configurations

```
1. Create a comparison table manually or in a new cell:

| Config | Feature Set | PCA | K | Silhouette | Best Algorithm |
|--------|-------------|-----|---|------------|----------------|
| Run 1  | static      | Yes | 5 | 0.245      | K-Means        |
| Run 2  | dynamic     | Yes | 4 | 0.371      | K-Means        |
| Run 3  | combined    | Yes | 5 | 0.289      | Hierarchical   |
| Run 4  | combined    | No  | 5 | 0.198      | K-Means        |

2. For each configuration:
   - Adjust parameters
   - Re-run relevant sections
   - Record metrics

3. Identify best overall configuration
```

---

## 📊 Understanding the Outputs

### Section 3: Preprocessing Outputs

**What you see:**
- Data shape and summary statistics
- Missing value analysis
- Feature categorization (static vs dynamic)

**What to check:**
- Do you have enough data? (Aim for >100 companies)
- Are there excessive missing values? (>50% is concerning)
- Do static and dynamic feature counts make sense?

---

### Section 4: Outlier Detection

**What you see:**
- Outlier counts and percentages per feature
- Box plots showing outlier distribution

**What to check:**
- Which features have the most outliers?
- Are outliers due to data quality or genuine extreme values?
- Should you use RobustScaler instead of StandardScaler?

**Action items:**
- If >30% outliers in many features → Consider enabling robust scaling
- If outliers are data errors → Go back and fix in preprocessing

---

### Section 5: Feature Selection

**What you see:**
- Selected feature list
- Correlation matrix heatmap
- Highly correlated pairs

**What to check:**
- Are there many highly correlated features (|r| > 0.8)?
- If yes, PCA will be very helpful
- If no, you might not need PCA

---

### Section 6: PCA Analysis

**What you see:**
- Number of components selected
- Explained variance per component
- Cumulative variance plot
- Top feature loadings per component

**What to check:**
- **Dimensionality reduction**: How many features → components?
  - 21 features → 6 components is good (71% reduction)
  - 21 features → 18 components is poor (only 14% reduction)
- **Explained variance**: Should be ≥ 85%
- **Feature loadings**: Which original features contribute most to PC1, PC2, etc.?

**Interpretation example:**
```
PC1 (35% variance):
  roe                  0.456
  roa                  0.432
  ebit_margin          0.398
```
→ PC1 represents "profitability"

---

### Section 8: K-Means Results

**What you see:**
- Optimal k recommendations from multiple metrics
- Quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Cluster distribution (sizes)
- Cluster profiles (mean values per cluster)
- 2D PCA visualization

**What to check:**
- **Silhouette score**:
  - > 0.5 = Excellent
  - 0.3-0.5 = Good
  - 0.2-0.3 = Fair
  - < 0.2 = Poor
- **Cluster sizes**: Are clusters balanced or is one cluster huge?
- **Visual separation**: Do clusters separate well in PCA plot?

**Red flags:**
- One cluster has >70% of data → Increase k or change features
- Silhouette < 0.2 → Try different features or enable PCA
- Clusters overlap heavily in visualization → Poor separation

---

### Section 11: Algorithm Comparison

**What you see:**
- Quality metrics side-by-side
- Adjusted Rand Index (ARI) showing algorithm agreement
- Visual comparison in PCA space

**What to check:**
- **Which algorithm performed best?** (Highest Silhouette)
- **Do algorithms agree?** (High ARI = similar clusterings)
- **Visual consistency**: Do clusters look similar across algorithms?

**Interpretation:**
- **High ARI (>0.7)**: Algorithms agree strongly → Robust clustering
- **Low ARI (<0.3)**: Algorithms disagree → Weak cluster structure
- **Medium ARI (0.3-0.7)**: Some agreement → Different algorithms capture different aspects

---

## 🎯 Best Practices

### 1. Start Simple, Then Iterate

```
First run: Use defaults
  ↓
Analyze results
  ↓
Identify issues (low quality, imbalanced clusters, etc.)
  ↓
Adjust ONE parameter
  ↓
Re-run and compare
```

**Don't**: Change multiple parameters at once (you won't know what helped)
**Do**: Change one thing, observe, then change another

---

### 2. Document Your Experiments

Create a markdown cell with your findings:

```markdown
## Experiment Log

### Run 1 (Baseline)
- Features: combined, PCA: Yes (85%), K: 5
- Silhouette: 0.289
- Issue: Cluster 0 has 68% of data
- Action: Try k=6

### Run 2
- Features: combined, PCA: Yes (85%), K: 6
- Silhouette: 0.312 ✅ Improved!
- Clusters more balanced
- Action: Try dynamic features only

### Run 3
- Features: dynamic, PCA: Yes (85%), K: 4
- Silhouette: 0.371 ✅✅ Best so far!
- Conclusion: Dynamic features work better than combined
```

---

### 3. Save Important Results

After finding good results:

```python
# Add a new cell in Section 12:
import pickle

# Save the best model
best_config = {
    'model': kmeans,
    'labels': kmeans_labels,
    'features': selected_features,
    'pca': pca if USE_PCA else None,
    'metrics': {
        'silhouette': kmeans_silhouette,
        'k': KMEANS_K
    }
}

with open(output_dir / 'best_model.pkl', 'wb') as f:
    pickle.dump(best_config, f)

print("✅ Best model saved!")
```

---

### 4. Know When to Stop Iterating

**Good stopping points:**
- Silhouette score > 0.4
- Metrics plateaued (no improvement after 3-4 iterations)
- Results make business/domain sense
- You've found actionable insights

**Don't obsess over perfection** - clustering is exploratory, not predictive!

---

## 🆘 Troubleshooting

### Issue: "Data not found! Please run 01_Data_Preprocessing.ipynb first"

**Cause**: Notebook state not found
**Fix**: Run Section 3 (Data Preprocessing) instead of loading from state

---

### Issue: PCA gives only 1-2 components

**Cause**: Features have very high correlation
**Fix**: This is actually good! It means PCA is working well - your features are redundant and can be compressed

---

### Issue: K-Means gives terrible results (Silhouette < 0.1)

**Possible causes & fixes:**
1. **Too many features**: Enable PCA
2. **Wrong feature set**: Try static or dynamic instead of combined
3. **Poor k choice**: Look at optimal k selection plots
4. **Data quality issues**: Check for outliers, missing values

---

### Issue: DBSCAN finds only outliers

**Cause**: `eps` too small
**Fix**: Increase `DBSCAN_EPS` by 50-100%

---

### Issue: Notebook is slow

**Optimization tips:**
1. Reduce `DENDROGRAM_SAMPLE` in Section 9 (default: 500)
2. Use fewer PCA components: Set `PCA_N_COMPONENTS = 10` instead of variance threshold
3. Reduce `KMEANS_N_INIT` from 10 to 5
4. Skip DBSCAN if not needed

---

### Issue: Want to start fresh

```python
# Run this in a new cell:
%reset -f
# Then: Kernel → Restart & Run All
```

---

## 🎓 Advanced Usage

### Custom Analysis: Cluster Profiling

Add this cell after Section 8 to deep-dive into cluster characteristics:

```python
# Detailed cluster profiling
for cluster_id in range(KMEANS_K):
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} PROFILE")
    print(f"{'='*80}")

    # Get companies in this cluster
    cluster_companies = df_kmeans[df_kmeans['cluster_kmeans'] == cluster_id]

    print(f"\nSize: {len(cluster_companies)} companies ({len(cluster_companies)/len(df_kmeans)*100:.1f}%)")

    # Show top features that differentiate this cluster
    cluster_means = cluster_companies[selected_features[:10]].mean()
    overall_means = df_kmeans[selected_features[:10]].mean()

    diff = ((cluster_means - overall_means) / overall_means * 100).sort_values(ascending=False)

    print(f"\nMost distinctive features (% difference from overall mean):")
    print(diff.head(5))

    # Sample companies
    print(f"\nSample companies:")
    display(cluster_companies.head(3))
```

---

### Custom Visualization: 3D PCA Plot

Add after Section 8:

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.nipy_spectral(np.linspace(0, 1, KMEANS_K))

for cluster_id in range(KMEANS_K):
    mask = kmeans_labels == cluster_id
    ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
              c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
              alpha=0.6, s=50)

explained_3d = pca_3d.explained_variance_ratio_
ax.set_xlabel(f'PC1 ({explained_3d[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({explained_3d[1]:.1%})', fontsize=11)
ax.set_zlabel(f'PC3 ({explained_3d[2]:.1%})', fontsize=11)
ax.set_title(f'3D PCA Visualization\nTotal Variance: {explained_3d.sum():.1%}',
            fontsize=14, fontweight='bold')
ax.legend()
plt.show()
```

---

### Export for Masterarbeit

Add a cell in Section 12:

```python
# Create summary report for Masterarbeit
report = f"""
CLUSTERING ANALYSIS SUMMARY
{'='*80}

DATA
- Companies: {len(df_latest):,}
- Features: {len(selected_features)} ({FEATURE_SET})
- PCA: {'Enabled' if USE_PCA else 'Disabled'}
{f'- PCA Components: {n_components} (explaining {cumulative_variance[-1]:.1%} variance)' if USE_PCA else ''}

ALGORITHMS COMPARED
- K-Means (k={KMEANS_K})
- Hierarchical ({HIER_LINKAGE} linkage, k={HIER_N_CLUSTERS})
- DBSCAN (eps={DBSCAN_EPS:.3f}, min_samples={DBSCAN_MIN_SAMPLES})

RESULTS
- Best Algorithm: {df_comparison.loc[df_comparison['Silhouette'].idxmax(), 'Algorithm']}
- Best Silhouette: {df_comparison['Silhouette'].max():.3f}
- Algorithm Agreement (mean ARI): {df_ari.values[np.triu_indices_from(df_ari.values, k=1)].mean():.3f}

CLUSTER DISTRIBUTION (K-Means)
{chr(10).join([f'- Cluster {cid}: {count} companies ({count/len(df_kmeans)*100:.1f}%)' for cid, count in sorted(cluster_counts.items())])}

INTERPRETATION
[Add your interpretation here based on domain knowledge]
"""

# Save report
report_file = output_dir / 'summary_report.txt'
with open(report_file, 'w') as f:
    f.write(report)

print(report)
print(f"\n✅ Summary report saved: {report_file}")
```

---

## 📚 Further Reading

- **K-Means**: [Scikit-learn K-Means Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- **Hierarchical**: [Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- **DBSCAN**: [DBSCAN Guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- **PCA**: [PCA Tutorial](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- **Silhouette Score**: [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

---

## 💡 Tips for Your Masterarbeit

1. **Document your reasoning**: Why did you choose these features? Why this k value?

2. **Show the exploration process**: Include plots showing how you selected k

3. **Compare approaches**: Show results with/without PCA, different feature sets

4. **Interpret clusters**: Give them meaningful names based on profiles
   - Example: "High-Growth Tech Companies" instead of "Cluster 2"

5. **Validate results**:
   - Do clusters make business sense?
   - Do they align with industry knowledge?
   - Are they stable across different random seeds?

6. **Export high-quality figures**: Use `plt.savefig('figure.png', dpi=300, bbox_inches='tight')`

---

## 🎉 You're All Set!

This notebook gives you **complete control** over the clustering pipeline. Use it to:
- Explore different configurations
- Compare algorithms
- Find optimal parameters
- Generate insights for your Masterarbeit

**Happy analyzing!** 🚀

---

**Questions or issues?** Check the troubleshooting section or review the inline documentation in the notebook cells.
