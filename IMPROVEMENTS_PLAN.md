# Research Excel Writer - Improvements Plan

## Status: 🚧 In Progress

### Requested Improvements:
1. ✅ PNG Embedding für bestehende Visualisierungen
2. ✅ Mehr Charts (Boxplots, Scatter Charts)
3. ✅ Bessere Beschreibungen für Klarheit

---

## 1. PNG Embedding Strategy

### Section 1b (Homogenität Charts):
- **performance_dashboard.png** (K-Means, Hierarchical, DBSCAN)
- **cluster_characteristics.png**
- Position: Below bar charts

### Section 2b (Kongruenz Charts):
- **algorithm_overlap.png** (if exists)
- Position: After contingency tables

### Section 3b (Treiber Charts):
- **correlation_heatmap.png** (K-Means, Hierarchical, DBSCAN)
- **pca_clusters.png** (if PCA enabled)
- Position: After feature importance bar chart

### Section 4b (Stabilität Charts):
- **temporal_stability.png** (placeholder for future)
- Position: TBD when temporal data available

---

## 2. New Charts to Add

### Boxplots (openpyxl doesn't support boxplots natively)
**Alternative: Use PNG embedding of matplotlib-generated boxplots**
- Section 1b: Score distribution boxplot per cluster
- Section 4b: Stability boxplot per algorithm

### Scatter Charts:
- **Section 1b**: Overall Score vs Proximity Score (colored by cluster)
- **Section 2b**: Multi-algorithm comparison matrix (all pairwise scatters)
- **Section 3b**: Feature 1 vs Feature 2 scatter (colored by cluster)

---

## 3. Description Improvements

### Add Interpretation Texts:
- What does this table show?
- What should I look for?
- What does "good" look like?

### Example Template:
```
┌─────────────────────────────────────────┐
│ 📊 WHAT THIS SHOWS:                     │
│ This table displays...                  │
│                                          │
│ 🔍 HOW TO INTERPRET:                    │
│ - High values indicate...               │
│ - Low values suggest...                 │
│                                          │
│ ✅ WHAT TO LOOK FOR:                    │
│ - Values > 0.8 are excellent            │
│ - Values < 0.3 need attention           │
└─────────────────────────────────────────┘
```

---

## Implementation Plan:

### Phase 1: PNG Embedding (Priority 1)
- [ ] Extend `_embed_png()` to handle missing files gracefully
- [ ] Add PNG embedding to Section 1b
- [ ] Add PNG embedding to Section 3b
- [ ] Use performance_dashboard.png for each algorithm

### Phase 2: Scatter Charts (Priority 2)
- [ ] Fix existing scatter chart in Section 2b
- [ ] Add score correlation scatter in Section 1b
- [ ] Add multi-algorithm scatter matrix in Section 2b

### Phase 3: Descriptions (Priority 3)
- [ ] Add interpretation boxes to Section 1a
- [ ] Add interpretation boxes to Section 2a
- [ ] Add interpretation boxes to Section 3a
- [ ] Add interpretation boxes to Section 4a

### Phase 4: Boxplot Alternative (Priority 4)
- [ ] Since openpyxl doesn't support boxplots, embed matplotlib PNGs instead
- [ ] Or: Use bar charts with error bars as workaround

---

## Technical Notes:

### PNG Paths:
```python
base_path = Path(f'output/{market}/02_algorithms/{algorithm}/{analysis_type}/plots/')
key_plots = [
    'performance_dashboard.png',
    'cluster_characteristics.png',
    'correlation_heatmap.png',
    'pca_clusters.png'
]
```

### Scatter Chart Fix (openpyxl):
```python
from openpyxl.chart.series import Series

series = Series(values=yvalues, xvalues=xvalues, title="Comparison")
chart.series.append(series)
```

### Interpretation Box Format:
```python
# Merged cells with border and light background
ws.merge_cells('A{row}:F{row+3}')
ws['A{row}'].value = "📊 WHAT THIS SHOWS: ..."
ws['A{row}'].fill = PatternFill(start_color='FFF4E6', fill_type='solid')
ws['A{row}'].border = Border(...)
```

---

## Expected File Size After Improvements:
- Current: ~1800 lines
- After improvements: ~2200 lines (+400 lines)
- New sections: None (enhancements only)
- New sheets: 0 (same 14 sheets, better content)

---

## Testing Checklist:
- [ ] PNGs embed correctly
- [ ] Scatter charts render
- [ ] Descriptions are readable
- [ ] No errors on missing files
- [ ] Excel file opens without issues
- [ ] Charts are interactive
