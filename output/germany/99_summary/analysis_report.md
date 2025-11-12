# Clustering Analysis Summary Report

**Market:** germany
**Algorithm:** kmeans
**Generated:** 2025-11-12 15:26:13

---

## 1. Analysis Overview

### Static Analysis

- **Companies:** 160
- **Clusters:** 5
- **Silhouette Score:** N/A
- **Davies-Bouldin Index:** N/A

### Dynamic Analysis

- **Companies:** 152
- **Clusters:** 5
- **Silhouette Score:** N/A
- **Davies-Bouldin Index:** N/A

### Combined Analysis

- **Companies:** 151
- **Clusters:** 6
- **Silhouette Score:** N/A
- **Davies-Bouldin Index:** N/A

## 2. Cluster Profiles

### Static Analysis Profiles

```
              roa  ebit_margin  gross_margin  current_ratio  working_capital_ratio  debt_to_equity  equity_ratio  asset_turnover  capital_intensity  days_sales_outstanding  fcf_margin  cash_conversion  financial_leverage     cluster_name
cluster                                                                                                                                                                                                                                      
0        2.457056     3.830645     25.997263       1.560476               9.230382        0.465610     34.531250        0.794929          35.747473               82.803937    2.978127       286.977753            3.286326        Lower-Mid
1        7.203344     9.197602     41.475376       1.965459              23.212707        0.414212     47.432913        1.003854          26.601068               68.078073    6.962533       111.267585            2.272898              Mid
2        8.458130    12.959641     43.986508       0.844080              -8.162427        1.364182     24.192068        0.871066          30.282775              556.716998   75.176434       188.447172            6.115249        Upper-Mid
3       -4.362073   -38.514768     47.585842       1.693165              10.963223       18.424556      2.469739        0.398719          79.261662               73.181514  -35.192445       491.825272           43.584454   Low Performers
4        4.865643    27.521765     42.870491       1.154180               0.000555        1.440482     36.300741        0.489997         285.944294               96.538911    6.552883      1199.277894            3.533524  High Performers
```


## 3. Validation Results

*External validation results can be found in 3_external_validation/*

## 4. Score Analysis

*Detailed score analysis can be found in 1_cluster_quality/scores/*

## 5. Output Structure

```
output/germany/02_algorithms/kmeans_comparative/
├── 1_cluster_quality/
│   ├── scores/
│   ├── naming/
│   └── visualizations/
├── 2_algorithm_congruence/
├── 3_external_validation/
├── 4_company_insights/
└── 5_pca_analysis/
```
