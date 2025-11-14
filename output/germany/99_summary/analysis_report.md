# Clustering Analysis Summary Report

**Market:** germany
**Algorithm:** kmeans
**Generated:** 2025-11-14 13:25:42

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

- **Companies:** 152
- **Clusters:** 6
- **Silhouette Score:** N/A
- **Davies-Bouldin Index:** N/A

## 2. Cluster Profiles

### Static Analysis Profiles

```
              roa         roe  ebit_margin  net_profit_margin  current_ratio  quick_ratio  debt_to_equity  equity_ratio  asset_turnover  revenue_per_employee     cluster_name
cluster                                                                                                                                                                       
0        6.284933    6.780541     9.193421           5.482506       1.738951     1.174915        0.338953     49.523062        0.844991            410.741863        Upper-Mid
1        4.648844    3.428186     8.047534           2.668569       1.183871     0.917403        1.068857     26.067930        0.639157            555.064750        Lower-Mid
2        1.795009   42.566366     1.204411          -0.767157       2.212493     1.803597        0.677167     25.334460        1.832553           1230.156906  High Performers
3       -0.589630 -144.874819    -7.143895         -23.597034       0.999847     0.928391       25.059635      2.494306        0.539283            212.243147   Low Performers
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
