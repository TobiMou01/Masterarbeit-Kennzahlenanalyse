# Clustering Analysis Summary Report

**Market:** germany
**Algorithm:** kmeans
**Generated:** 2025-11-15 16:48:49

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
              roa        roe  ebit_margin  net_profit_margin  current_ratio  quick_ratio  debt_to_equity  equity_ratio  asset_turnover  revenue_per_employee cluster_name
cluster                                                                                                                                                                  
0        2.630379  19.229081     1.800161          -0.945263       1.701398     1.405368        3.009009     22.540973        1.400830           1070.571613       Tier 1
1        5.541389   4.237680     8.979662           4.150217       1.488735     1.055387        0.567764     38.932496        0.724763            441.045897       Tier 2
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
