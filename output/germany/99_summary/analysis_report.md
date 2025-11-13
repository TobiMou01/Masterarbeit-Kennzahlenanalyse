# Clustering Analysis Summary Report

**Market:** germany
**Algorithm:** kmeans
**Generated:** 2025-11-13 15:17:22

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
              roa  ebit_margin  gross_margin  current_ratio  working_capital_ratio  debt_to_equity  equity_ratio  asset_turnover  capital_intensity  days_sales_outstanding  fcf_margin  cash_conversion  financial_leverage cluster_name
cluster                                                                                                                                                                                                                                  
0        7.193581    12.757447     43.748516       1.647198              10.207753        0.755749     40.538206        0.656546          57.176425              228.882460   27.661123       142.875844            2.953989       Tier 1
1        1.180305    -1.064861     20.934763       1.316391               8.744270        1.457793     27.912900        1.182765          29.109911               62.999921   -0.782576       571.475401            6.690880       Tier 2
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
