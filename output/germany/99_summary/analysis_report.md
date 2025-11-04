# Clustering Analysis Summary Report

**Market:** germany
**Algorithm:** kmeans
**Generated:** 2025-11-04 17:02:47

---

## 1. Analysis Overview

### Static Analysis

- **Companies:** 158
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
0        9.189276    20.320039     68.319991       0.852453              -6.939956        1.438877     29.480350        0.513147         122.926772               61.745769   20.410408       178.992296            4.353014  High Performers
1        6.640527     9.065679     44.091642       2.515736              28.470430        0.299482     55.932212        0.758637          36.337976               77.701067    5.132504       127.387623            1.875213              Mid
2        0.975088     0.454319     23.491067       1.375396              10.129807        0.633702     32.102595        1.142759          34.410226              380.233004   -0.633347       383.904157            4.538706        Lower-Mid
3        6.437592     9.509628     36.559254       1.290982               8.668310        0.695786     35.079726        0.791055          35.753842               87.337295    6.416708       290.851922            3.122030        Upper-Mid
4       -1.005601   -18.699006     36.071367       1.251994              -0.655286        8.667402      4.769282        0.921881          40.433523               47.988986  -20.643119       318.899354           24.174840   Low Performers
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
