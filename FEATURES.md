# Financial Ratios Documentation

Comprehensive documentation of all financial ratios calculated in the Feature Engineering pipeline.

**Total Features: ~40 Ratios**
- **Static Features**: ~25 (snapshot metrics)
- **Dynamic Features**: ~15 (time-series metrics)

---

## Table of Contents

1. [Profitability Ratios](#profitability-ratios) (8 metrics)
2. [Liquidity Ratios](#liquidity-ratios) (5 metrics)
3. [Leverage Ratios](#leverage-ratios) (7 metrics)
4. [Efficiency Ratios](#efficiency-ratios) (8 metrics)
5. [Cashflow Metrics](#cashflow-metrics) (5 metrics)
6. [Structure Metrics](#structure-metrics) (4 metrics)
7. [Dynamic Metrics](#dynamic-metrics) (15 metrics)
8. [Data Sources](#data-sources)
9. [Usage Guidelines](#usage-guidelines)

---

## Profitability Ratios

Measure a company's ability to generate profit relative to various bases.

### 1. ROA (Return on Assets)

**Formula:** `(EBIT / Total Assets) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 15%**: Very good - company generates strong returns
- **10-15%**: Good
- **5-10%**: Average
- **< 5%**: Poor asset utilization

**What it measures:** How efficiently a company uses its assets to generate profit, before considering capital structure.

**Data Sources:** `ebit`, `at`

---

### 2. ROE (Return on Equity)

**Formula:** `(Income Before Extraordinary Items / Stockholders Equity) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 20%**: Excellent
- **15-20%**: Very good
- **10-15%**: Average
- **< 10%**: Underperforming

**What it measures:** Return generated for equity investors. High leverage can inflate ROE.

**Data Sources:** `ib`, `seq`

**Note:** Can be misleading for highly leveraged companies - compare with ROA.

---

### 3. EBIT Margin

**Formula:** `(EBIT / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 20%**: Excellent operating efficiency
- **10-20%**: Good
- **5-10%**: Average
- **< 5%**: Low margins

**What it measures:** Operating profitability before interest and taxes. Good for comparing companies with different capital structures.

**Data Sources:** `ebit`, `revt`

---

### 4. EBITDA Margin

**Formula:** `(EBITDA / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- Similar to EBIT Margin but excludes depreciation
- Useful for capital-intensive industries

**What it measures:** Cash-based operating performance.

**Data Sources:** `ebitda`, `revt`

---

### 5. Net Profit Margin

**Formula:** `(Income Before Extraordinary / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 15%**: Excellent
- **10-15%**: Good
- **5-10%**: Average
- **< 5%**: Thin margins

**What it measures:** Bottom-line profitability after all expenses.

**Data Sources:** `ib`, `revt`

---

### 6. Operating Margin

**Formula:** `(Operating Income Before D&A / Revenue) × 100`

**Unit:** Percent (%)

**What it measures:** Core operating profitability before depreciation and amortization.

**Data Sources:** `oibdp`, `revt`

---

### 7. Gross Margin

**Formula:** `((Revenue - COGS) / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 60%**: High - software, pharma
- **40-60%**: Good - many tech companies
- **20-40%**: Average - retail, manufacturing
- **< 20%**: Low - commodities

**What it measures:** Pricing power and production efficiency before operating expenses.

**Data Sources:** `revt`, `cogs`

**Note:** Highly industry-specific.

---

### 8. ROC (Return on Capital)

**Formula:** `(EBIT / (Equity + Long-term Debt)) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 15%**: Creating value
- **10-15%**: Good
- **< 10%**: May be destroying value (if < cost of capital)

**What it measures:** Return on all capital invested (debt + equity). Similar to ROIC.

**Data Sources:** `ebit`, `seq`, `dltt`

---

## Liquidity Ratios

Measure a company's ability to meet short-term obligations.

### 1. Current Ratio

**Formula:** `Current Assets / Current Liabilities`

**Unit:** Ratio

**Interpretation:**
- **> 2.0**: Very strong
- **1.5-2.0**: Good
- **1.0-1.5**: Adequate
- **< 1.0**: Potential liquidity issues

**What it measures:** Ability to pay short-term liabilities with short-term assets.

**Data Sources:** `act`, `lct`

---

### 2. Quick Ratio (Acid Test)

**Formula:** `(Current Assets - Inventory) / Current Liabilities`

**Unit:** Ratio

**Interpretation:**
- **> 1.5**: Strong
- **1.0-1.5**: Adequate
- **< 1.0**: May struggle with immediate obligations

**What it measures:** More conservative than current ratio - excludes inventory.

**Data Sources:** `act`, `invt`, `lct`

---

### 3. Cash Ratio

**Formula:** `Cash / Current Liabilities`

**Unit:** Ratio

**Interpretation:**
- **> 0.5**: Very strong cash position
- **0.2-0.5**: Adequate
- **< 0.2**: Relies on operations/credit

**What it measures:** Most conservative liquidity measure.

**Data Sources:** `che`, `lct`

---

### 4. Cash Ratio Enhanced

**Formula:** `(Cash + Short-term Investments) / Current Liabilities`

**Unit:** Ratio

**What it measures:** Includes marketable securities as liquid assets.

**Data Sources:** `che`, `ivst`, `lct`

**Note:** Falls back to regular cash ratio if `ivst` not available.

---

### 5. Working Capital Ratio

**Formula:** `((Current Assets - Current Liabilities) / Total Assets) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 20%**: Strong
- **10-20%**: Adequate
- **< 10%**: Tight working capital
- **< 0%**: Negative working capital

**What it measures:** Working capital as a percentage of total assets.

**Data Sources:** `act`, `lct`, `at`

---

## Leverage Ratios

Measure the degree of financial leverage and capital structure.

### 1. Debt-to-Equity

**Formula:** `Long-term Debt / Stockholders Equity`

**Unit:** Ratio

**Interpretation:**
- **< 0.5**: Conservative
- **0.5-1.0**: Moderate
- **1.0-2.0**: Leveraged
- **> 2.0**: Highly leveraged

**What it measures:** Financial leverage - how much debt relative to equity.

**Data Sources:** `dltt`, `seq`

---

### 2. Total Debt-to-Equity

**Formula:** `(Short-term Debt + Long-term Debt) / Equity`

**Unit:** Ratio

**What it measures:** Includes all debt, not just long-term.

**Data Sources:** `dlc`, `dltt`, `seq`

---

### 3. Equity Ratio

**Formula:** `(Equity / Total Assets) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 50%**: Strong equity base
- **40-50%**: Good
- **30-40%**: Moderate
- **< 30%**: Highly leveraged

**What it measures:** Proportion of assets financed by equity.

**Data Sources:** `seq`, `at`

---

### 4. Debt Ratio

**Formula:** `(Total Liabilities / Total Assets) × 100`

**Unit:** Percent (%)

**What it measures:** Proportion of assets financed by debt.

**Data Sources:** `lt`, `at`

**Note:** Equity Ratio + Debt Ratio ≈ 100%

---

### 5. Interest Coverage

**Formula:** `EBIT / Interest Expense`

**Unit:** Ratio (times)

**Interpretation:**
- **> 5**: Very safe
- **3-5**: Adequate
- **1.5-3**: Tight
- **< 1.5**: At risk of default

**What it measures:** Ability to pay interest on debt.

**Data Sources:** `ebit`, `xint`

**Special Handling:** Only calculated when `xint > 0`, otherwise `NaN`.

---

### 6. Net Debt to EBITDA

**Formula:** `(Total Debt - Cash) / EBITDA`

**Unit:** Ratio (years)

**Interpretation:**
- **< 2**: Strong
- **2-3**: Healthy
- **3-5**: Moderate leverage
- **> 5**: High leverage risk

**What it measures:** How many years of EBITDA needed to pay off net debt.

**Data Sources:** `dlc`, `dltt`, `che`, `ebitda`

---

### 7. Debt to Assets

**Formula:** `((Short-term + Long-term Debt) / Total Assets) × 100`

**Unit:** Percent (%)

**What it measures:** Total debt burden relative to asset base.

**Data Sources:** `dlc`, `dltt`, `at`

---

## Efficiency Ratios

Measure how effectively a company uses its assets and resources.

### 1. Asset Turnover

**Formula:** `Revenue / Total Assets`

**Unit:** Ratio (times per year)

**Interpretation:**
- **> 2**: Highly efficient (retail, services)
- **1-2**: Good
- **0.5-1**: Average (capital-intensive)
- **< 0.5**: Low (utilities, heavy industry)

**What it measures:** Revenue generated per dollar of assets.

**Data Sources:** `revt`, `at`

**Note:** Highly industry-specific.

---

### 2. Revenue per Employee

**Formula:** `Revenue / Number of Employees`

**Unit:** Thousands (USD or local currency)

**Interpretation:**
- Varies wildly by industry
- Tech/Finance: often > $500k
- Retail: often < $200k

**What it measures:** Labor productivity.

**Data Sources:** `revt`, `emp`

---

### 3. Receivables Turnover

**Formula:** `Revenue / Receivables`

**Unit:** Ratio (times per year)

**Interpretation:**
- Higher is better (faster collection)
- Industry-specific

**What it measures:** How quickly receivables are collected.

**Data Sources:** `revt`, `rect`

---

### 4. Days Sales Outstanding (DSO)

**Formula:** `365 / Receivables Turnover`

**Unit:** Days

**Interpretation:**
- **< 30 days**: Excellent
- **30-45**: Good
- **45-60**: Average
- **> 60**: May indicate collection issues

**What it measures:** Average days to collect receivables.

**Derived from:** `receivables_turnover`

---

### 5. Capital Intensity

**Formula:** `(PP&E / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 100%**: Highly capital-intensive (utilities, manufacturing)
- **50-100%**: Moderate
- **< 50%**: Capital-light (services, tech)

**What it measures:** Asset intensity of business model.

**Data Sources:** `ppent`, `revt`

---

### 6. Working Capital Turnover

**Formula:** `Revenue / (Current Assets - Current Liabilities)`

**Unit:** Ratio

**What it measures:** Efficiency of working capital usage.

**Data Sources:** `revt`, `act`, `lct`

**Special Handling:** Only calculated when working capital > 0.

---

### 7. Inventory Turnover

**Formula:** `COGS / Inventory`

**Unit:** Ratio (times per year)

**Interpretation:**
- Higher generally better (fresher inventory)
- Perishables: very high
- Luxury goods: lower

**What it measures:** How quickly inventory is sold and replaced.

**Data Sources:** `cogs`, `invt`

**Special Handling:** Only calculated when `invt > 0`.

---

### 8. Asset Quality

**Formula:** `((Cash + Receivables) / Total Assets) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 40%**: High liquid assets
- **20-40%**: Moderate
- **< 20%**: Asset-heavy

**What it measures:** Proportion of assets that are easily converted to cash.

**Data Sources:** `che`, `rect`, `at`

---

## Cashflow Metrics

Measure cash generation and investment activity.

### 1. Capex to Revenue

**Formula:** `(|Capex| / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 20%**: High investment phase
- **10-20%**: Growth mode
- **5-10%**: Maintenance
- **< 5%**: Low capex (services, mature tech)

**What it measures:** Investment intensity.

**Data Sources:** `capx`, `revt`

**Note:** Capex is typically negative, we use absolute value.

---

### 2. Capex to Depreciation

**Formula:** `|Capex| / Depreciation`

**Unit:** Ratio

**Interpretation:**
- **> 1.5**: Growing asset base
- **1.0-1.5**: Replacement rate
- **< 1.0**: Underinvesting (assets aging)

**What it measures:** Whether company is growing or just maintaining assets.

**Data Sources:** `capx`, `dp`

---

### 3. FCF Margin

**Formula:** `((Operating Cash Flow - |Capex|) / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 15%**: Excellent cash generation
- **10-15%**: Good
- **5-10%**: Moderate
- **< 5%**: Weak FCF generation
- **< 0%**: Negative FCF

**What it measures:** Free cash flow as % of revenue - true cash profitability.

**Data Sources:** `oancf`, `capx`, `revt`

**Note:** Free Cash Flow = Operating CF - Capex

---

### 4. Reinvestment Rate

**Formula:** `(|Capex| / Operating Cash Flow) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 100%**: Investing more than generating (growth mode)
- **50-100%**: High reinvestment
- **< 50%**: Conservative or returning cash

**What it measures:** Proportion of cash flow reinvested in business.

**Data Sources:** `capx`, `oancf`

---

### 5. Cash Conversion

**Formula:** `(Operating Cash Flow / EBIT) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 100%**: Better cash than earnings (good)
- **80-100%**: Healthy conversion
- **< 80%**: Working capital buildup or quality issues

**What it measures:** How well earnings translate to cash.

**Data Sources:** `oancf`, `ebit`

---

## Structure Metrics

Measure corporate structure and capital allocation.

### 1. Financial Leverage

**Formula:** `Total Assets / Equity`

**Unit:** Ratio

**Interpretation:**
- **< 2**: Conservative
- **2-3**: Moderate
- **> 3**: High leverage

**What it measures:** DuPont component - how much assets per dollar of equity.

**Data Sources:** `at`, `seq`

**Note:** Part of DuPont analysis: ROE = ROA × Financial Leverage

---

### 2. R&D Intensity

**Formula:** `(R&D Expense / Revenue) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 15%**: Very R&D intensive (pharma, biotech)
- **10-15%**: High (tech)
- **5-10%**: Moderate
- **< 5%**: Low R&D

**What it measures:** Innovation investment intensity.

**Data Sources:** `xrd`, `revt`

**Note:** Only available if company reports R&D separately.

---

### 3. Dividend Payout Ratio

**Formula:** `(Total Dividends / Net Income) × 100`

**Unit:** Percent (%)

**Interpretation:**
- **> 75%**: High payout (mature, low growth)
- **40-75%**: Moderate
- **< 40%**: Low payout (growth mode)
- **0%**: No dividends

**What it measures:** Proportion of earnings distributed to shareholders.

**Data Sources:** `dvt`, `ni`

---

### 4. Retention Ratio

**Formula:** `100 - Dividend Payout Ratio`

**Unit:** Percent (%)

**What it measures:** Proportion of earnings retained for reinvestment.

**Derived from:** `dividend_payout_ratio`

---

## Dynamic Metrics

Time-series based metrics measuring change, trends, and stability.

### Growth Metrics (Year-over-Year)

#### 1. Revenue Growth

**Formula:** `Year-over-year % change in Revenue`

**Unit:** Percent (%)

**Interpretation:**
- **> 20%**: High growth
- **10-20%**: Good growth
- **0-10%**: Moderate growth
- **< 0%**: Declining

**Data Sources:** `revt` (time-series)

---

#### 2. Asset Growth

**Formula:** `Year-over-year % change in Total Assets`

**Unit:** Percent (%)

**Data Sources:** `at` (time-series)

---

#### 3. Employee Growth

**Formula:** `Year-over-year % change in Employees`

**Unit:** Percent (%)

**Data Sources:** `emp` (time-series)

---

#### 4. FCF Growth

**Formula:** `Year-over-year % change in Free Cash Flow`

**Unit:** Percent (%)

**What it measures:** Cash generation growth rate.

**Derived from:** `fcf` (time-series)

---

#### 5. Capex Growth

**Formula:** `Year-over-year % change in Capex`

**Unit:** Percent (%)

**What it measures:** Investment acceleration/deceleration.

**Data Sources:** `capx` (time-series)

---

### Trend Metrics (Linear Regression Slope)

**Minimum Years Required:** 3

#### 6. Margin Trend

**Formula:** `Slope of EBIT Margin over time (linear regression)`

**Unit:** Percentage points per year

**Interpretation:**
- **> +1 pp/year**: Strong margin expansion
- **0 to +1**: Improving margins
- **-1 to 0**: Deteriorating margins
- **< -1 pp/year**: Significant margin pressure

**What it measures:** Direction of profitability trend.

**Data Sources:** `ebit_margin`, `fyear`

---

#### 7. Leverage Trend

**Formula:** `Slope of Total Debt-to-Equity over time`

**Unit:** Ratio points per year

**Interpretation:**
- **Positive**: Increasing leverage
- **Negative**: Deleveraging

**Data Sources:** `total_debt_to_equity`, `fyear`

---

#### 8. Capex Trend

**Formula:** `Slope of Capex-to-Revenue over time`

**Unit:** Percentage points per year

**Data Sources:** `capex_to_revenue`, `fyear`

---

#### 9. FCF Trend

**Formula:** `Slope of FCF Margin over time`

**Unit:** Percentage points per year

**Data Sources:** `fcf_margin`, `fyear`

---

### Volatility Metrics (Standard Deviation)

**Minimum Years Required:** 3

#### 10. Margin Volatility

**Formula:** `Standard deviation of EBIT Margin over time`

**Unit:** Percentage points

**Interpretation:**
- **< 3 pp**: Very stable
- **3-5 pp**: Moderate stability
- **> 5 pp**: High volatility

**What it measures:** Earnings stability/cyclicality.

**Data Sources:** `ebit_margin` (time-series)

---

#### 11. Leverage Volatility

**Formula:** `Standard deviation of Total Debt-to-Equity`

**Unit:** Ratio points

**Data Sources:** `total_debt_to_equity` (time-series)

---

#### 12. Cashflow Volatility

**Formula:** `Standard deviation of (Operating CF / Assets)`

**Unit:** Percentage points

**What it measures:** Cash flow stability.

**Data Sources:** `oancf`, `at` (time-series)

---

### Quality Metrics

#### 13. Margin Consistency

**Formula:** `R² of linear regression of margin over time`

**Unit:** 0 to 1

**Interpretation:**
- **> 0.8**: Very predictable margins
- **0.5-0.8**: Moderate predictability
- **< 0.5**: Erratic margins

**What it measures:** How predictable/consistent margins are.

**Data Sources:** `ebit_margin`, `fyear`

---

#### 14. Growth Quality

**Formula:** `Correlation between Revenue Growth and FCF Growth`

**Unit:** -1 to +1

**Interpretation:**
- **> 0.7**: High-quality growth (cash-backed)
- **0.3-0.7**: Moderate quality
- **< 0.3**: Revenue growth not converting to cash

**What it measures:** Whether revenue growth translates to cash generation.

**Data Sources:** `revenue_growth`, `fcf_growth`

---

## Data Sources

### WRDS Compustat Data Items

| Mnemonic | Description | Usage |
|----------|-------------|-------|
| `at` | Total Assets | Denominator for many ratios |
| `act` | Current Assets | Liquidity |
| `che` | Cash and Equivalents | Liquidity |
| `rect` | Receivables | Efficiency |
| `invt` | Inventory | Efficiency |
| `ppent` | PP&E Net | Capital intensity |
| `lt` | Total Liabilities | Leverage |
| `lct` | Current Liabilities | Liquidity |
| `dlc` | Short-term Debt | Leverage |
| `dltt` | Long-term Debt | Leverage |
| `seq` | Stockholders Equity | Leverage, profitability |
| `revt` | Revenue | Denominator for margins |
| `cogs` | Cost of Goods Sold | Gross margin |
| `ebit` | EBIT | Profitability |
| `ebitda` | EBITDA | Cashflow proxy |
| `ib` | Income Before Extraordinary | Net income |
| `ni` | Net Income | Dividend payout |
| `oibdp` | Operating Income Before D&A | Operating margin |
| `oancf` | Operating Cash Flow | Cashflow metrics |
| `capx` | Capital Expenditures | Investment |
| `xint` | Interest Expense | Coverage ratio |
| `xrd` | R&D Expense | Innovation |
| `dvt` | Dividends | Payout ratio |
| `emp` | Employees | Productivity |
| `dp` | Depreciation | Capex comparison |

---

## Usage Guidelines

### For Static Analysis

**Recommended Features:**
- Profitability: `roa`, `ebit_margin`, `gross_margin`
- Liquidity: `current_ratio`, `working_capital_ratio`
- Leverage: `debt_to_equity`, `equity_ratio`
- Efficiency: `asset_turnover`, `capital_intensity`
- Cashflow: `fcf_margin`, `cash_conversion`

**Avoid:**
- Highly correlated pairs (e.g., `current_ratio` + `quick_ratio` + `cash_ratio`)
- Industry-specific metrics for cross-industry analysis

### For Dynamic Analysis

**Recommended Features:**
- Growth: `revenue_growth`, `fcf_growth`
- Trends: `margin_trend`, `leverage_trend`
- Volatility: `margin_volatility`, `cashflow_volatility`
- Quality: `margin_consistency`, `growth_quality`

**Requirements:**
- Minimum 3 years of data for trends/volatility
- Minimum 5 years recommended for stability metrics

### For PCA Analysis

**Optimization Tips:**
1. **Remove one from highly correlated pairs:**
   - Keep `roa`, drop `roe` (correlated via leverage)
   - Keep `current_ratio`, drop `quick_ratio` and `cash_ratio`

2. **Balance categories:**
   - 2-3 from profitability
   - 1-2 from liquidity
   - 2 from leverage
   - 2-3 from efficiency
   - 2 from cashflow
   - 2-4 from dynamic

3. **Target 15-25 features total** for effective PCA with ~100-200 companies

### Handling Missing Data

**Common Issues:**
- `xint`: Not all companies report separately (often in `xsga`)
- `xrd`: Only reported if material (tech, pharma)
- `capx`: Usually available, but may be in aggregated cashflow
- `dvt`: Zero for non-dividend payers (not missing)
- `invt`: Zero for service companies (not missing)

**Strategies:**
1. **Imputation:** Not recommended for financial ratios
2. **Exclusion:** Drop companies with > 30% missing features
3. **Subset Analysis:** Use only features with < 20% missingness
4. **Category Fallback:** If `interest_coverage` missing, use `net_debt_to_ebitda`

### Industry Considerations

**Capital-Intensive (Utilities, Manufacturing, Telecom):**
- Focus on: `capex_to_revenue`, `capital_intensity`, `debt_to_ebitda`
- De-emphasize: `asset_turnover` (naturally low)

**Capital-Light (Software, Services, Consulting):**
- Focus on: `gross_margin`, `revenue_per_employee`, `fcf_margin`
- Less relevant: `inventory_turnover`, `capex_to_depreciation`

**Financial Services:**
- Most ratios not applicable
- Use specialized financial metrics instead

**Retail:**
- Critical: `inventory_turnover`, `days_sales_outstanding`, `working_capital_turnover`

---

## Calculation Notes

### Division by Zero

All ratios handle division by zero:
- Result set to `NaN` (not 0 or infinity)
- Outlier detection skips `NaN` values

### Negative Values

**Debt-based ratios:**
- Negative equity → ratio set to `NaN`
- Negative EBITDA → `net_debt_to_ebitda` may be misleading

**Growth rates:**
- Negative baseline → growth % can exceed ±100%
- Outlier detection critical

### Outlier Treatment

**Default: IQR Method**
- Outliers flagged in `*_outlier` columns
- Not automatically removed (user decision)
- Thresholds configurable in `features_config.yaml`

**When to use Z-Score:**
- When data is approximately normal
- For growth rates and trends

---

## Configuration

All features can be enabled/disabled in [`features_config.yaml`](features_config.yaml):

```yaml
profitability:
  enabled: true
  metrics:
    roa:
      enabled: true
      outlier_threshold: 3
    # ...
```

Use **presets** for common scenarios:
- `minimal`: Quick analysis (5-7 features)
- `standard`: Regular analysis (15-20 features)
- `comprehensive`: All features (~40)
- `pca_optimized`: Optimized for dimensionality reduction (20-25 features)

---

## References

### Financial Analysis Resources
- Damodaran, A. (2012). *Investment Valuation*
- Penman, S. (2012). *Financial Statement Analysis and Security Valuation*
- McKinsey & Company. (2015). *Valuation: Measuring and Managing the Value of Companies*

### Academic Papers
- Fama, E. F., & French, K. R. (1992). The Cross-Section of Expected Stock Returns
- Piotroski, J. D. (2000). Value Investing: The Use of Historical Financial Statement Information

### Data Documentation
- WRDS Compustat Manual: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/compustat/
- Compustat North America Data Guide

---

**Last Updated:** 2025-10-31
**Version:** 2.0
**Maintainer:** Feature Engineering Pipeline
