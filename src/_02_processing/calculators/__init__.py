"""
Calculators Package - Modular Feature Engineering

This package contains specialized calculator modules for financial metrics:
- ratio_calculators: Profitability, Liquidity, Leverage, Efficiency ratios
- cashflow_calculators: Cashflow and structure metrics
- trend_calculators: Growth and trend analysis
- volatility_calculators: Volatility, consistency, and quality metrics
- feature_coordinator: Main orchestrator for all calculations
"""

from .ratio_calculators import (
    calculate_profitability_ratios,
    calculate_liquidity_ratios,
    calculate_leverage_ratios,
    calculate_efficiency_ratios
)

from .cashflow_calculators import (
    calculate_cashflow_metrics,
    calculate_structure_metrics
)

from .trend_calculators import (
    calculate_growth_metrics,
    calculate_dynamic_trends
)

from .volatility_calculators import (
    calculate_dynamic_volatility,
    handle_outliers,
    clean_calculated_features
)

from .feature_coordinator import (
    create_all_features,
    summary_statistics,
    save_features,
    main
)

__all__ = [
    # Ratio calculators
    'calculate_profitability_ratios',
    'calculate_liquidity_ratios',
    'calculate_leverage_ratios',
    'calculate_efficiency_ratios',
    # Cashflow calculators
    'calculate_cashflow_metrics',
    'calculate_structure_metrics',
    # Trend calculators
    'calculate_growth_metrics',
    'calculate_dynamic_trends',
    # Volatility calculators
    'calculate_dynamic_volatility',
    'handle_outliers',
    'clean_calculated_features',
    # Coordinator
    'create_all_features',
    'summary_statistics',
    'save_features',
    'main'
]
