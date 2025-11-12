"""
Comparison Phase - Algorithm and Feature Analysis
"""

from .comparison_pipeline import ComparisonPipeline
from .feature_analyzer import FeatureImportance
from .temporal_analyzer import TemporalStability
from .company_analysis import CompanyClusterAnalyzer
from .gics_analyzer import GICSComparison
from .algorithm_analyzer import AlgorithmComparison as AlgorithmAnalyzer

# Excel Writers - NEW modular structure
from .section_writers import create_research_excel, ResearchExcelCoordinator

# Legacy (backward compatible)
from .research_excel_writer import ResearchExcelWriter  # Legacy
from .consolidated_excel_writer import ConsolidatedExcelWriter

__all__ = [
    'ComparisonPipeline',
    'FeatureImportance',
    'TemporalStability',
    'CompanyClusterAnalyzer',
    'GICSComparison',
    'AlgorithmAnalyzer',
    # New modular Excel writers
    'create_research_excel',
    'ResearchExcelCoordinator',
    # Legacy
    'ResearchExcelWriter',
    'ConsolidatedExcelWriter',
]
