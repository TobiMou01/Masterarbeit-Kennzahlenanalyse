"""
Comparison Phase - Algorithm and Feature Analysis
"""

from .comparison_pipeline import ComparisonPipeline
from .feature_analyzer import FeatureAnalyzer
from .temporal_analyzer import TemporalAnalyzer
from .company_analysis import CompanyAnalysis

# Excel Writers - NEW modular structure
from .section_writers import create_research_excel, ResearchExcelCoordinator

# Legacy (backward compatible)
from .research_excel_writer import ResearchExcelWriter  # Legacy
from .consolidated_excel_writer import ConsolidatedExcelWriter

__all__ = [
    'ComparisonPipeline',
    'FeatureAnalyzer',
    'TemporalAnalyzer',
    'CompanyAnalysis',
    # New modular Excel writers
    'create_research_excel',
    'ResearchExcelCoordinator',
    # Legacy
    'ResearchExcelWriter',
    'ConsolidatedExcelWriter',
]
