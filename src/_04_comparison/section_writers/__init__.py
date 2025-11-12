"""
Section Writers Package - Modular Excel research report generation

This package contains specialized writers for each section of the research Excel file.
Each section addresses a specific research question from the Master thesis.

Structure:
- base_section_writer.py: Shared utilities and base class
- section_0_config_writer.py: Configuration and overview
- section_1_homogeneity_writer.py: Internal cluster quality analysis
- section_2_congruence_writer.py: Agreement with existing classifications
- section_3_drivers_writer.py: Key financial ratio drivers
- section_4_stability_writer.py: Temporal and size-based stability
- research_excel_coordinator.py: Main orchestrator

Usage:
    from src._04_comparison.section_writers import create_research_excel

    # Create comprehensive research Excel
    excel_path = create_research_excel(
        algorithm_results=results,
        output_dir=Path('output/comparison'),
        market='germany'
    )
"""

from .research_excel_coordinator import ResearchExcelCoordinator, create_research_excel
from .base_section_writer import BaseSectionWriter
from .section_0_config_writer import Section0ConfigWriter
from .section_1_homogeneity_writer import Section1HomogeneityWriter
from .section_2_congruence_writer import Section2CongruenceWriter
from .section_3_drivers_writer import Section3DriversWriter
from .section_4_stability_writer import Section4StabilityWriter

__all__ = [
    'ResearchExcelCoordinator',
    'create_research_excel',
    'BaseSectionWriter',
    'Section0ConfigWriter',
    'Section1HomogeneityWriter',
    'Section2CongruenceWriter',
    'Section3DriversWriter',
    'Section4StabilityWriter',
]

__version__ = '1.0.0'
__author__ = 'Master Thesis - Kennzahlenanalyse'
