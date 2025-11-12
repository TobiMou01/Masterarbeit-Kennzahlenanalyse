"""
Section 0 Writer - Configuration & Overview

Creates the configuration and overview sheet showing:
- Analysis parameters from config.yaml
- Sample statistics (companies, algorithms, outliers)
- Cluster distribution by sector
- Overview charts
"""

import pandas as pd
import logging
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference
from .base_section_writer import BaseSectionWriter

logger = logging.getLogger(__name__)


class Section0ConfigWriter(BaseSectionWriter):
    """Writer for Section 0: Config & Overview"""

    def create_section(self, wb: Workbook, df: pd.DataFrame):
        """
        Create Section 0: Config & Overview

        Args:
            wb: Workbook object
            df: Overview dataframe with all algorithm results
        """
        ws = wb.create_sheet("0_Config_Overview")

        # Title
        ws['A1'] = "RESEARCH ANALYSIS - CONFIGURATION & OVERVIEW"
        ws['A1'].font = Font(size=16, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws['A1'].alignment = Alignment(horizontal='center')
        ws.merge_cells('A1:H1')

        row = 3

        # ===== CONFIG PARAMETERS =====
        ws[f'A{row}'] = "ANALYSIS PARAMETERS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:B{row}')
        row += 1

        # Read parameters from config
        params = [
            ("Market", self.config.get('data', {}).get('market', 'germany')),
            ("Algorithms", ', '.join([a.capitalize() for a in df['algorithm'].unique()])),
            ("N_Clusters (K-Means)", self.config.get('static_analysis', {}).get('n_clusters', 'N/A')),
            ("DBSCAN eps", self.config.get('classification', {}).get('dbscan', {}).get('eps', 'N/A')),
            ("DBSCAN min_samples", self.config.get('classification', {}).get('dbscan', {}).get('min_samples', 'N/A')),
            ("Feature Selection Mode", self.config.get('feature_selection', {}).get('mode', 'N/A')),
            ("Feature Preset", self.config.get('feature_selection', {}).get('preset', 'N/A')),
            ("PCA Enabled", self.config.get('pca', {}).get('enabled', False)),
            ("Scoring Enabled", self.config.get('scoring', {}).get('enabled', False)),
            ("Validation Enabled", self.config.get('validation', {}).get('enabled', False)),
        ]

        for param, value in params:
            ws[f'A{row}'] = param
            ws[f'B{row}'] = str(value)
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

        row += 1

        # ===== SAMPLE STATISTICS =====
        ws[f'A{row}'] = "SAMPLE STATISTICS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:B{row}')
        row += 1

        # Calculate statistics
        n_companies = df['gvkey'].nunique()
        n_algorithms = df['algorithm'].nunique()

        stats = [
            ("Total Companies", n_companies),
            ("Algorithms Compared", n_algorithms),
            ("Total Observations", len(df)),
            ("Outliers Detected", df['is_outlier'].sum() if 'is_outlier' in df.columns else 'N/A'),
            ("Outlier Rate (%)", f"{df['is_outlier'].mean()*100:.1f}%" if 'is_outlier' in df.columns else 'N/A'),
        ]

        # Add score statistics
        if 'overall_score' in df.columns:
            stats.extend([
                ("Avg Overall Score", f"{df['overall_score'].mean():.1f}"),
                ("Score Std Dev", f"{df['overall_score'].std():.1f}"),
            ])

        for stat, value in stats:
            ws[f'A{row}'] = stat
            ws[f'B{row}'] = str(value)
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

        row += 2

        # ===== SECTOR DISTRIBUTION =====
        ws[f'A{row}'] = "CLUSTER DISTRIBUTION BY SECTOR"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:D{row}')
        row += 1

        if 'gsector' in df.columns:
            # Create contingency table
            sector_cluster = pd.crosstab(df['gsector'], df['cluster'], margins=True)

            # Write to sheet
            for r_idx, row_data in enumerate(dataframe_to_rows(sector_cluster, index=True, header=True)):
                if r_idx == 0:
                    continue  # Skip first empty row
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 1:  # Header
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                row += 1

            # Add bar chart
            chart = BarChart()
            chart.title = "Companies per Sector"
            chart.x_axis.title = "Sector"
            chart.y_axis.title = "Count"

            # Data for chart (exclude margins)
            data = Reference(ws, min_col=2, max_col=sector_cluster.shape[1],
                           min_row=row-sector_cluster.shape[0]-1, max_row=row-2)
            cats = Reference(ws, min_col=1, min_row=row-sector_cluster.shape[0], max_row=row-2)

            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.height = 10
            chart.width = 20

            ws.add_chart(chart, f'F3')

        # Column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20

        logger.info("  ✓ Section 0 sheet created")
