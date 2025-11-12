"""
Section 3 Writer - Drivers/Feature Importance Analysis

Creates Section 3: Treiber (3 sheets) showing:
- 3a: Cluster profiles and feature importance tables
- 3b: Feature importance charts and correlation heatmaps
- 3c: Sector-specific feature analysis

Addresses research question: Welche Finanzkennzahlen bestimmen die Cluster-Zugehörigkeit?
"""

import pandas as pd
import logging
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from .base_section_writer import BaseSectionWriter

logger = logging.getLogger(__name__)


class Section3DriversWriter(BaseSectionWriter):
    """Writer for Section 3: Drivers/Feature Importance Analysis"""

    def __init__(self, config, colors=None, score_columns=None, market='germany',
                 feature_analyzer=None, cluster_namer=None):
        """
        Initialize drivers writer

        Args:
            config: Configuration dictionary
            colors: Color scheme dictionary
            score_columns: List of score column names
            market: Market name for PNG paths
            feature_analyzer: FeatureImportance analyzer instance
            cluster_namer: ClusterNamer instance
        """
        super().__init__(config, colors, score_columns)
        self.market = market
        self.feature_analyzer = feature_analyzer
        self.cluster_namer = cluster_namer

    def create_section(self, wb: Workbook, df: pd.DataFrame):
        """
        Create Section 3: Drivers (3 sheets)

        Args:
            wb: Workbook object
            df: Overview dataframe with all algorithm results
        """
        self._create_section_3a_tables(wb, df)
        self._create_section_3b_charts(wb, df)
        self._create_section_3c_sector(wb, df)

    def _create_section_3a_tables(self, wb: Workbook, df: pd.DataFrame):
        """Section 3a: Treiber Tables (Feature importance, cluster profiles)"""
        ws = wb.create_sheet("3a_Treiber_Tabellen")

        # Title
        ws['A1'] = "SECTION 3a: TREIBER - TABLES"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # Interpretation box - Kernfrage 3
        row = self._add_interpretation_box(ws, row, "KERNFRAGE 3: TREIBER", [
            "Welche Finanzkennzahlen bestimmen die Cluster-Zugehörigkeit?",
            "Feature Importance via Random Forest: Supervised Learning zur Identifikation der Treiber",
            "Höhere Importance = Diese Kennzahl ist wichtiger für die Cluster-Trennung",
            "Cluster-Profile zeigen durchschnittliche Werte pro Cluster für jede Kennzahl"
        ], merge_cols=8)
        row += 1

        # ===== CLUSTER PROFILES (MEAN FEATURE VALUES) =====
        ws[f'A{row}'] = "CLUSTER PROFILES (MEAN FEATURE VALUES)"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Get feature columns (financial ratios)
        feature_candidates = ['roa', 'roe', 'ebit_margin', 'gross_margin', 'debt_to_equity',
                            'current_ratio', 'asset_turnover', 'fcf_margin', 'revenue_growth']
        feature_cols = [f for f in feature_candidates if f in df.columns]

        if len(feature_cols) == 0:
            ws[f'A{row}'] = "No feature columns available"
            logger.warning("  ⚠ No feature columns for Section 3a")
            return

        # Calculate cluster profiles for each algorithm
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()
            algo_df = algo_df[algo_df['cluster'] >= 0]

            if len(algo_df) == 0:
                continue

            ws[f'A{row}'] = f"{algo.upper()} - CLUSTER PROFILES"
            ws[f'A{row}'].font = Font(bold=True)
            ws[f'A{row}'].fill = PatternFill(start_color=self.colors[algo], fill_type='solid')
            ws.merge_cells(f'A{row}:H{row}')
            row += 1

            # Calculate mean per cluster
            cluster_profiles = algo_df.groupby('cluster')[feature_cols].mean()

            # Write profiles
            start_row = row
            for r_idx, row_data in enumerate(dataframe_to_rows(cluster_profiles, index=True, header=True)):
                if r_idx == 0:
                    continue
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 1:  # Header
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.00'
                row += 1

            # Add heatmap conditional formatting
            if len(cluster_profiles) > 0:
                max_col = len(feature_cols) + 1
                ws.conditional_formatting.add(
                    f'B{start_row+1}:{chr(64+max_col)}{row-1}',
                    ColorScaleRule(
                        start_type='min', start_color='FFC7CE',
                        mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                        end_type='max', end_color='C6EFCE'
                    )
                )

                # Add color legend
                ws[f'A{row}'] = "Color Legend: 🔴 Red = Low values | 🟡 Yellow = Medium | 🟢 Green = High values"
                ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
                ws.merge_cells(f'A{row}:H{row}')

            row += 1

        row += 2

        # ===== CLUSTER NAMES (if available) =====
        ws[f'A{row}'] = "CLUSTER NAMES (BASED ON DOMINANT FEATURES)"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:D{row}')
        row += 1

        # Try to generate cluster names
        cluster_names_data = []
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()
            algo_df = algo_df[algo_df['cluster'] >= 0]

            if len(algo_df) == 0:
                continue

            # Calculate cluster profiles
            cluster_profiles = algo_df.groupby('cluster')[feature_cols].mean()

            if len(cluster_profiles) > 0 and self.cluster_namer:
                try:
                    # Generate names
                    names = self.cluster_namer.generate_names(
                        profiles=cluster_profiles,
                        features=feature_cols,
                        style='hybrid',
                        top_n=2
                    )

                    for cluster_id, name in names.items():
                        cluster_names_data.append({
                            'Algorithm': algo,
                            'Cluster': cluster_id,
                            'Name': name
                        })
                except Exception as e:
                    logger.warning(f"  ⚠ Could not generate names for {algo}: {e}")

        if cluster_names_data:
            names_df = pd.DataFrame(cluster_names_data)

            # Write to sheet
            for r_idx, row_data in enumerate(dataframe_to_rows(names_df, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                row += 1
        else:
            ws[f'A{row}'] = "Cluster names not available"
            row += 1

        # Column widths
        for col in range(1, 9):
            ws.column_dimensions[chr(64+col)].width = 18

        logger.info("  ✓ Section 3a sheet created")

    def _create_section_3b_charts(self, wb: Workbook, df: pd.DataFrame):
        """Section 3b: Treiber Charts (Feature importance bar charts)"""
        ws = wb.create_sheet("3b_Treiber_Charts")

        # Title
        ws['A1'] = "SECTION 3b: TREIBER - CHARTS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # Get feature columns
        feature_candidates = ['roa', 'roe', 'ebit_margin', 'gross_margin', 'debt_to_equity',
                            'current_ratio', 'asset_turnover', 'fcf_margin', 'revenue_growth']
        feature_cols = [f for f in feature_candidates if f in df.columns]

        if len(feature_cols) == 0:
            ws[f'A{row}'] = "No feature columns available"
            logger.warning("  ⚠ No feature columns for Section 3b")
            return

        # ===== FEATURE IMPORTANCE (Random Forest) =====
        ws[f'A{row}'] = "FEATURE IMPORTANCE (RANDOM FOREST)"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:E{row}')
        row += 1

        # Calculate feature importance for each algorithm
        all_importance = []
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()

            if 'cluster' not in algo_df.columns or len(algo_df) == 0:
                continue

            if self.feature_analyzer:
                try:
                    importance_df = self.feature_analyzer.compute_importance(
                        df=algo_df,
                        feature_cols=feature_cols,
                        cluster_col='cluster',
                        algorithm_name=algo
                    )

                    if importance_df is not None:
                        all_importance.append(importance_df)
                except Exception as e:
                    logger.warning(f"  ⚠ Could not compute importance for {algo}: {e}")

        if all_importance:
            # Combine all importance scores
            combined_importance = pd.concat(all_importance)

            # Average across algorithms
            avg_importance = combined_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)

            # Create data for sheet
            importance_data = pd.DataFrame({
                'Feature': avg_importance.index,
                'Avg_Importance': avg_importance.values
            })

            # Write to sheet
            start_row = row
            ws[f'A{row}'] = "Feature"
            ws[f'B{row}'] = "Avg Importance"
            for col in ['A', 'B']:
                ws[f'{col}{row}'].font = Font(bold=True)
                ws[f'{col}{row}'].fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
            row += 1

            for _, row_data in importance_data.iterrows():
                ws[f'A{row}'] = row_data['Feature']
                ws[f'B{row}'] = row_data['Avg_Importance']
                ws[f'B{row}'].number_format = '0.0000'
                row += 1

            # Add bar chart
            chart = BarChart()
            chart.title = "Feature Importance (Avg across Algorithms)"
            chart.x_axis.title = "Feature"
            chart.y_axis.title = "Importance"
            chart.type = "col"

            data = Reference(ws, min_col=2, min_row=start_row, max_row=row-1)
            cats = Reference(ws, min_col=1, min_row=start_row+1, max_row=row-1)

            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.height = 15
            chart.width = 20

            ws.add_chart(chart, 'D3')
        else:
            ws[f'A{row}'] = "Feature importance could not be calculated"
            row += 1

        row += 2

        # ===== EMBED CORRELATION HEATMAP PNGs =====
        ws[f'A{row}'] = "FEATURE CORRELATION HEATMAPS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:E{row}')
        row += 1

        # Embed PCA analysis and feature importance plots
        base_path_combined = Path(f'output/{self.market}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.market}/03_comparisons')
        plots_to_embed = [
            (base_path_combined / '5_pca_analysis/plots/scree_plot.png', 'A', 0.35),
            (base_path_combined / '5_pca_analysis/plots/biplot_pc1_pc2.png', 'I', 0.35),
            (base_path_combined / '5_pca_analysis/plots/component_loadings.png', 'A', 0.35),
            (base_path_combined / '5_pca_analysis/plots/cluster_separation.png', 'I', 0.35),
            (base_path_combined / 'plots/pca_clusters.png', 'A', 0.35),
            (base_path_comparisons / 'features/combined_importance_combined.png', 'I', 0.35),
        ]

        embedded_count = 0
        current_row = row
        for i, (plot_path, col_letter, scale) in enumerate(plots_to_embed):
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{current_row}', scale=scale)
                embedded_count += 1
                if (i + 1) % 2 == 0:  # Every 2 plots, new row
                    current_row += 28

        if embedded_count == 0:
            ws[f'A{row}'] = "PCA and feature importance plots not available"
            ws[f'A{row}'].font = Font(italic=True, size=9)
            row += 1
        else:
            row = current_row + 28  # Space for embedded images

        logger.info(f"  ✓ Section 3b sheet created ({embedded_count} plots embedded)")

    def _create_section_3c_sector(self, wb: Workbook, df: pd.DataFrame):
        """Section 3c: Sector-specific Feature Analysis"""
        ws = wb.create_sheet("3c_Treiber_Sektor")

        # Title
        ws['A1'] = "SECTION 3c: SECTOR-SPECIFIC FEATURES"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        if 'gsector' not in df.columns:
            ws['A3'] = "No sector data available"
            logger.warning("  ⚠ No sector data for Section 3c")
            return

        row = 3

        # Get feature columns
        feature_candidates = ['roa', 'roe', 'ebit_margin', 'gross_margin', 'debt_to_equity',
                            'current_ratio', 'asset_turnover', 'fcf_margin', 'revenue_growth']
        feature_cols = [f for f in feature_candidates if f in df.columns]

        if len(feature_cols) == 0:
            ws[f'A{row}'] = "No feature columns available"
            return

        # ===== SECTOR FEATURE PROFILES =====
        ws[f'A{row}'] = "AVERAGE FEATURE VALUES BY SECTOR"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Calculate sector profiles
        sector_profiles = df.groupby('gsector')[feature_cols].mean()

        # Write profiles
        start_row = row
        for r_idx, row_data in enumerate(dataframe_to_rows(sector_profiles, index=True, header=True)):
            if r_idx == 0:
                continue
            for c_idx, value in enumerate(row_data):
                cell = ws.cell(row=row, column=c_idx+1, value=value)
                if r_idx == 1:  # Header
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                elif isinstance(value, float):
                    cell.number_format = '0.00'
            row += 1

        # Add heatmap conditional formatting
        if len(sector_profiles) > 0:
            max_col = len(feature_cols) + 1
            ws.conditional_formatting.add(
                f'B{start_row+1}:{chr(64+max_col)}{row-1}',
                ColorScaleRule(
                    start_type='min', start_color='FFC7CE',
                    mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                    end_type='max', end_color='C6EFCE'
                )
            )

            # Add color legend
            ws[f'A{row}'] = "Color Legend: 🔴 Red = Low values | 🟡 Yellow = Medium | 🟢 Green = High values"
            ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
            ws.merge_cells(f'A{row}:H{row}')

        # Column widths
        for col in range(1, 9):
            ws.column_dimensions[chr(64+col)].width = 15

        # ===== EMBED CLUSTER CHARACTERISTICS & PERFORMANCE VISUALIZATIONS =====
        row += 2
        ws[f'A{row}'] = "CLUSTER CHARACTERISTICS & PERFORMANCE"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed cluster characteristics and top/bottom performers
        base_path_combined = Path(f'output/{self.market}/02_algorithms/kmeans_comparative/combined')
        plots_to_embed = [
            (base_path_combined / 'plots/cluster_characteristics.png', 'A', 0.4),
            (base_path_combined / '4_company_insights/plots/top_bottom_performers.png', 'G', 0.4),
        ]

        embedded_count = 0
        for plot_path, col_letter, scale in plots_to_embed:
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{row}', scale=scale)
                embedded_count += 1

        if embedded_count > 0:
            row += 30  # Space for embedded images

        logger.info(f"  ✓ Section 3c sheet created ({embedded_count} plots embedded)")
