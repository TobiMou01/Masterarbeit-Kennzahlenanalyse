"""
Section 4 Writer - Stability & Context Analysis

Creates Section 4: Stabilität (3 sheets) showing:
- 4a: Stability metrics tables (migration matrices, score consistency)
- 4b: Temporal evolution charts
- 4c: Size-based analysis

Addresses research question: Wie stabil sind Cluster über Zeit und über verschiedene Kontexte?
"""

import pandas as pd
import logging
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from .base_section_writer import BaseSectionWriter

logger = logging.getLogger(__name__)


class Section4StabilityWriter(BaseSectionWriter):
    """Writer for Section 4: Stability & Context Analysis"""

    def create_section(self, wb: Workbook, df: pd.DataFrame):
        """
        Create Section 4: Stability (3 sheets)

        Args:
            wb: Workbook object
            df: Overview dataframe with all algorithm results
        """
        self._create_section_4a_tables(wb, df)
        self._create_section_4b_charts(wb, df)
        self._create_section_4c_size_classes(wb, df)

    def _create_section_4a_tables(self, wb: Workbook, df: pd.DataFrame):
        """Section 4a: Stabilität Tables (Migration matrices, stability metrics)"""
        ws = wb.create_sheet("4a_Stabilität_Tabellen")

        # Title
        ws['A1'] = "SECTION 4a: STABILITÄT - TABLES"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # Interpretation box - Kernfrage 4
        row = self._add_interpretation_box(ws, row, "KERNFRAGE 4: STABILITÄT & KONTEXT", [
            "Wie stabil sind Cluster über Zeit und über verschiedene Kontexte (Größenklassen)?",
            "Score StdDev < 10 = Hohe Stabilität innerhalb des Clusters",
            "Migrationsmatrizen zeigen Cluster-Wechsel über Zeit (wenn zeitliche Daten vorliegen)",
            "Größenklassen-Analyse: Gelten Cluster auch für kleine, mittlere und große Unternehmen?"
        ], merge_cols=8)
        row += 1

        # ===== CLUSTER STABILITY METRICS =====
        ws[f'A{row}'] = "CLUSTER STABILITY METRICS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Calculate stability metrics per cluster
        stability_data = []
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()
            algo_df = algo_df[algo_df['cluster'] >= 0]

            if len(algo_df) == 0:
                continue

            # Calculate metrics per cluster
            for cluster_id in sorted(algo_df['cluster'].unique()):
                cluster_df = algo_df[algo_df['cluster'] == cluster_id]

                metrics = {
                    'Algorithm': algo,
                    'Cluster ID': cluster_id,
                    'Size (N)': len(cluster_df),
                    'Size (%)': f"{len(cluster_df)/len(algo_df)*100:.1f}%"
                }

                # Score stability (std dev)
                if 'overall_score' in cluster_df.columns:
                    metrics['Score Mean (Ø)'] = cluster_df['overall_score'].mean()
                    metrics['Score StdDev (σ)'] = cluster_df['overall_score'].std()
                    metrics['Stability Level'] = 'High' if metrics['Score StdDev (σ)'] < 10 else ('Medium' if metrics['Score StdDev (σ)'] < 20 else 'Low')

                stability_data.append(metrics)

        if stability_data:
            stability_df = pd.DataFrame(stability_data)

            # Write to sheet
            for r_idx, row_data in enumerate(dataframe_to_rows(stability_df, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.00'
                row += 1
        else:
            ws[f'A{row}'] = "No stability data available"
            row += 1

        row += 2

        # ===== TEMPORAL ANALYSIS PLACEHOLDER =====
        ws[f'A{row}'] = "TEMPORAL STABILITY (Migration Matrices)"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        ws[f'A{row}'] = "Temporal data not yet available"
        ws[f'A{row}'].font = Font(italic=True)
        row += 1

        ws[f'A{row}'] = "Future implementation will include:"
        row += 1
        ws[f'B{row}'] = "- Migration matrices (cluster transitions over time)"
        row += 1
        ws[f'B{row}'] = "- Stability scores (% companies remaining in same cluster)"
        row += 1
        ws[f'B{row}'] = "- Temporal evolution of cluster characteristics"
        row += 1

        # Column widths
        for col in range(1, 9):
            ws.column_dimensions[chr(64+col)].width = 18

        # ===== EMBED TEMPORAL STABILITY VISUALIZATIONS =====
        from pathlib import Path
        row += 2
        ws[f'A{row}'] = "TEMPORAL STABILITY VISUALIZATIONS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed migration heatmaps and yearly stability plots
        base_path_comparisons = Path(f'output/{self.config.get("market", "germany")}/03_comparisons')
        plots_to_embed = [
            (base_path_comparisons / 'temporal/kmeans_migration_heatmap.png', 'A', 0.35),
            (base_path_comparisons / 'temporal/kmeans_yearly_stability.png', 'I', 0.35),
            (base_path_comparisons / 'temporal/hierarchical_migration_heatmap.png', 'A', 0.35),
            (base_path_comparisons / 'temporal/dbscan_migration_heatmap.png', 'I', 0.35),
        ]

        embedded_count = 0
        current_row = row
        for i, (plot_path, col_letter, scale) in enumerate(plots_to_embed):
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{current_row}', scale=scale)
                embedded_count += 1
                if (i + 1) % 2 == 0:  # Every 2 plots, new row
                    current_row += 28

        if embedded_count > 0:
            row = current_row + 28  # Space for embedded images

        logger.info(f"  ✓ Section 4a sheet created ({embedded_count} plots embedded)")

    def _create_section_4b_charts(self, wb: Workbook, df: pd.DataFrame):
        """Section 4b: Stabilität Charts (Temporal evolution)"""
        ws = wb.create_sheet("4b_Stabilität_Charts")

        # Title
        ws['A1'] = "SECTION 4b: STABILITÄT - CHARTS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # ===== SCORE STABILITY DISTRIBUTION =====
        ws[f'A{row}'] = "SCORE STABILITY BY CLUSTER"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Calculate score standard deviation per cluster
        if 'overall_score' in df.columns:
            stability_by_cluster = []
            for algo in df['algorithm'].unique():
                algo_df = df[df['algorithm'] == algo].copy()
                algo_df = algo_df[algo_df['cluster'] >= 0]

                if len(algo_df) > 0:
                    cluster_stability = algo_df.groupby('cluster')['overall_score'].std().reset_index()
                    cluster_stability.columns = ['Cluster', f'{algo}_StdDev']
                    stability_by_cluster.append(cluster_stability)

            if stability_by_cluster:
                # Merge all stability data
                combined_stability = stability_by_cluster[0]
                for i in range(1, len(stability_by_cluster)):
                    combined_stability = combined_stability.merge(
                        stability_by_cluster[i],
                        on='Cluster',
                        how='outer'
                    )

                # Write data
                start_row = row
                for r_idx, row_data in enumerate(dataframe_to_rows(combined_stability, index=False, header=True)):
                    for c_idx, value in enumerate(row_data):
                        cell = ws.cell(row=row, column=c_idx+1, value=value)
                        if r_idx == 0:
                            cell.font = Font(bold=True)
                            cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                        elif isinstance(value, float):
                            cell.number_format = '0.00'
                    row += 1

                # Add bar chart
                try:
                    chart = BarChart()
                    chart.title = "Score Stability (StdDev) by Cluster"
                    chart.x_axis.title = "Cluster"
                    chart.y_axis.title = "Standard Deviation"
                    chart.type = "col"
                    chart.grouping = "clustered"

                    data = Reference(ws, min_col=2, max_col=len(combined_stability.columns),
                                   min_row=start_row, max_row=row-1)
                    cats = Reference(ws, min_col=1, min_row=start_row+1, max_row=row-1)

                    chart.add_data(data, titles_from_data=True)
                    chart.set_categories(cats)
                    chart.height = 15
                    chart.width = 20

                    ws.add_chart(chart, 'A' + str(row + 2))
                except Exception as e:
                    logger.warning(f"  ⚠ Could not create stability chart: {e}")
        else:
            ws[f'A{row}'] = "No score data available"

        row += 20

        # ===== TEMPORAL CHARTS PLACEHOLDER =====
        ws[f'A{row}'] = "TEMPORAL EVOLUTION CHARTS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        ws[f'A{row}'] = "Temporal charts will be added when time-series data is available"
        ws[f'A{row}'].font = Font(italic=True)
        row += 2

        # ===== EMBED SCORE EVOLUTION & STABILITY COMPARISON =====
        from pathlib import Path
        ws[f'A{row}'] = "SCORE EVOLUTION & STABILITY COMPARISON"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed evolution scatter and algorithm stability comparison
        base_path_combined = Path(f'output/{self.config.get("market", "germany")}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.config.get("market", "germany")}/03_comparisons')
        plots_to_embed = [
            (base_path_combined / '1_cluster_quality/scores/evolution/evolution_scatter.png', 'A', 0.4),
            (base_path_comparisons / 'temporal/algorithm_stability_comparison.png', 'I', 0.4),
        ]

        embedded_count = 0
        for plot_path, col_letter, scale in plots_to_embed:
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{row}', scale=scale)
                embedded_count += 1

        if embedded_count > 0:
            row += 30  # Space for embedded images

        logger.info(f"  ✓ Section 4b sheet created ({embedded_count} plots embedded)")

    def _create_section_4c_size_classes(self, wb: Workbook, df: pd.DataFrame):
        """Section 4c: Size-Based Analysis"""
        ws = wb.create_sheet("4c_Stabilität_Größenklassen")

        # Title
        ws['A1'] = "SECTION 4c: SIZE-BASED ANALYSIS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        if 'size_category' not in df.columns:
            ws['A3'] = "No size category data available"
            logger.warning("  ⚠ No size category data for Section 4c")
            return

        row = 3

        # ===== CLUSTER DISTRIBUTION BY SIZE =====
        ws[f'A{row}'] = "CLUSTER DISTRIBUTION BY COMPANY SIZE"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Create size × cluster contingency table
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()
            algo_df = algo_df[algo_df['cluster'] >= 0]

            if len(algo_df) > 0:
                ws[f'A{row}'] = f"{algo.upper()} - Size × Cluster Distribution"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'A{row}'].fill = PatternFill(start_color=self.colors[algo], fill_type='solid')
                ws.merge_cells(f'A{row}:F{row}')
                row += 1

                # Create contingency table
                size_cluster = pd.crosstab(algo_df['size_category'], algo_df['cluster'], margins=True)

                # Write table
                start_row = row
                for r_idx, row_data in enumerate(dataframe_to_rows(size_cluster, index=True, header=True)):
                    if r_idx == 0:
                        continue
                    for c_idx, value in enumerate(row_data):
                        cell = ws.cell(row=row, column=c_idx+1, value=value)
                        if r_idx == 1:
                            cell.font = Font(bold=True)
                            cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    row += 1

                # Add conditional formatting
                if row > start_row + 1:
                    max_col = size_cluster.shape[1] + 1
                    ws.conditional_formatting.add(
                        f'B{start_row+1}:{chr(64+max_col)}{row-1}',
                        ColorScaleRule(
                            start_type='num', start_value=0, start_color='FFFFFF',
                            end_type='max', end_color='4472C4'
                        )
                    )

                    # Add color legend
                    ws[f'A{row}'] = "Color Legend: White = Few companies | Dark Blue = Many companies"
                    ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
                    ws.merge_cells(f'A{row}:F{row}')

                row += 1

        row += 2

        # ===== PERFORMANCE BY SIZE =====
        ws[f'A{row}'] = "PERFORMANCE METRICS BY SIZE CATEGORY"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Calculate metrics by size
        if 'overall_score' in df.columns:
            size_metrics = df.groupby('size_category').agg({
                'overall_score': ['mean', 'std', 'count'],
                'cluster': 'nunique'
            }).reset_index()
            size_metrics.columns = ['Size_Category', 'Avg_Score', 'Score_StdDev', 'N_Companies', 'N_Clusters']

            # Write to sheet
            start_row = row
            for r_idx, row_data in enumerate(dataframe_to_rows(size_metrics, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.00'
                row += 1

            # Add conditional formatting for avg score
            if row > start_row + 1:
                ws.conditional_formatting.add(
                    f'B{start_row+1}:B{row-1}',
                    ColorScaleRule(
                        start_type='min', start_color='FFC7CE',
                        mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                        end_type='max', end_color='C6EFCE'
                    )
                )

                # Add color legend
                row += 1
                ws[f'A{row}'] = "Color Legend: 🔴 Red = Low score | 🟡 Yellow = Medium | 🟢 Green = High score"
                ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
                ws.merge_cells(f'A{row}:F{row}')

        # Column widths
        ws.column_dimensions['A'].width = 25
        for col in range(2, 9):
            ws.column_dimensions[chr(64+col)].width = 15

        # ===== EMBED SIZE CATEGORY VISUALIZATION =====
        from pathlib import Path
        row += 2
        ws[f'A{row}'] = "SIZE CATEGORY VISUALIZATIONS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed size category contingency plot
        base_path_combined = Path(f'output/{self.config.get("market", "germany")}/02_algorithms/kmeans_comparative/combined')
        plot_path = base_path_combined / '3_external_validation/plots/contingency_size_category.png'

        if plot_path.exists():
            self._embed_png(ws, plot_path, f'A{row}', scale=0.5)
            row += 30  # Space for embedded image
            embedded_count = 1
        else:
            embedded_count = 0

        logger.info(f"  ✓ Section 4c sheet created ({embedded_count} plot embedded)")
