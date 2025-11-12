"""
Section 2 Writer - Congruence Analysis

Creates Section 2: Kongruenz (3 sheets) showing:
- 2a: Cramér's V and ARI tables
- 2b: Contingency heatmaps and scatter plots
- 2c: Algorithm agreement analysis

Addresses research question: Wie stark stimmen kennzahlenbasierte Cluster
mit bestehenden Klassifikationen überein?
"""

import pandas as pd
import logging
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.marker import Marker
from openpyxl.formatting.rule import ColorScaleRule
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from .base_section_writer import BaseSectionWriter

logger = logging.getLogger(__name__)


class Section2CongruenceWriter(BaseSectionWriter):
    """Writer for Section 2: Congruence Analysis"""

    def create_section(self, wb: Workbook, df: pd.DataFrame):
        """
        Create Section 2: Congruence (3 sheets)

        Args:
            wb: Workbook object
            df: Overview dataframe with all algorithm results
        """
        self._create_section_2a_tables(wb, df)
        self._create_section_2b_charts(wb, df)
        self._create_section_2c_algorithms(wb, df)

    def _create_section_2a_tables(self, wb: Workbook, df: pd.DataFrame):
        """Section 2a: Kongruenz Tables (Cramér's V, ARI, Chi-Square)"""
        ws = wb.create_sheet("2a_Kongruenz_Tabellen")

        # Title
        ws['A1'] = "SECTION 2a: KONGRUENZ - TABLES"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        if 'gsector' not in df.columns:
            ws['A3'] = "No GICS sector data available for congruence analysis"
            logger.warning("  ⚠ No GICS sector data for Section 2a")
            return

        row = 3

        # Interpretation box - Kernfrage 2
        row = self._add_interpretation_box(ws, row, "KERNFRAGE 2: KONGRUENZ", [
            "Wie stark stimmen kennzahlenbasierte Cluster mit bestehenden Klassifikationen überein?",
            "Cramér's V misst Korrelation zwischen Cluster und GICS-Sektor (0 = keine, 1 = perfekt)",
            "Niedrige Werte (<0.3) erwünscht: Cluster bieten neue Perspektive jenseits von Branchen",
            "ARI (Adjusted Rand Index) misst Übereinstimmung zwischen Algorithmen (0-1)"
        ], merge_cols=8)
        row += 1

        # ===== CRAMÉR'S V: CLUSTER VS GICS SECTOR =====
        ws[f'A{row}'] = "CRAMÉR'S V: CLUSTER VS GICS SECTOR"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:E{row}')
        row += 1

        # Calculate Cramér's V for each algorithm
        cramers_results = []
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()
            algo_df = algo_df[algo_df['cluster'] >= 0]  # Remove noise

            if len(algo_df) > 0 and 'gsector' in algo_df.columns:
                # Create contingency table
                contingency = pd.crosstab(algo_df['cluster'], algo_df['gsector'])

                # Validate contingency table
                if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    logger.warning(f"  ⚠ {algo}: Contingency table too small (shape={contingency.shape}), skipping")
                    continue

                # Check if table has any data
                if contingency.sum().sum() == 0:
                    logger.warning(f"  ⚠ {algo}: Contingency table is empty, skipping")
                    continue

                try:
                    # Calculate Cramér's V
                    chi2, p_value, dof, _ = chi2_contingency(contingency.values)
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
                    cramers_v = (chi2 / (n * min_dim)) ** 0.5

                    # Interpretation
                    if cramers_v < 0.2:
                        interpretation = "Very weak correlation (good!)"
                    elif cramers_v < 0.3:
                        interpretation = "Weak correlation"
                    elif cramers_v < 0.5:
                        interpretation = "Moderate correlation"
                    else:
                        interpretation = "Strong correlation"

                    cramers_results.append({
                        'Algorithm': algo,
                        "Cramér's V (0-1)": cramers_v,
                        'Chi² Statistic': chi2,
                        'p-value (Sig.)': p_value,
                        'Interpretation': interpretation
                    })
                except Exception as e:
                    logger.warning(f"  ⚠ {algo}: Could not calculate Cramér's V: {e}")

        if cramers_results:
            cramers_df = pd.DataFrame(cramers_results)

            # Write to sheet
            start_row = row
            for r_idx, row_data in enumerate(dataframe_to_rows(cramers_df, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:  # Header
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.0000'
                row += 1

            # Add conditional formatting
            cramers_col = 'B'
            ws.conditional_formatting.add(
                f'{cramers_col}{start_row+1}:{cramers_col}{row-1}',
                ColorScaleRule(
                    start_type='num', start_value=0, start_color='C6EFCE',
                    mid_type='num', mid_value=0.3, mid_color='FFEB9C',
                    end_type='num', end_value=0.6, end_color='FFC7CE'
                )
            )

            # Add color legend
            row += 1
            ws[f'A{row}'] = "Color Legend: 🟢 Green = Weak correlation (good) | 🟡 Yellow = Moderate | 🔴 Red = Strong (clusters = sectors)"
            ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
            ws.merge_cells(f'A{row}:E{row}')

        row += 2

        # ===== ADJUSTED RAND INDEX: ALGORITHM COMPARISON =====
        ws[f'A{row}'] = "ADJUSTED RAND INDEX: ALGORITHM COMPARISON"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:E{row}')
        row += 1

        # Calculate ARI between algorithms
        algorithms = df['algorithm'].unique().tolist()
        ari_matrix = []

        for algo1 in algorithms:
            ari_row = {'Algorithm': algo1}
            for algo2 in algorithms:
                if algo1 == algo2:
                    ari_row[algo2] = 1.0
                else:
                    # Merge clusters from both algorithms
                    df1 = df[df['algorithm'] == algo1][['gvkey', 'cluster']].copy()
                    df2 = df[df['algorithm'] == algo2][['gvkey', 'cluster']].copy()

                    merged = df1.merge(df2, on='gvkey', suffixes=('_1', '_2'))
                    merged = merged[(merged['cluster_1'] >= 0) & (merged['cluster_2'] >= 0)]

                    if len(merged) > 0:
                        try:
                            ari = adjusted_rand_score(merged['cluster_1'], merged['cluster_2'])
                            ari_row[algo2] = ari
                        except Exception as e:
                            logger.warning(f"  ⚠ Could not calculate ARI for {algo1} vs {algo2}: {e}")
                            ari_row[algo2] = 0.0
                    else:
                        ari_row[algo2] = 0.0

            ari_matrix.append(ari_row)

        if ari_matrix:
            ari_df = pd.DataFrame(ari_matrix)

            # Write to sheet
            start_row = row
            for r_idx, row_data in enumerate(dataframe_to_rows(ari_df, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:  # Header
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.0000'
                row += 1

            # Add conditional formatting (higher ARI = better agreement)
            for col_idx in range(2, len(algorithms)+2):
                col_letter = chr(64+col_idx)
                ws.conditional_formatting.add(
                    f'{col_letter}{start_row+1}:{col_letter}{row-1}',
                    ColorScaleRule(
                        start_type='num', start_value=0, start_color='FFC7CE',
                        mid_type='num', mid_value=0.5, mid_color='FFEB9C',
                        end_type='num', end_value=1, end_color='C6EFCE'
                    )
                )

            # Add color legend
            row += 1
            ws[f'A{row}'] = "Color Legend: 🟢 Green = High agreement | 🟡 Yellow = Moderate | 🔴 Red = Low agreement between algorithms"
            ws[f'A{row}'].font = Font(size=9, italic=True, color='666666')
            ws.merge_cells(f'A{row}:E{row}')

        # Column widths
        for col in range(1, 9):
            ws.column_dimensions[chr(64+col)].width = 18

        # ===== EMBED CONGRUENCE VISUALIZATIONS =====
        from pathlib import Path
        row += 2
        ws[f'A{row}'] = "CONGRUENCE VISUALIZATIONS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed GICS contingency and cramers_v plots
        base_path_combined = Path(f'output/{self.config.get("market", "germany")}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.config.get("market", "germany")}/03_comparisons')
        plots_to_embed = [
            (base_path_combined / '3_external_validation/plots/contingency_gics_sector.png', 'A', 0.35),
            (base_path_combined / '3_external_validation/plots/cramers_v_comparison.png', 'I', 0.35),
            (base_path_comparisons / 'gics_tables/kmeans_vs_gsector.png', 'A', 0.35),
        ]

        embedded_count = 0
        current_row = row
        for plot_path, col_letter, scale in plots_to_embed:
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{current_row}', scale=scale)
                embedded_count += 1
                if embedded_count % 2 == 0:  # Every 2 plots, new row
                    current_row += 28

        if embedded_count > 0:
            row = current_row + 28  # Space for embedded images

        logger.info(f"  ✓ Section 2a sheet created ({embedded_count} plots embedded)")

    def _create_section_2b_charts(self, wb: Workbook, df: pd.DataFrame):
        """Section 2b: Kongruenz Charts (Heatmaps, Scatter plots)"""
        ws = wb.create_sheet("2b_Kongruenz_Charts")

        # Title
        ws['A1'] = "SECTION 2b: KONGRUENZ - CHARTS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # ===== CONTINGENCY TABLE: CLUSTER VS GICS SECTOR =====
        if 'gsector' in df.columns:
            ws[f'A{row}'] = "CONTINGENCY TABLE: CLUSTERS VS GICS SECTORS"
            ws[f'A{row}'].font = Font(size=12, bold=True)
            ws.merge_cells(f'A{row}:F{row}')
            row += 1

            # Create contingency table for each algorithm
            for algo in df['algorithm'].unique():
                algo_df = df[df['algorithm'] == algo].copy()
                algo_df = algo_df[algo_df['cluster'] >= 0]

                if len(algo_df) > 0:
                    # Create contingency table
                    contingency = pd.crosstab(algo_df['cluster'], algo_df['gsector'])

                    # Skip if contingency is too small
                    if contingency.size == 0 or contingency.shape[0] < 1 or contingency.shape[1] < 1:
                        logger.warning(f"  ⚠ {algo}: Contingency table too small for 2b, skipping")
                        continue

                    ws[f'A{row}'] = f"{algo.upper()}"
                    ws[f'A{row}'].font = Font(bold=True)
                    ws[f'A{row}'].fill = PatternFill(start_color=self.colors[algo], fill_type='solid')
                    row += 1

                    # Write table
                    start_row = row
                    for r_idx, row_data in enumerate(dataframe_to_rows(contingency, index=True, header=True)):
                        if r_idx == 0:
                            continue
                        for c_idx, value in enumerate(row_data):
                            cell = ws.cell(row=row, column=c_idx+1, value=value)
                            if r_idx == 1:  # Header
                                cell.font = Font(bold=True)
                                cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                        row += 1

                    # Add heatmap-style conditional formatting
                    if row > start_row + 1:  # Only if we have data rows
                        max_col = contingency.shape[1] + 1
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

        # ===== ALGORITHM AGREEMENT SCATTER PLOT DATA =====
        ws[f'A{row}'] = "ALGORITHM SCORE COMPARISON (for scatter plots)"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Prepare data for scatter plots
        algorithms = df['algorithm'].unique().tolist()
        if len(algorithms) >= 2 and 'overall_score' in df.columns:
            algo1, algo2 = algorithms[0], algorithms[1]

            # Pivot data
            df_pivot = df.pivot_table(
                index='gvkey',
                columns='algorithm',
                values='overall_score',
                aggfunc='first'
            ).reset_index()

            if algo1 in df_pivot.columns and algo2 in df_pivot.columns:
                scatter_data = df_pivot[['gvkey', algo1, algo2]].dropna()

                # Write scatter data
                ws[f'A{row}'] = "Company"
                ws[f'B{row}'] = f"{algo1} Score"
                ws[f'C{row}'] = f"{algo2} Score"
                for col in ['A', 'B', 'C']:
                    ws[f'{col}{row}'].font = Font(bold=True)
                    ws[f'{col}{row}'].fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                row += 1

                # Write data (limit to 100 rows for chart performance)
                for _, row_data in scatter_data.head(100).iterrows():
                    ws[f'A{row}'] = row_data['gvkey']
                    ws[f'B{row}'] = row_data[algo1]
                    ws[f'C{row}'] = row_data[algo2]
                    row += 1

                # Create scatter chart
                chart = ScatterChart()
                chart.title = f"Score Comparison: {algo1} vs {algo2}"
                chart.x_axis.title = f"{algo1} Score"
                chart.y_axis.title = f"{algo2} Score"

                xvalues = Reference(ws, min_col=2, min_row=row-len(scatter_data.head(100)), max_row=row-1)
                yvalues = Reference(ws, min_col=3, min_row=row-len(scatter_data.head(100)), max_row=row-1)

                series = Series(yvalues, xvalues, title=f"{algo1} vs {algo2}")
                series.marker = Marker('circle')
                series.marker.size = 5
                series.graphicalProperties.line.noFill = True
                chart.series.append(series)

                chart.height = 15
                chart.width = 20

                ws.add_chart(chart, f'F{row-100}')

        # ===== EMBED GICS SUMMARY VISUALIZATIONS =====
        from pathlib import Path
        row += 2
        ws[f'A{row}'] = "GICS SECTOR CONGRUENCE VISUALIZATIONS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Embed GICS summary and size category plots
        base_path_combined = Path(f'output/{self.config.get("market", "germany")}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.config.get("market", "germany")}/03_comparisons')
        plots_to_embed = [
            (base_path_comparisons / 'gics/summary_gics_combined.png', 'A', 0.4),
            (base_path_combined / '3_external_validation/plots/contingency_size_category.png', 'I', 0.4),
        ]

        embedded_count = 0
        for plot_path, col_letter, scale in plots_to_embed:
            if plot_path.exists():
                self._embed_png(ws, plot_path, f'{col_letter}{row}', scale=scale)
                embedded_count += 1

        if embedded_count > 0:
            row += 30  # Space for embedded images

        logger.info(f"  ✓ Section 2b sheet created ({embedded_count} plots embedded)")

    def _create_section_2c_algorithms(self, wb: Workbook, df: pd.DataFrame):
        """Section 2c: Algorithm Comparison Metrics"""
        ws = wb.create_sheet("2c_Kongruenz_Algorithmen")

        # Title
        ws['A1'] = "SECTION 2c: ALGORITHM COMPARISON"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:G1')

        row = 3

        # ===== ALGORITHM AGREEMENT SUMMARY =====
        ws[f'A{row}'] = "ALGORITHM AGREEMENT SUMMARY"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:D{row}')
        row += 1

        if 'algorithm_agreement' in df.columns:
            # Calculate agreement stats per company
            agreement_stats = df.groupby('gvkey').agg({
                'algorithm_agreement': 'first',
                'unique_clusters': 'first',
                'conm': 'first'
            }).reset_index()

            agreement_stats = agreement_stats.sort_values('algorithm_agreement')

            # Summary
            summary = [
                ("Total Companies", len(agreement_stats)),
                ("Perfect Agreement (100%)", len(agreement_stats[agreement_stats['algorithm_agreement'] == 100])),
                ("High Agreement (>80%)", len(agreement_stats[agreement_stats['algorithm_agreement'] > 80])),
                ("Moderate Agreement (50-80%)", len(agreement_stats[(agreement_stats['algorithm_agreement'] >= 50) & (agreement_stats['algorithm_agreement'] <= 80)])),
                ("Low Agreement (<50%)", len(agreement_stats[agreement_stats['algorithm_agreement'] < 50])),
                ("Average Agreement", f"{agreement_stats['algorithm_agreement'].mean():.1f}%"),
            ]

            for label, value in summary:
                ws[f'A{row}'] = label
                ws[f'B{row}'] = value
                ws[f'A{row}'].font = Font(bold=True)
                row += 1

            row += 2

            # ===== LOW AGREEMENT COMPANIES =====
            ws[f'A{row}'] = "COMPANIES WITH LOW ALGORITHM AGREEMENT (<50%)"
            ws[f'A{row}'].font = Font(size=12, bold=True)
            ws.merge_cells(f'A{row}:E{row}')
            row += 1

            low_agreement = agreement_stats[agreement_stats['algorithm_agreement'] < 50]

            if len(low_agreement) > 0:
                low_agreement_display = low_agreement[['conm', 'gvkey', 'algorithm_agreement', 'unique_clusters']]

                # Write to sheet
                for r_idx, row_data in enumerate(dataframe_to_rows(low_agreement_display, index=False, header=True)):
                    for c_idx, value in enumerate(row_data):
                        cell = ws.cell(row=row, column=c_idx+1, value=value)
                        if r_idx == 0:
                            cell.font = Font(bold=True)
                            cell.fill = PatternFill(start_color=self.colors['warning'], fill_type='solid')
                        elif isinstance(value, float):
                            cell.number_format = '0.0'
                    row += 1
            else:
                ws[f'A{row}'] = "No companies with low agreement found"
                row += 1

        else:
            ws[f'A{row}'] = "Algorithm agreement metrics not available"
            row += 1

        # Column widths
        ws.column_dimensions['A'].width = 40
        for col in range(2, 8):
            ws.column_dimensions[chr(64+col)].width = 18

        # ===== EMBED ALGORITHM COMPARISON VISUALIZATIONS =====
        from pathlib import Path
        row += 2
        ws[f'A{row}'] = "ALGORITHM COMPARISON VISUALIZATIONS"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:G{row}')
        row += 1

        # Embed algorithm congruence and comparison plots
        base_path_combined = Path(f'output/{self.config.get("market", "germany")}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.config.get("market", "germany")}/03_comparisons')
        plots_to_embed = [
            (base_path_combined / '2_algorithm_congruence/plots/ari_heatmap_robustness.png', 'A', 0.35),
            (base_path_combined / '2_algorithm_congruence/plots/confusion_matrix.png', 'I', 0.35),
            (base_path_comparisons / 'algorithms/algorithm_overlap_combined.png', 'A', 0.35),
            (base_path_comparisons / 'algorithms/metrics_comparison_combined.png', 'I', 0.35),
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

        logger.info(f"  ✓ Section 2c sheet created ({embedded_count} plots embedded)")
