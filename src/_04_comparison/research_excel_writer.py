"""
Research Excel Writer - Structured Analysis for Master Thesis

Creates research-oriented Excel file aligned with 4 core research questions:
1. Homogenität (Homogeneity) - Internal cluster quality
2. Kongruenz (Congruence) - Agreement with classifications
3. Treiber (Drivers) - Key financial ratios
4. Stabilität & Kontext (Stability & Context) - Temporal & size effects

Each section has multiple sheets with tables, charts, and embedded visualizations.
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, ScatterChart, Reference, LineChart
from openpyxl.chart.marker import Marker
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ResearchExcelWriter:
    """
    Creates research-oriented Excel file structured around thesis questions

    Structure:
    - Section 0: Config & Overview (1 sheet)
    - Section 1: Homogenität (4 sheets)
    - Section 2: Kongruenz (3 sheets) - TODO
    - Section 3: Treiber (3 sheets) - TODO
    - Section 4: Stabilität (3 sheets) - TODO
    - Appendix: Raw Data (3 sheets) - TODO
    """

    def __init__(self, algorithm_results: Dict, market: str = 'germany', config_path: str = 'config.yaml'):
        """
        Initialize research Excel writer

        Args:
            algorithm_results: Dict with results from all algorithms
            market: Market name
            config_path: Path to config.yaml
        """
        self.algorithm_results = algorithm_results
        self.market = market
        self.config_path = Path(config_path)

        # Load config
        self.config = self._load_config()

        # Score columns
        self.score_columns = [
            'proximity_score',
            'profitability_score',
            'leverage_score',
            'efficiency_score',
            'growth_score',
            'relative_score',
            'overall_score'
        ]

        # Color scheme
        self.colors = {
            'header': 'FF1F4E78',      # Dark blue header
            'subheader': 'FF4472C4',   # Medium blue
            'config': 'FFF4B084',      # Orange for config boxes
            'good': 'FFC6EFCE',        # Light green
            'warning': 'FFFFF2CC',     # Light yellow
            'bad': 'FFFFC7CE',         # Light red
            'neutral': 'FFE7E6E6',     # Light gray
            'kmeans': 'FFCCE5FF',      # Light blue
            'hierarchical': 'FFCCFFCC', # Light green
            'dbscan': 'FFFFCCCC'       # Light red
        }

        logger.info("✓ ResearchExcelWriter initialized")

    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✓ Config loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def create_research_excel(self, output_path: Path) -> str:
        """
        Create research-oriented Excel file with all sections

        Args:
            output_path: Path for output Excel file

        Returns:
            Path to created Excel file
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING RESEARCH-ORIENTED EXCEL FILE")
        logger.info("="*80)

        # Step 1: Create consolidated overview dataframe
        logger.info("\n→ Step 1: Building consolidated overview...")
        overview_df = self._create_overview_dataframe()

        if overview_df is None or len(overview_df) == 0:
            logger.error("❌ Failed to create overview dataframe")
            return None

        logger.info(f"  ✓ Overview created: {len(overview_df)} companies")

        # Step 2: Calculate additional metrics
        logger.info("\n→ Step 2: Calculating metrics...")
        overview_df = self._add_size_categories(overview_df)
        overview_df = self._identify_outliers(overview_df)
        overview_df = self._calculate_consensus_metrics(overview_df)
        logger.info("  ✓ Metrics calculated")

        # Step 3: Create Excel file
        logger.info("\n→ Step 3: Creating Excel workbook...")
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        # Section 0: Config & Overview
        logger.info("\n→ Step 4: Creating Section 0 (Config & Overview)...")
        self._create_section_0_config_overview(wb, overview_df)
        logger.info("  ✓ Section 0 complete")

        # Section 1: Homogenität
        logger.info("\n→ Step 5: Creating Section 1 (Homogenität)...")
        self._create_section_1_homogeneity(wb, overview_df)
        logger.info("  ✓ Section 1 complete")

        # Section 2: Kongruenz (TODO)
        logger.info("\n→ Step 6: Section 2 (Kongruenz) - TODO")

        # Section 3: Treiber (TODO)
        logger.info("\n→ Step 7: Section 3 (Treiber) - TODO")

        # Section 4: Stabilität (TODO)
        logger.info("\n→ Step 8: Section 4 (Stabilität) - TODO")

        # Save workbook
        logger.info("\n→ Step 9: Saving workbook...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(output_path)
        logger.info(f"  ✓ Excel file saved: {output_path}")

        logger.info("\n" + "="*80)
        logger.info("✓ RESEARCH EXCEL FILE CREATED SUCCESSFULLY")
        logger.info("="*80)

        return str(output_path)

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def _create_overview_dataframe(self) -> pd.DataFrame:
        """Create consolidated overview dataframe from all algorithms"""
        dfs = []

        for algo_name, algo_data in self.algorithm_results.items():
            # Try different analysis types
            for analysis_type in ['combined', 'static', 'dynamic']:
                if analysis_type in algo_data:
                    df = algo_data[analysis_type].get('df')
                    if df is not None and len(df) > 0:
                        df = df.copy()
                        df['algorithm'] = algo_name
                        df['analysis_type'] = analysis_type
                        dfs.append(df)
                        break

        if not dfs:
            logger.error("No dataframes found in algorithm results")
            return None

        # Merge all dataframes
        overview_df = pd.concat(dfs, ignore_index=True)

        # Ensure required columns exist
        required = ['conm', 'gvkey', 'cluster', 'algorithm']
        for col in required:
            if col not in overview_df.columns:
                logger.error(f"Missing required column: {col}")
                return None

        return overview_df

    def _add_size_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add revenue-based size categories"""
        if 'revt' not in df.columns:
            logger.warning("'revt' column not found, skipping size categories")
            df['size_category'] = 'Unknown'
            return df

        # Calculate quantiles
        q33 = df['revt'].quantile(0.33)
        q66 = df['revt'].quantile(0.66)

        def categorize_size(revenue):
            if pd.isna(revenue):
                return 'Unknown'
            elif revenue <= q33:
                return 'Small'
            elif revenue <= q66:
                return 'Medium'
            else:
                return 'Large'

        df['size_category'] = df['revt'].apply(categorize_size)
        logger.info(f"  ✓ Size categories added: {df['size_category'].value_counts().to_dict()}")
        return df

    def _identify_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify outliers based on multiple criteria"""
        df['is_outlier'] = False
        df['outlier_reason'] = ''

        # Criterion 1: DBSCAN Noise
        if 'cluster' in df.columns:
            dbscan_mask = (df['algorithm'] == 'dbscan') & (df['cluster'] == -1)
            df.loc[dbscan_mask, 'is_outlier'] = True
            df.loc[dbscan_mask, 'outlier_reason'] = 'DBSCAN Noise'

        # Criterion 2: Low Overall Score
        if 'overall_score' in df.columns:
            low_score_mask = df['overall_score'] < 30
            df.loc[low_score_mask, 'is_outlier'] = True
            df.loc[low_score_mask & (df['outlier_reason'] == ''), 'outlier_reason'] = 'Low Score (<30)'
            df.loc[low_score_mask & (df['outlier_reason'] != ''), 'outlier_reason'] += ' + Low Score'

        # Criterion 3: High Score Variance (if multiple algorithms present)
        score_cols = [col for col in self.score_columns if col in df.columns]
        if len(score_cols) > 0:
            df['score_std'] = df[score_cols].std(axis=1)
            high_var_mask = df['score_std'] > 20
            df.loc[high_var_mask, 'is_outlier'] = True
            df.loc[high_var_mask & (df['outlier_reason'] == ''), 'outlier_reason'] = 'High Variance'
            df.loc[high_var_mask & (df['outlier_reason'] != ''), 'outlier_reason'] += ' + High Variance'

        outlier_count = df['is_outlier'].sum()
        logger.info(f"  ✓ Outliers identified: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
        return df

    def _calculate_consensus_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consensus metrics across algorithms"""
        # Group by company
        company_groups = df.groupby('gvkey')

        # Count unique cluster assignments
        df['unique_clusters'] = company_groups['cluster'].transform('nunique')

        # Calculate agreement score (0-100)
        df['algorithm_agreement'] = 100 * (1 - (df['unique_clusters'] - 1) / (df['algorithm'].nunique() - 1))

        # Determine consensus cluster (most frequent)
        df['consensus_cluster'] = company_groups['cluster'].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])

        logger.info("  ✓ Consensus metrics calculated")
        return df

    # =========================================================================
    # SECTION 0: CONFIG & OVERVIEW
    # =========================================================================

    def _create_section_0_config_overview(self, wb: Workbook, df: pd.DataFrame):
        """Create Section 0: Config & Overview"""
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

    # =========================================================================
    # SECTION 1: HOMOGENITÄT
    # =========================================================================

    def _create_section_1_homogeneity(self, wb: Workbook, df: pd.DataFrame):
        """Create Section 1: Homogenität (4 sheets)"""
        self._create_section_1a_tables(wb, df)
        self._create_section_1b_charts(wb, df)
        self._create_section_1c_sector(wb, df)
        self._create_section_1d_outliers(wb, df)

    def _create_section_1a_tables(self, wb: Workbook, df: pd.DataFrame):
        """Section 1a: Homogenität Tables"""
        ws = wb.create_sheet("1a_Homogenität_Tabellen")

        # Title
        ws['A1'] = "SECTION 1a: HOMOGENITÄT - TABLES"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:J1')

        row = 3

        # ===== CLUSTER STATISTICS PER ALGORITHM =====
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo].copy()

            ws[f'A{row}'] = f"{algo.upper()} - CLUSTER STATISTICS"
            ws[f'A{row}'].font = Font(size=12, bold=True)
            ws[f'A{row}'].fill = PatternFill(start_color=self.colors[algo], fill_type='solid')
            ws.merge_cells(f'A{row}:J{row}')
            row += 1

            # Group by cluster
            cluster_stats = []
            for cluster_id in sorted(algo_df['cluster'].unique()):
                cluster_df = algo_df[algo_df['cluster'] == cluster_id]

                stats = {
                    'Cluster': cluster_id,
                    'Size': len(cluster_df),
                    'Size %': f"{len(cluster_df)/len(algo_df)*100:.1f}%"
                }

                # Add score statistics
                for score_col in self.score_columns:
                    if score_col in cluster_df.columns:
                        stats[f'{score_col}_mean'] = cluster_df[score_col].mean()
                        stats[f'{score_col}_std'] = cluster_df[score_col].std()

                cluster_stats.append(stats)

            # Convert to dataframe
            stats_df = pd.DataFrame(cluster_stats)

            # Write to sheet
            for r_idx, row_data in enumerate(dataframe_to_rows(stats_df, index=False, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:  # Header
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        cell.number_format = '0.00'
                row += 1

            row += 2

        # Column widths
        for col in range(1, 11):
            ws.column_dimensions[chr(64+col)].width = 15

        logger.info("  ✓ Section 1a sheet created")

    def _create_section_1b_charts(self, wb: Workbook, df: pd.DataFrame):
        """Section 1b: Homogenität Charts"""
        ws = wb.create_sheet("1b_Homogenität_Charts")

        # Title
        ws['A1'] = "SECTION 1b: HOMOGENITÄT - CHARTS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        row = 3

        # ===== CLUSTER SIZE BAR CHART =====
        ws[f'A{row}'] = "CLUSTER SIZES BY ALGORITHM"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        row += 1

        # Prepare data
        cluster_sizes = df.groupby(['algorithm', 'cluster']).size().unstack(fill_value=0)

        # Write data
        start_row = row
        for r_idx, row_data in enumerate(dataframe_to_rows(cluster_sizes, index=True, header=True)):
            for c_idx, value in enumerate(row_data):
                cell = ws.cell(row=row, column=c_idx+1, value=value)
                if r_idx == 0:  # Header
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
            row += 1

        # Add bar chart
        chart = BarChart()
        chart.title = "Cluster Sizes"
        chart.x_axis.title = "Algorithm"
        chart.y_axis.title = "Number of Companies"
        chart.type = "col"
        chart.grouping = "clustered"

        data = Reference(ws, min_col=2, max_col=cluster_sizes.shape[1]+1,
                        min_row=start_row, max_row=row-1)
        cats = Reference(ws, min_col=1, min_row=start_row+1, max_row=row-1)

        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 15
        chart.width = 25

        ws.add_chart(chart, 'A' + str(row + 2))

        row += 20

        # ===== SCORE DISTRIBUTION TABLE =====
        if 'overall_score' in df.columns:
            ws[f'A{row}'] = "OVERALL SCORE DISTRIBUTION"
            ws[f'A{row}'].font = Font(size=12, bold=True)
            row += 1

            # Calculate percentiles
            score_dist = df.groupby('algorithm')['overall_score'].describe()

            # Write data
            for r_idx, row_data in enumerate(dataframe_to_rows(score_dist, index=True, header=True)):
                for c_idx, value in enumerate(row_data):
                    cell = ws.cell(row=row, column=c_idx+1, value=value)
                    if r_idx == 0:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                    elif isinstance(value, float):
                        cell.number_format = '0.00'
                row += 1

        logger.info("  ✓ Section 1b sheet created")

    def _create_section_1c_sector(self, wb: Workbook, df: pd.DataFrame):
        """Section 1c: Homogenität by Sector"""
        ws = wb.create_sheet("1c_Homogenität_Sektor")

        # Title
        ws['A1'] = "SECTION 1c: HOMOGENITÄT BY SECTOR"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:F1')

        if 'gsector' not in df.columns:
            ws['A3'] = "No sector data available"
            logger.warning("  ⚠ No sector data for Section 1c")
            return

        row = 3

        # ===== CLUSTER QUALITY BY SECTOR =====
        ws[f'A{row}'] = "CLUSTER QUALITY METRICS BY SECTOR"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:F{row}')
        row += 1

        # Calculate metrics per sector
        sector_metrics = []
        for sector in sorted(df['gsector'].unique()):
            if pd.isna(sector):
                continue

            sector_df = df[df['gsector'] == sector]

            metrics = {
                'Sector': sector,
                'N_Companies': len(sector_df),
                'N_Clusters': sector_df['cluster'].nunique(),
                'Outlier_Rate_%': f"{sector_df['is_outlier'].mean()*100:.1f}" if 'is_outlier' in sector_df.columns else 'N/A'
            }

            # Add score statistics
            if 'overall_score' in sector_df.columns:
                metrics['Avg_Score'] = sector_df['overall_score'].mean()
                metrics['Score_StdDev'] = sector_df['overall_score'].std()

            sector_metrics.append(metrics)

        # Convert to dataframe
        sector_df = pd.DataFrame(sector_metrics)

        # Write to sheet
        start_row = row
        for r_idx, row_data in enumerate(dataframe_to_rows(sector_df, index=False, header=True)):
            for c_idx, value in enumerate(row_data):
                cell = ws.cell(row=row, column=c_idx+1, value=value)
                if r_idx == 0:  # Header
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color=self.colors['neutral'], fill_type='solid')
                elif isinstance(value, float):
                    cell.number_format = '0.00'
            row += 1

        # Add conditional formatting for scores
        if 'Avg_Score' in sector_df.columns:
            score_col = chr(64 + sector_df.columns.get_loc('Avg_Score') + 1)
            ws.conditional_formatting.add(
                f'{score_col}{start_row+1}:{score_col}{row-1}',
                ColorScaleRule(
                    start_type='min', start_color='FFC7CE',
                    mid_type='percentile', mid_value=50, mid_color='FFEB9C',
                    end_type='max', end_color='C6EFCE'
                )
            )

        # Column widths
        for col in range(1, 7):
            ws.column_dimensions[chr(64+col)].width = 18

        logger.info("  ✓ Section 1c sheet created")

    def _create_section_1d_outliers(self, wb: Workbook, df: pd.DataFrame):
        """Section 1d: Outlier Analysis"""
        ws = wb.create_sheet("1d_Homogenität_Outliers")

        # Title
        ws['A1'] = "SECTION 1d: OUTLIER ANALYSIS"
        ws['A1'].font = Font(size=14, bold=True, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color=self.colors['header'], fill_type='solid')
        ws.merge_cells('A1:H1')

        if 'is_outlier' not in df.columns:
            ws['A3'] = "No outlier data available"
            logger.warning("  ⚠ No outlier data for Section 1d")
            return

        row = 3

        # ===== OUTLIER SUMMARY =====
        ws[f'A{row}'] = "OUTLIER SUMMARY"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws[f'A{row}'].fill = PatternFill(start_color=self.colors['subheader'], fill_type='solid')
        ws.merge_cells(f'A{row}:B{row}')
        row += 1

        outliers_df = df[df['is_outlier'] == True]

        summary = [
            ("Total Outliers", len(outliers_df)),
            ("Outlier Rate (%)", f"{len(outliers_df)/len(df)*100:.1f}%"),
            ("DBSCAN Noise", len(outliers_df[outliers_df['outlier_reason'].str.contains('Noise', na=False)])),
            ("Low Score", len(outliers_df[outliers_df['outlier_reason'].str.contains('Low Score', na=False)])),
            ("High Variance", len(outliers_df[outliers_df['outlier_reason'].str.contains('High Variance', na=False)])),
        ]

        for label, value in summary:
            ws[f'A{row}'] = label
            ws[f'B{row}'] = value
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

        row += 2

        # ===== OUTLIER DETAILS =====
        ws[f'A{row}'] = "OUTLIER COMPANIES"
        ws[f'A{row}'].font = Font(size=12, bold=True)
        ws.merge_cells(f'A{row}:H{row}')
        row += 1

        # Select relevant columns
        outlier_cols = ['conm', 'gvkey', 'algorithm', 'cluster', 'outlier_reason']
        if 'overall_score' in outliers_df.columns:
            outlier_cols.append('overall_score')
        if 'gsector' in outliers_df.columns:
            outlier_cols.append('gsector')

        outlier_display = outliers_df[outlier_cols].copy()

        # Write to sheet
        for r_idx, row_data in enumerate(dataframe_to_rows(outlier_display, index=False, header=True)):
            for c_idx, value in enumerate(row_data):
                cell = ws.cell(row=row, column=c_idx+1, value=value)
                if r_idx == 0:  # Header
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color=self.colors['bad'], fill_type='solid')
                elif isinstance(value, float):
                    cell.number_format = '0.00'
            row += 1

        # Column widths
        ws.column_dimensions['A'].width = 30
        for col in range(2, 9):
            ws.column_dimensions[chr(64+col)].width = 15

        logger.info("  ✓ Section 1d sheet created")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _embed_png(self, ws, png_path: Path, cell: str, scale: float = 0.5):
        """Embed PNG image into worksheet"""
        try:
            if not png_path.exists():
                logger.warning(f"PNG not found: {png_path}")
                return False

            # Load and resize image
            img = Image.open(png_path)

            # Resize if needed
            if scale != 1.0:
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save to BytesIO
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Create Excel image
            xl_img = XLImage(img_byte_arr)
            ws.add_image(xl_img, cell)

            logger.info(f"  ✓ PNG embedded: {png_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error embedding PNG {png_path}: {e}")
            return False

    def _apply_border(self, ws, cell_range: str, style: str = 'thin'):
        """Apply border to cell range"""
        border = Border(
            left=Side(style=style),
            right=Side(style=style),
            top=Side(style=style),
            bottom=Side(style=style)
        )

        for row in ws[cell_range]:
            for cell in row:
                cell.border = border


# =========================================================================
# INTEGRATION WITH COMPARISON PIPELINE
# =========================================================================

def create_research_excel(algorithm_results: Dict, output_dir: Path, market: str = 'germany') -> str:
    """
    Convenience function to create research Excel file

    Args:
        algorithm_results: Dict with results from all algorithms
        output_dir: Output directory
        market: Market name

    Returns:
        Path to created Excel file
    """
    writer = ResearchExcelWriter(algorithm_results, market=market)
    output_path = output_dir / 'research_analysis_master.xlsx'
    return writer.create_research_excel(output_path)
