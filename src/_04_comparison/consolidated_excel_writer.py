"""
Consolidated Excel Writer - Multi-Algorithm Comparison with Visuals
Creates a single comprehensive Excel file with all algorithms and visualizations
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, ScatterChart, Reference
from openpyxl.chart.series import DataPoint
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule

logger = logging.getLogger(__name__)


class ConsolidatedExcelWriter:
    """
    Creates consolidated Excel file with multi-algorithm comparison

    Features:
    - Single Overview sheet with all algorithms side-by-side
    - Boxplot visualizations for score distributions
    - Scatter plots for algorithm comparisons
    - Cluster analysis with consensus metrics
    - Excel-native filters and conditional formatting
    """

    def __init__(self, algorithm_results: Dict, market: str = 'germany'):
        """
        Initialize consolidated Excel writer

        Args:
            algorithm_results: Dict with results from all algorithms
                Format: {
                    'kmeans': {'combined': {'df': DataFrame, ...}},
                    'hierarchical': {'static': {'df': DataFrame, ...}},
                    'dbscan': {'static': {'df': DataFrame, ...}}
                }
            market: Market name
        """
        self.algorithm_results = algorithm_results
        self.market = market

        # Score columns to track
        self.score_columns = [
            'proximity_score',
            'profitability_score',
            'leverage_score',
            'efficiency_score',
            'growth_score',
            'relative_score',
            'overall_score'
        ]

        # Color schemes
        self.colors = {
            'header': 'FFD3D3D3',  # Light gray
            'kmeans': 'FFCCE5FF',   # Light blue
            'hierarchical': 'FFCCFFCC',  # Light green
            'dbscan': 'FFFFCCCC',   # Light red
            'consensus': 'FFFFF4CC'  # Light yellow
        }

        logger.info("✓ ConsolidatedExcelWriter initialized")

    def create_consolidated_excel(self, output_path: Path) -> str:
        """
        Create consolidated Excel file with all sheets and visuals

        Args:
            output_path: Path for output Excel file

        Returns:
            Path to created Excel file
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING CONSOLIDATED ALGORITHM COMPARISON EXCEL")
        logger.info("="*80)

        # Step 1: Create consolidated overview dataframe
        logger.info("\n→ Step 1: Building consolidated overview...")
        overview_df = self._create_overview_dataframe()

        if overview_df is None or len(overview_df) == 0:
            logger.error("❌ Failed to create overview dataframe")
            return None

        logger.info(f"  ✓ Overview created: {len(overview_df)} companies")

        # Step 2: Create Excel file with pandas first
        logger.info("\n→ Step 2: Writing Excel file...")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write Overview sheet
            overview_df.to_excel(writer, sheet_name='Overview', index=False, freeze_panes=(1, 3))
            logger.info("  ✓ Sheet 'Overview' written")

            # Write Summary sheet
            summary_df = self._create_summary_dataframe(overview_df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            logger.info("  ✓ Sheet 'Summary' written")

            # Write Cluster Analysis
            cluster_df = self._create_cluster_analysis_dataframe(overview_df)
            if cluster_df is not None:
                cluster_df.to_excel(writer, sheet_name='Cluster_Analysis', index=False)
                logger.info("  ✓ Sheet 'Cluster_Analysis' written")

        # Step 3: Load workbook and add formatting + visuals
        logger.info("\n→ Step 3: Adding formatting and visuals...")
        wb = load_workbook(output_path)

        # Apply conditional formatting to Overview
        self._apply_conditional_formatting(wb, 'Overview', overview_df)
        logger.info("  ✓ Conditional formatting applied")

        # Add filters to Overview
        self._add_excel_filters(wb, 'Overview', overview_df)
        logger.info("  ✓ Excel filters added")

        # Step 4: Create visualization sheets
        logger.info("\n→ Step 4: Creating visualization sheets...")

        # Boxplot sheet
        self._create_boxplot_sheet(wb, overview_df)
        logger.info("  ✓ Sheet 'Score_Distributions' created")

        # Scatter plot sheet
        self._create_scatter_sheet(wb, overview_df)
        logger.info("  ✓ Sheet 'Score_Comparisons' created")

        # Step 5: Add PNG visualizations
        logger.info("\n→ Step 5: Adding PNG visualizations...")
        self._add_png_visualizations(wb)
        logger.info("  ✓ PNG visualizations added")

        # Save workbook
        wb.save(output_path)

        logger.info("\n" + "="*80)
        logger.info(f"✓ Consolidated Excel created: {output_path}")
        logger.info(f"  Total sheets: {len(wb.sheetnames)}")
        logger.info(f"  Companies: {len(overview_df)}")
        logger.info("="*80 + "\n")

        return str(output_path)

    def _add_png_visualizations(self, wb):
        """
        Add PNG visualizations from output folders to Excel sheets

        Args:
            wb: Workbook object
        """
        from openpyxl.drawing.image import Image as XLImage
        from PIL import Image
        import io

        base_path_combined = Path(f'output/{self.market}/02_algorithms/kmeans_comparative/combined')
        base_path_comparisons = Path(f'output/{self.market}/03_comparisons')

        embedded_count = 0

        # ===== OVERVIEW SHEET =====
        if 'Overview' in wb.sheetnames:
            ws = wb['Overview']
            plot_path = base_path_comparisons / 'algorithms/metrics_comparison_combined.png'
            if plot_path.exists():
                try:
                    img = Image.open(plot_path)
                    new_size = (int(img.width * 0.3), int(img.height * 0.3))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    xl_img = XLImage(img_byte_arr)
                    ws.add_image(xl_img, 'O3')
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"  ⚠ Could not embed metrics_comparison: {e}")

        # ===== SUMMARY SHEET =====
        if 'Summary' in wb.sheetnames:
            ws = wb['Summary']
            plot_path = base_path_comparisons / 'algorithms/algorithm_overlap_combined.png'
            if plot_path.exists():
                try:
                    img = Image.open(plot_path)
                    new_size = (int(img.width * 0.3), int(img.height * 0.3))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    xl_img = XLImage(img_byte_arr)
                    ws.add_image(xl_img, 'F3')
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"  ⚠ Could not embed algorithm_overlap: {e}")

        # ===== CLUSTER_ANALYSIS SHEET =====
        if 'Cluster_Analysis' in wb.sheetnames:
            ws = wb['Cluster_Analysis']
            plots_to_embed = [
                (base_path_combined / '2_algorithm_congruence/plots/confusion_matrix.png', 'H', 3, 0.3),
                (base_path_combined / '2_algorithm_congruence/plots/ari_heatmap_robustness.png', 'H', 25, 0.3),
            ]

            for plot_path, col, row, scale in plots_to_embed:
                if plot_path.exists():
                    try:
                        img = Image.open(plot_path)
                        new_size = (int(img.width * scale), int(img.height * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        xl_img = XLImage(img_byte_arr)
                        ws.add_image(xl_img, f'{col}{row}')
                        embedded_count += 1
                    except Exception as e:
                        logger.warning(f"  ⚠ Could not embed {plot_path.name}: {e}")

        # ===== SCORE_DISTRIBUTIONS SHEET =====
        if 'Score_Distributions' in wb.sheetnames:
            ws = wb['Score_Distributions']
            plot_path = base_path_combined / '1_cluster_quality/plots/score_distribution_overall.png'
            if plot_path.exists():
                try:
                    img = Image.open(plot_path)
                    new_size = (int(img.width * 0.35), int(img.height * 0.35))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    xl_img = XLImage(img_byte_arr)
                    ws.add_image(xl_img, 'G3')
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"  ⚠ Could not embed score_distribution: {e}")

        # ===== SCORE_COMPARISONS SHEET =====
        if 'Score_Comparisons' in wb.sheetnames:
            ws = wb['Score_Comparisons']
            plot_path = base_path_combined / '1_cluster_quality/plots/score_correlations.png'
            if plot_path.exists():
                try:
                    img = Image.open(plot_path)
                    new_size = (int(img.width * 0.35), int(img.height * 0.35))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    xl_img = XLImage(img_byte_arr)
                    ws.add_image(xl_img, 'G3')
                    embedded_count += 1
                except Exception as e:
                    logger.warning(f"  ⚠ Could not embed score_correlations: {e}")

        logger.info(f"    → Embedded {embedded_count} PNG visualizations")

    def _create_overview_dataframe(self) -> pd.DataFrame:
        """
        Create consolidated overview dataframe with all algorithms

        Returns:
            DataFrame with columns:
                - gvkey, company_name, sector info
                - For each algorithm: cluster, cluster_name, all scores
                - Consensus metrics
        """
        logger.info("  → Building overview dataframe...")

        # Start with base company data
        base_df = None

        # Try to get base data from kmeans combined first
        if 'kmeans' in self.algorithm_results:
            if 'combined' in self.algorithm_results['kmeans']:
                base_df = self.algorithm_results['kmeans']['combined']['df'][
                    ['gvkey', 'conm']
                ].copy()
                base_df = base_df.rename(columns={'conm': 'company_name'})

        # Fallback to other algorithms
        if base_df is None:
            for algo_name in ['hierarchical', 'dbscan', 'kmeans']:
                if algo_name in self.algorithm_results:
                    for stage in ['combined', 'static', 'dynamic']:
                        if stage in self.algorithm_results[algo_name]:
                            df = self.algorithm_results[algo_name][stage]['df']
                            if 'conm' in df.columns:
                                base_df = df[['gvkey', 'conm']].copy()
                                base_df = base_df.rename(columns={'conm': 'company_name'})
                                break
                    if base_df is not None:
                        break

        if base_df is None:
            logger.error("  ❌ No base company data found")
            return None

        base_df = base_df.drop_duplicates(subset=['gvkey'])
        logger.info(f"  → Found {len(base_df)} unique companies")

        # Add data from each algorithm
        overview_df = base_df.copy()

        for algo_name in ['kmeans', 'hierarchical', 'dbscan']:
            if algo_name not in self.algorithm_results:
                logger.info(f"  ⚠️  {algo_name}: No results available")
                continue

            # Get combined data if available, else static
            stage = 'combined' if 'combined' in self.algorithm_results[algo_name] else 'static'

            if stage not in self.algorithm_results[algo_name]:
                logger.info(f"  ⚠️  {algo_name}: No {stage} data available")
                continue

            algo_df = self.algorithm_results[algo_name][stage]['df']

            # Select columns to merge
            cols_to_merge = ['gvkey', 'cluster']

            # Add cluster_name if available
            if 'cluster_name' in algo_df.columns:
                cols_to_merge.append('cluster_name')

            # Add all score columns that exist
            for score_col in self.score_columns:
                if score_col in algo_df.columns:
                    cols_to_merge.append(score_col)

            # Prepare merge dataframe with renamed columns
            algo_merge_df = algo_df[cols_to_merge].copy()

            # Rename columns with algorithm prefix
            rename_dict = {}
            for col in algo_merge_df.columns:
                if col != 'gvkey':
                    rename_dict[col] = f'{algo_name}_{col}'

            algo_merge_df = algo_merge_df.rename(columns=rename_dict)

            # Merge with overview
            overview_df = overview_df.merge(algo_merge_df, on='gvkey', how='left')

            logger.info(f"  ✓ {algo_name}: {len(cols_to_merge)-1} columns added")

        # Add consensus metrics
        overview_df = self._add_consensus_metrics(overview_df)

        # Add metadata (sector info) if available
        overview_df = self._add_metadata(overview_df)

        logger.info(f"  ✓ Overview dataframe created: {overview_df.shape}")

        return overview_df

    def _add_consensus_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add consensus metrics across algorithms"""
        logger.info("  → Adding consensus metrics...")

        # Find overall_score columns for each algorithm
        score_cols = []
        for algo in ['kmeans', 'hierarchical', 'dbscan']:
            col_name = f'{algo}_overall_score'
            if col_name in df.columns:
                score_cols.append(col_name)

        if len(score_cols) > 0:
            # Average overall score across algorithms
            df['consensus_overall_score'] = df[score_cols].mean(axis=1)

            # Score standard deviation (measures disagreement)
            df['score_std_dev'] = df[score_cols].std(axis=1)

            # Count how many algorithms have valid scores
            df['algorithms_with_data'] = df[score_cols].notna().sum(axis=1)

            logger.info(f"    ✓ Consensus metrics calculated from {len(score_cols)} algorithms")

        # Cluster agreement metric
        cluster_cols = []
        for algo in ['kmeans', 'hierarchical', 'dbscan']:
            col_name = f'{algo}_cluster'
            if col_name in df.columns:
                cluster_cols.append(col_name)

        if len(cluster_cols) >= 2:
            # Count unique cluster assignments
            df['unique_cluster_assignments'] = df[cluster_cols].apply(
                lambda row: len(set(row.dropna().astype(int))),
                axis=1
            )

            # High agreement = 1, low agreement = number of algorithms
            df['cluster_agreement_score'] = 1.0 / df['unique_cluster_assignments']

            logger.info(f"    ✓ Cluster agreement calculated from {len(cluster_cols)} algorithms")

        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns (sector, size, etc.)"""
        # Try to find metadata in any algorithm result
        metadata_cols = ['gsector', 'ggroup', 'gind', 'gsubind', 'revt', 'at']

        for algo_name in ['kmeans', 'hierarchical', 'dbscan']:
            if algo_name not in self.algorithm_results:
                continue

            for stage in ['combined', 'static', 'dynamic']:
                if stage not in self.algorithm_results[algo_name]:
                    continue

                source_df = self.algorithm_results[algo_name][stage]['df']

                # Find available metadata columns
                available_metadata = [col for col in metadata_cols if col in source_df.columns]

                if len(available_metadata) > 0:
                    # Create metadata lookup
                    metadata_lookup = source_df[['gvkey'] + available_metadata].drop_duplicates(subset=['gvkey'])

                    # Merge metadata
                    df = df.merge(metadata_lookup, on='gvkey', how='left', suffixes=('', '_new'))

                    # Handle duplicates by keeping non-null values
                    for col in available_metadata:
                        if f'{col}_new' in df.columns:
                            df[col] = df[col].fillna(df[f'{col}_new'])
                            df = df.drop(columns=[f'{col}_new'])

                    logger.info(f"  ✓ Metadata added: {', '.join(available_metadata)}")
                    break

            if any(col in df.columns for col in metadata_cols):
                break

        return df

    def _create_summary_dataframe(self, overview_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics dataframe"""
        logger.info("  → Creating summary dataframe...")

        summary_data = []

        # General statistics
        summary_data.append({
            'Category': 'General',
            'Metric': 'Total Companies',
            'Value': len(overview_df)
        })

        # Algorithm-specific statistics
        for algo in ['kmeans', 'hierarchical', 'dbscan']:
            cluster_col = f'{algo}_cluster'
            score_col = f'{algo}_overall_score'

            if cluster_col in overview_df.columns:
                n_clusters = overview_df[cluster_col].nunique()
                summary_data.append({
                    'Category': algo.capitalize(),
                    'Metric': 'Number of Clusters',
                    'Value': n_clusters
                })

                # Companies with valid clustering
                n_valid = overview_df[cluster_col].notna().sum()
                summary_data.append({
                    'Category': algo.capitalize(),
                    'Metric': 'Companies Clustered',
                    'Value': n_valid
                })

            if score_col in overview_df.columns:
                avg_score = overview_df[score_col].mean()
                summary_data.append({
                    'Category': algo.capitalize(),
                    'Metric': 'Average Overall Score',
                    'Value': f'{avg_score:.2f}'
                })

        # Consensus statistics
        if 'consensus_overall_score' in overview_df.columns:
            avg_consensus = overview_df['consensus_overall_score'].mean()
            summary_data.append({
                'Category': 'Consensus',
                'Metric': 'Average Consensus Score',
                'Value': f'{avg_consensus:.2f}'
            })

        if 'cluster_agreement_score' in overview_df.columns:
            avg_agreement = overview_df['cluster_agreement_score'].mean()
            summary_data.append({
                'Category': 'Consensus',
                'Metric': 'Average Cluster Agreement',
                'Value': f'{avg_agreement:.3f}'
            })

            # High agreement companies (>= 0.9)
            high_agreement = (overview_df['cluster_agreement_score'] >= 0.9).sum()
            summary_data.append({
                'Category': 'Consensus',
                'Metric': 'High Agreement Companies',
                'Value': high_agreement
            })

        summary_df = pd.DataFrame(summary_data)
        logger.info(f"  ✓ Summary dataframe created: {len(summary_df)} metrics")

        return summary_df

    def _create_cluster_analysis_dataframe(self, overview_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create detailed cluster analysis"""
        logger.info("  → Creating cluster analysis dataframe...")

        cluster_data = []

        for algo in ['kmeans', 'hierarchical', 'dbscan']:
            cluster_col = f'{algo}_cluster'
            score_col = f'{algo}_overall_score'

            if cluster_col not in overview_df.columns:
                continue

            for cluster_id in sorted(overview_df[cluster_col].dropna().unique()):
                cluster_mask = overview_df[cluster_col] == cluster_id
                cluster_df = overview_df[cluster_mask]

                row = {
                    'Algorithm': algo,
                    'Cluster_ID': int(cluster_id),
                    'Company_Count': len(cluster_df)
                }

                if score_col in overview_df.columns:
                    row['Avg_Overall_Score'] = cluster_df[score_col].mean()
                    row['Min_Score'] = cluster_df[score_col].min()
                    row['Max_Score'] = cluster_df[score_col].max()
                    row['Std_Score'] = cluster_df[score_col].std()

                cluster_data.append(row)

        if len(cluster_data) == 0:
            return None

        cluster_df = pd.DataFrame(cluster_data)
        logger.info(f"  ✓ Cluster analysis created: {len(cluster_df)} clusters")

        return cluster_df

    def _apply_conditional_formatting(self, wb: Workbook, sheet_name: str, df: pd.DataFrame):
        """Apply conditional formatting to score columns"""
        ws = wb[sheet_name]

        # Find score columns
        score_cols = {}
        for col_idx, col_name in enumerate(df.columns, start=1):
            if 'score' in col_name.lower():
                # Get column letter
                from openpyxl.utils import get_column_letter
                col_letter = get_column_letter(col_idx)
                score_cols[col_name] = col_letter

        # Apply color scale to score columns
        for col_name, col_letter in score_cols.items():
            # Data starts at row 2 (after header)
            cell_range = f'{col_letter}2:{col_letter}{len(df)+1}'

            # Color scale: Red (0) -> Yellow (50) -> Green (100)
            color_scale_rule = ColorScaleRule(
                start_type='num', start_value=0, start_color='F8696B',
                mid_type='num', mid_value=50, mid_color='FFEB84',
                end_type='num', end_value=100, end_color='63BE7B'
            )

            ws.conditional_formatting.add(cell_range, color_scale_rule)

        logger.info(f"    ✓ Conditional formatting applied to {len(score_cols)} columns")

    def _add_excel_filters(self, wb: Workbook, sheet_name: str, df: pd.DataFrame):
        """Add Excel auto-filters"""
        ws = wb[sheet_name]

        # Add autofilter to header row
        ws.auto_filter.ref = ws.dimensions

        logger.info("    ✓ Excel filters enabled")

    def _create_boxplot_sheet(self, wb: Workbook, overview_df: pd.DataFrame):
        """Create sheet with boxplot descriptions (actual charts need external lib)"""
        ws = wb.create_sheet('Score_Distributions')

        # Add title
        ws['A1'] = 'Score Distribution Analysis'
        ws['A1'].font = Font(size=14, bold=True)

        # Add description
        ws['A3'] = 'This sheet shows the distribution of scores across algorithms.'
        ws['A4'] = 'Use the Overview sheet with filters to create boxplots in Excel (Insert > Charts > Box Plot)'

        # Add statistics table for each score type
        row = 6
        for score_type in ['overall_score', 'proximity_score', 'profitability_score']:
            ws[f'A{row}'] = f'{score_type.replace("_", " ").title()} Statistics:'
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

            # Header
            ws[f'A{row}'] = 'Algorithm'
            ws[f'B{row}'] = 'Mean'
            ws[f'C{row}'] = 'Median'
            ws[f'D{row}'] = 'Std Dev'
            ws[f'E{row}'] = 'Min'
            ws[f'F{row}'] = 'Max'
            row += 1

            # Data for each algorithm
            for algo in ['kmeans', 'hierarchical', 'dbscan']:
                col_name = f'{algo}_{score_type}'
                if col_name in overview_df.columns:
                    data = overview_df[col_name].dropna()
                    if len(data) > 0:
                        ws[f'A{row}'] = algo.capitalize()
                        ws[f'B{row}'] = data.mean()
                        ws[f'C{row}'] = data.median()
                        ws[f'D{row}'] = data.std()
                        ws[f'E{row}'] = data.min()
                        ws[f'F{row}'] = data.max()
                        row += 1

            row += 2

        logger.info("    ✓ Boxplot statistics added")

    def _create_scatter_sheet(self, wb: Workbook, overview_df: pd.DataFrame):
        """Create sheet with scatter plot data and instructions"""
        ws = wb.create_sheet('Score_Comparisons')

        # Add title
        ws['A1'] = 'Algorithm Score Comparison'
        ws['A1'].font = Font(size=14, bold=True)

        # Add instructions
        ws['A3'] = 'This sheet provides data for creating scatter plots comparing algorithms.'
        ws['A4'] = 'Instructions: Select data columns → Insert > Charts > Scatter Plot'

        # Prepare comparison data
        row = 6

        # Comparison 1: K-Means vs Hierarchical
        if 'kmeans_overall_score' in overview_df.columns and 'hierarchical_overall_score' in overview_df.columns:
            ws[f'A{row}'] = 'K-Means vs Hierarchical (Overall Score)'
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

            ws[f'A{row}'] = 'K-Means'
            ws[f'B{row}'] = 'Hierarchical'
            row += 1

            comp_df = overview_df[['kmeans_overall_score', 'hierarchical_overall_score']].dropna()
            for _, comp_row in comp_df.iterrows():
                ws[f'A{row}'] = comp_row['kmeans_overall_score']
                ws[f'B{row}'] = comp_row['hierarchical_overall_score']
                row += 1

            row += 2

        # Comparison 2: K-Means vs DBSCAN
        if 'kmeans_overall_score' in overview_df.columns and 'dbscan_overall_score' in overview_df.columns:
            ws[f'A{row}'] = 'K-Means vs DBSCAN (Overall Score)'
            ws[f'A{row}'].font = Font(bold=True)
            row += 1

            ws[f'A{row}'] = 'K-Means'
            ws[f'B{row}'] = 'DBSCAN'
            row += 1

            comp_df = overview_df[['kmeans_overall_score', 'dbscan_overall_score']].dropna()
            for _, comp_row in comp_df.iterrows():
                ws[f'A{row}'] = comp_row['kmeans_overall_score']
                ws[f'B{row}'] = comp_row['dbscan_overall_score']
                row += 1

        logger.info("    ✓ Scatter plot data added")


def create_consolidated_comparison_excel(
    algorithm_results: Dict,
    market: str = 'germany',
    output_dir: str = None
) -> str:
    """
    Create consolidated algorithm comparison Excel file

    Args:
        algorithm_results: Results from all algorithms
        market: Market name
        output_dir: Output directory (optional)

    Returns:
        Path to created Excel file
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING CONSOLIDATED ALGORITHM COMPARISON")
    logger.info("="*80)

    # Determine output path
    if output_dir is None:
        output_dir = f'output/{market}/03_comparisons'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_file = output_path / 'algorithm_comparison_combined.xlsx'

    # Create writer and generate Excel
    writer = ConsolidatedExcelWriter(algorithm_results, market)
    result = writer.create_consolidated_excel(excel_file)

    return result


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("CONSOLIDATED EXCEL WRITER TEST")
    print("="*80)
    print("\nThis module creates consolidated Excel comparison files.")
    print("Call from ComparisonPipeline.run_full_comparison_pipeline()")
    print("="*80)
