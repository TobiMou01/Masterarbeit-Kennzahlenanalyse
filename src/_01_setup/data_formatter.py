"""
Data Formatter Module
Handles data formatting and transformation for OutputHandler
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class DataFormatter:
    """
    Handles all data formatting and transformation

    Responsible for:
    - Formatting data before saving
    - Calculating derived values (averages, relative values)
    - Applying Excel formatting
    - Creating summary tables
    """

    @staticmethod
    def calculate_cluster_averages(
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Adds cluster average columns to DataFrame

        Args:
            df: DataFrame with cluster assignments
            profiles: Cluster profiles
            features: List of features

        Returns:
            DataFrame with added columns: {feature}_cluster_avg
        """
        df_result = df.copy()

        for feature in features:
            if feature not in profiles.columns:
                continue

            # Map cluster ID to cluster average
            cluster_avg_map = profiles[feature].to_dict()
            df_result[f'{feature}_cluster_avg'] = df_result['cluster'].map(cluster_avg_map)

        return df_result

    @staticmethod
    def calculate_relative_values(
        df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Calculates relative values (absolute and percentage) vs cluster average

        Args:
            df: DataFrame with features and cluster averages
            features: List of features

        Returns:
            DataFrame with added columns: {feature}_vs_cluster, {feature}_rel_pct
        """
        df_result = df.copy()

        for feature in features:
            avg_col = f'{feature}_cluster_avg'

            if avg_col not in df_result.columns:
                continue

            # Absolute difference
            df_result[f'{feature}_vs_cluster'] = df_result[feature] - df_result[avg_col]

            # Percentage difference
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                df_result[f'{feature}_rel_pct'] = (
                    (df_result[feature] / df_result[avg_col] - 1) * 100
                )
                # Replace inf and -inf with NaN
                df_result[f'{feature}_rel_pct'] = df_result[f'{feature}_rel_pct'].replace([np.inf, -np.inf], np.nan)

        return df_result

    @staticmethod
    def apply_excel_formatting(
        writer: pd.ExcelWriter,
        sheet_name: str,
        df: pd.DataFrame,
        style: str,
        feature_columns: List[str] = None,
        score_columns: List[str] = None
    ):
        """
        Applies formatting to Excel sheet

        Args:
            writer: Excel writer object
            sheet_name: Name of sheet to format
            df: DataFrame being written
            style: 'overview', 'cluster', 'evolution', 'dimensional', or 'summary'
            feature_columns: List of feature column names
            score_columns: List of score column names
        """
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.formatting.rule import ColorScaleRule
        except ImportError:
            logger.warning("  ⚠️  openpyxl not available for formatting")
            return

        worksheet = writer.sheets[sheet_name]

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)  # Max 50 chars
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Header formatting (first row)
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # Apply conditional formatting for scores (0-100 scale)
        if score_columns and style in ['overview', 'cluster', 'dimensional']:
            for score_col in score_columns:
                if score_col in df.columns:
                    col_idx = df.columns.get_loc(score_col) + 1  # Excel is 1-indexed
                    col_letter = worksheet.cell(1, col_idx).column_letter

                    # Color scale: Red (0) → Yellow (50) → Green (100)
                    color_scale = ColorScaleRule(
                        start_type='num', start_value=0, start_color='FF6B6B',
                        mid_type='num', mid_value=50, mid_color='FFF176',
                        end_type='num', end_value=100, end_color='81C784'
                    )

                    range_str = f'{col_letter}2:{col_letter}{len(df)+1}'
                    worksheet.conditional_formatting.add(range_str, color_scale)

        # Apply conditional formatting for relative values
        if feature_columns and style in ['overview', 'cluster']:
            for feature in feature_columns:
                vs_cluster_col = f'{feature}_vs_cluster'

                if vs_cluster_col in df.columns:
                    col_idx = df.columns.get_loc(vs_cluster_col) + 1
                    col_letter = worksheet.cell(1, col_idx).column_letter

                    # Two-color scale: Red (negative) → White (0) → Green (positive)
                    # Find min/max for dynamic range
                    min_val = df[vs_cluster_col].min() if not df[vs_cluster_col].isna().all() else -10
                    max_val = df[vs_cluster_col].max() if not df[vs_cluster_col].isna().all() else 10

                    color_scale = ColorScaleRule(
                        start_type='num', start_value=min_val, start_color='FF6B6B',
                        mid_type='num', mid_value=0, mid_color='FFFFFF',
                        end_type='num', end_value=max_val, end_color='81C784'
                    )

                    range_str = f'{col_letter}2:{col_letter}{len(df)+1}'
                    worksheet.conditional_formatting.add(range_str, color_scale)

    @staticmethod
    def create_cluster_summary_table(
        df: pd.DataFrame,
        score_column: str
    ) -> pd.DataFrame:
        """
        Creates cluster summary statistics table

        Args:
            df: DataFrame with cluster assignments and scores
            score_column: Primary score column name

        Returns:
            Summary DataFrame with statistics per cluster
        """
        summary_data = []

        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id < 0:
                continue

            cluster_df = df[df['cluster'] == cluster_id]
            cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in cluster_df.columns else f'Cluster {cluster_id}'

            # Calculate statistics
            if score_column in cluster_df.columns:
                scores = cluster_df[score_column]
                summary_data.append({
                    'Cluster': cluster_name,
                    'Count': len(cluster_df),
                    'Avg Score': scores.mean(),
                    'Std Dev': scores.std(),
                    'Min Score': scores.min(),
                    'Max Score': scores.max(),
                    'Median': scores.median()
                })
            else:
                summary_data.append({
                    'Cluster': cluster_name,
                    'Count': len(cluster_df),
                    'Avg Score': np.nan,
                    'Std Dev': np.nan,
                    'Min Score': np.nan,
                    'Max Score': np.nan,
                    'Median': np.nan
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.set_index('Cluster')

        return summary_df
