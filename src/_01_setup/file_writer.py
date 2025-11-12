"""
File Writer Module
Handles all file writing operations for OutputHandler
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import joblib
import json

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization

    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)

    Returns:
        Object with all numpy types converted to Python native types
    """
    # Handle pandas DataFrames/Series first (before pd.isna check)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return convert_numpy_types(obj.to_dict())
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for scalar NaN values (not DataFrames/Series)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try pd.isna only for scalar values
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass

        # For any other objects, check if they're common non-serializable types
        # (models, transformers, functions, etc.)
        if hasattr(obj, '__module__') and hasattr(obj, '__class__'):
            # Check for common non-serializable types
            class_name = obj.__class__.__name__
            module_name = obj.__module__ if hasattr(obj, '__module__') else ''

            # Check for sklearn, scipy, or other ML library objects
            if 'sklearn' in module_name or 'scipy' in module_name:
                return f"<{class_name}>"

            # Check for common non-serializable class names
            if any(x in class_name.lower() for x in ['transformer', 'model', 'scaler', 'estimator', 'pipeline', 'pca']):
                return f"<{class_name}>"

        # For everything else, try to return as-is (will fail at json.dump if not serializable)
        return obj


class FileWriter:
    """
    Handles all file writing operations

    Responsible for:
    - Saving DataFrames (CSV, Excel)
    - Writing models and serialized data
    - Creating reports and READMEs
    - Managing file I/O operations
    """

    def __init__(self, path_manager, data_formatter):
        """
        Initialize FileWriter

        Args:
            path_manager: PathManager instance for directory paths
            data_formatter: DataFormatter instance for data formatting
        """
        self.path_manager = path_manager
        self.data_formatter = data_formatter

    def save_cluster_data(
        self,
        df: pd.DataFrame,
        cluster_profiles: pd.DataFrame,
        analysis_type: str = 'static',
        metrics: Dict = None
    ):
        """
        Speichert Cluster-Daten

        Args:
            df: DataFrame mit Cluster-Spalte
            cluster_profiles: Cluster-Profile
            analysis_type: 'static', 'dynamic', oder 'combined'
            metrics: Clustering-Metriken
        """
        analysis_name = self.path_manager.analysis_types[analysis_type]
        data_dir = self.path_manager.algorithm_dir / analysis_name / 'data'

        logger.info(f"\n  Speichere {analysis_type} → {analysis_name}/data/")

        # 1. Assignments
        assignments_path = data_dir / 'assignments.csv'
        df.to_csv(assignments_path, index=False)
        logger.info(f"    ✓ {assignments_path.name} ({len(df)} Unternehmen)")

        # 2. Profiles
        profiles_path = data_dir / 'profiles.csv'
        cluster_profiles.to_csv(profiles_path)
        logger.info(f"    ✓ {profiles_path.name} ({len(cluster_profiles)} Cluster)")

        # 3. Metrics
        if metrics:
            # Filter out non-serializable objects (models, transformers, scalers)
            metrics_clean = {k: v for k, v in metrics.items()
                           if k not in ['scaler', 'model', 'pca', 'pca_transformer', 'transformer']}
            # Convert numpy types to native Python types for JSON serialization
            metrics_clean = convert_numpy_types(metrics_clean)
            metrics_path = data_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_clean, f, indent=2)
            logger.info(f"    ✓ {metrics_path.name}")

    def save_cluster_lists(
        self,
        df: pd.DataFrame,
        analysis_type: str = 'static',
        sort_by: str = 'roa'
    ):
        """Erstellt separate CSV pro Cluster"""
        analysis_name = self.path_manager.analysis_types[analysis_type]
        clusters_dir = self.path_manager.algorithm_dir / analysis_name / 'reports' / 'clusters'

        logger.info(f"\n  Erstelle Cluster-Listen ({analysis_type})...")

        valid_df = df[df['cluster'] >= 0]

        for cluster_id in sorted(valid_df['cluster'].unique()):
            cluster_df = valid_df[valid_df['cluster'] == cluster_id].copy()

            cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in cluster_df.columns else f'cluster_{cluster_id}'
            safe_name = cluster_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

            if sort_by in cluster_df.columns:
                cluster_df = cluster_df.sort_values(sort_by, ascending=False)

            filename = f'{cluster_id}_{safe_name}.csv'
            path = clusters_dir / filename
            cluster_df.to_csv(path, index=False)

            logger.info(f"    ✓ Cluster {cluster_id}: {len(cluster_df):3} Unternehmen → {filename}")

    def save_models(
        self,
        scaler,
        model,
        analysis_type: str = 'static',
        pca_model = None
    ):
        """Speichert ML-Modelle"""
        analysis_name = self.path_manager.analysis_types[analysis_type]
        models_dir = self.path_manager.algorithm_dir / analysis_name / 'models'

        joblib.dump(scaler, models_dir / 'scaler.pkl')
        joblib.dump(model, models_dir / 'model.pkl')

        if pca_model is not None:
            joblib.dump(pca_model, models_dir / 'pca_model.pkl')
            logger.info(f"    ✓ Models: scaler, model, pca_model")
        else:
            logger.info(f"    ✓ Models: scaler, model")

    def save_processed_features(self, df: pd.DataFrame):
        """Speichert verarbeitete Features in 01_data/"""
        path = self.path_manager.data_dir / 'processed_features.csv'
        df.to_csv(path, index=False)
        logger.info(f"✓ Processed Features: {path}")

    def save_comparison_data(
        self,
        comp_type: str,
        data: pd.DataFrame,
        filename: str
    ):
        """Speichert Comparison-Daten direkt in comp_type/ Ordner"""
        # Dateien direkt in comp_type/ speichern (keine data/ Unterordner)
        comp_dir = self.path_manager.comparisons_dir / comp_type
        comp_dir.mkdir(parents=True, exist_ok=True)
        path = comp_dir / filename
        data.to_csv(path, index=False)
        logger.info(f"    ✓ {comp_type}/{filename}")

    def create_readme(self):
        """Erstellt README in 99_summary/"""
        readme_content = f"""# Clustering Analysis Results - {self.path_manager.market.upper()}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Structure

### 01_data/
Processed feature data used for clustering

### 02_algorithms/
Clustering results per algorithm:
- `{self.path_manager.algorithm_name}/` - {self.path_manager.algorithm.upper()} results
  - `{self.path_manager.analysis_types['static']}/` - Static features analysis
  - `{self.path_manager.analysis_types['dynamic']}/` - Dynamic features analysis
  - `{self.path_manager.analysis_types['combined']}/` - Combined analysis

### 03_comparisons/
Cross-algorithm comparisons:
- `algorithms/` - Performance metrics comparison
- `gics/` - Independence from GICS sectors
- `features/` - Feature importance analysis
- `temporal/` - Temporal stability

### 99_summary/
Executive summaries and interpretation reports

## Mode

This analysis uses **{self.path_manager.mode.upper()} Mode**:
"""

        if self.path_manager.mode == 'comparative':
            readme_content += """
- Static, Dynamic, and Combined are **3 independent clusterings**
- Each uses different feature sets
- Clusters are not directly comparable across analyses
"""
        else:
            readme_content += """
- Static creates **master cluster labels**
- Dynamic and Combined **reuse same labels**
- Only scores change, not cluster assignments
- Allows tracking companies across feature dimensions
"""

        readme_path = self.path_manager.summary_dir / 'README.md'
        readme_path.write_text(readme_content)
        logger.info(f"✓ README: {readme_path}")

    def save_enhanced_company_analysis(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        analysis_type: str,
        score_columns: Optional[List[str]] = None,
        dimensional_scores: Optional[pd.DataFrame] = None,
        evolution_data: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        Saves enhanced multi-sheet Excel file with comprehensive score information

        Excel Structure:
        - Sheet 1: Overview (All companies + cluster averages + relative values + scores)
        - Sheet 2-N: Per-Cluster Rankings
        - Sheet N+1: Score Evolution (only for combined analysis)
        - Sheet N+2: Dimensional Scores (if provided)
        - Sheet N+3: Summary (Cluster statistics)

        Args:
            df: DataFrame with clustering results and scores
            profiles: Cluster profiles (average values per cluster)
            analysis_type: 'static', 'dynamic', or 'combined'
            score_columns: List of score column names (e.g., ['proximity_score', 'overall_score'])
            dimensional_scores: Optional DataFrame with dimensional scores
            evolution_data: Optional DataFrame with score evolution (for combined only)

        Returns:
            Path to created Excel file
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"CREATING ENHANCED EXCEL ANALYSIS - {analysis_type.upper()}")
        logger.info(f"{'='*80}")

        # Determine output directory (use new 4_company_insights structure)
        if hasattr(self.path_manager, 'get_analysis_level_dir'):
            # New 5-level structure
            output_dir = self.path_manager.get_analysis_level_dir(4, analysis_type) / 'data'
        else:
            # Legacy structure
            analysis_name = self.path_manager.analysis_types[analysis_type]
            output_dir = self.path_manager.algorithm_dir / analysis_name / 'reports'

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"company_analysis_enhanced_{analysis_type}_{timestamp}.xlsx"
        excel_path = output_dir / filename

        # Auto-detect features if not explicitly in profiles
        feature_columns = [col for col in profiles.columns if col in df.columns]
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Companies: {len(df)}")

        # Detect score columns if not provided
        if score_columns is None:
            score_columns = [col for col in df.columns if 'score' in col.lower()]

        logger.info(f"  Score Columns: {score_columns}")

        # Calculate cluster averages and relative values
        df_enhanced = df.copy()
        df_enhanced = self.data_formatter.calculate_cluster_averages(df_enhanced, profiles, feature_columns)
        df_enhanced = self.data_formatter.calculate_relative_values(df_enhanced, feature_columns)

        # Sort data
        primary_score = score_columns[0] if score_columns else None
        if primary_score and primary_score in df_enhanced.columns:
            df_enhanced = df_enhanced.sort_values(['cluster', primary_score], ascending=[True, False])
        else:
            df_enhanced = df_enhanced.sort_values('cluster')

        # Create Excel with multiple sheets
        logger.info(f"\n  Writing Excel sheets...")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Overview
            logger.info(f"    ✓ Overview")
            df_enhanced.to_excel(writer, sheet_name='Overview', index=False)
            self.data_formatter.apply_excel_formatting(writer, 'Overview', df_enhanced, 'overview', feature_columns, score_columns)

            # Sheets 2-N: Per-Cluster Rankings
            for cluster_id in sorted(df_enhanced['cluster'].unique()):
                if cluster_id < 0:  # Skip noise points
                    continue

                cluster_df = df_enhanced[df_enhanced['cluster'] == cluster_id].copy()

                # Add cluster rank
                if primary_score and primary_score in cluster_df.columns:
                    cluster_df['cluster_rank'] = cluster_df[primary_score].rank(ascending=False, method='min').astype(int)

                cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in cluster_df.columns else f'Cluster_{cluster_id}'
                # Safe sheet name (max 31 chars, no special chars)
                sheet_name = f"Cluster_{cluster_id}"[:31]

                logger.info(f"    ✓ {sheet_name} ({len(cluster_df)} companies)")
                cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
                self.data_formatter.apply_excel_formatting(writer, sheet_name, cluster_df, 'cluster', feature_columns, score_columns)

            # Sheet N+1: Score Evolution (only for combined)
            if evolution_data is not None and analysis_type == 'combined':
                logger.info(f"    ✓ Score_Evolution")

                # Merge evolution data with company info
                evolution_merged = evolution_data.copy()
                if 'company_name' not in evolution_merged.columns and 'company_name' in df.columns:
                    name_map = df.set_index('gvkey')['company_name'].to_dict() if 'gvkey' in df.columns else {}
                    if name_map:
                        evolution_merged['company_name'] = evolution_merged.index.map(name_map)

                # Add evolution icons
                if 'total_score_change' in evolution_merged.columns or 'change_static_to_combined' in evolution_merged.columns:
                    change_col = 'total_score_change' if 'total_score_change' in evolution_merged.columns else 'change_static_to_combined'

                    def get_icon(change):
                        if pd.isna(change):
                            return '➡️'
                        elif change > 10:
                            return '⬆️'
                        elif change < -10:
                            return '⬇️'
                        else:
                            return '➡️'

                    evolution_merged['trend'] = evolution_merged[change_col].apply(get_icon)

                evolution_merged.to_excel(writer, sheet_name='Score_Evolution', index=False)
                self.data_formatter.apply_excel_formatting(writer, 'Score_Evolution', evolution_merged, 'evolution', [], [])

            # Sheet N+2: Dimensional Scores (if provided)
            if dimensional_scores is not None and len(dimensional_scores) > 0:
                logger.info(f"    ✓ Dimensional_Scores")

                dim_df = dimensional_scores.copy()

                # Merge with company info if needed
                if 'gvkey' in df.columns and 'gvkey' not in dim_df.columns:
                    dim_df = df[['gvkey', 'company_name', 'cluster', 'cluster_name']].merge(
                        dim_df, left_index=True, right_index=True, how='inner'
                    )

                # Identify dimensional score columns
                dim_score_cols = [col for col in dim_df.columns if 'score' in col.lower() and col not in score_columns]

                dim_df.to_excel(writer, sheet_name='Dimensional_Scores', index=False)
                self.data_formatter.apply_excel_formatting(writer, 'Dimensional_Scores', dim_df, 'dimensional', [], dim_score_cols)

            # Sheet N+3: Summary
            logger.info(f"    ✓ Summary")
            summary_df = self.data_formatter.create_cluster_summary_table(df_enhanced, primary_score if primary_score else 'cluster')
            summary_df.to_excel(writer, sheet_name='Summary', index=True)
            self.data_formatter.apply_excel_formatting(writer, 'Summary', summary_df, 'summary', [], score_columns)

        logger.info(f"\n✓ Excel saved: {excel_path}")
        logger.info(f"  Sheets: {len(df_enhanced['cluster'].unique()) + 2 + (1 if evolution_data is not None else 0) + (1 if dimensional_scores is not None else 0)}")
        logger.info(f"{'='*80}\n")

        return excel_path
