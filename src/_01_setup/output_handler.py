"""
Output Handler - Option B (Finale Struktur)
Unterscheidet zwischen Comparative und Hierarchical Mode
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


class OutputHandler:
    """
    Verwaltet alle Outputs in finaler Option B Struktur

    Struktur:
    output/{market}/
    ├── 01_data/                      # Rohdaten
    ├── 02_algorithms/                # Pro Algorithmus
    │   ├── kmeans_comparative/       # K-Means: 3 separate Clusterings
    │   │   ├── static/
    │   │   ├── dynamic/
    │   │   └── combined/
    │   ├── hierarchical/             # Hierarchical: Label-Consistency
    │   │   ├── master_clustering/
    │   │   ├── dynamic_enrichment/
    │   │   └── combined_scores/
    │   └── dbscan/                   # DBSCAN: Label-Consistency
    │       ├── master_clustering/
    │       ├── dynamic_enrichment/
    │       └── combined_scores/
    ├── 03_comparisons/               # Cross-Analysen
    │   ├── algorithms/
    │   ├── gics/
    │   ├── features/
    │   └── temporal/
    └── 99_summary/                   # Executive Reports
    """

    def __init__(
        self,
        market: str = 'germany',
        algorithm: str = 'kmeans',
        mode: str = 'auto',
        base_dir: str = 'output'
    ):
        """
        Initialisiert Output Handler

        Args:
            market: Market-Bezeichnung (germany, usa, etc.)
            algorithm: Clustering-Algorithmus ('kmeans', 'hierarchical', 'dbscan')
            mode: 'comparative' oder 'hierarchical' (oder 'auto' für Auto-Detect)
            base_dir: Basis-Verzeichnis
        """
        self.market = market
        self.algorithm = algorithm

        # Auto-detect mode based on algorithm
        if mode == 'auto':
            if algorithm == 'kmeans':
                self.mode = 'comparative'  # Default für K-Means
            else:
                self.mode = 'hierarchical'  # Hierarchical/DBSCAN nutzen Label-Consistency
        else:
            self.mode = mode

        # Base paths
        self.market_dir = Path(base_dir) / market
        self.data_dir = self.market_dir / '01_data'

        # Algorithm directory name
        if self.mode == 'comparative' and algorithm == 'kmeans':
            self.algorithm_name = 'kmeans_comparative'
        else:
            self.algorithm_name = algorithm

        self.algorithm_dir = self.market_dir / '02_algorithms' / self.algorithm_name
        self.comparisons_dir = self.market_dir / '03_comparisons'
        self.summary_dir = self.market_dir / '99_summary'

        # Analysis type naming based on mode
        if self.mode == 'comparative':
            self.analysis_types = {
                'static': 'static',
                'dynamic': 'dynamic',
                'combined': 'combined',
                'unified': 'unified'
            }
        else:  # hierarchical mode
            self.analysis_types = {
                'static': 'master_clustering',
                'dynamic': 'dynamic_enrichment',
                'combined': 'combined_scores',
                'unified': 'unified_all_features'
            }

        # Create directory structure
        self._create_directories()

        logger.info(f"✓ OutputHandler: {market} / {self.algorithm_name} ({self.mode} mode)")

    def _create_directories(self):
        """Erstellt die komplette Verzeichnisstruktur (Legacy-Struktur für Backward-Compatibility)"""

        # 02_algorithms/{algorithm}/
        for analysis_key in ['static', 'dynamic', 'combined']:
            analysis_dir = self.algorithm_dir / self.analysis_types[analysis_key]

            # data/, plots/, reports/
            (analysis_dir / 'data').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'plots').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'reports').mkdir(parents=True, exist_ok=True)
            (analysis_dir / 'reports' / 'clusters').mkdir(parents=True, exist_ok=True)

            # models/ nur für comparative mode (K-Means)
            # hierarchical/dbscan nutzen HierarchicalPipeline und speichern keine Modelle
            if self.mode == 'comparative':
                (analysis_dir / 'models').mkdir(parents=True, exist_ok=True)

        # 03_comparisons/
        # Dateien werden direkt in comp_type/ gespeichert, keine data/plots Unterordner
        for comp_type in ['algorithms', 'gics', 'features', 'temporal']:
            (self.comparisons_dir / comp_type).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # NEW: 5-LEVEL OUTPUT STRUCTURE (Enhanced Analysis Levels)
    # =========================================================================

    def create_output_structure(self):
        """
        Erstellt neue 5-Ebenen Output-Struktur

        Struktur:
        output/{market}/{algorithm}/{analysis_type}/
        ├── 1_cluster_quality/
        ├── 2_algorithm_congruence/
        ├── 3_external_validation/
        ├── 4_company_insights/
        │   ├── data/
        │   ├── plots/
        │   └── rankings/
        ├── 5_pca_analysis/
        └── summary/
        """
        logger.info(f"Creating new 5-level output structure for {self.market}/{self.algorithm_name}...")

        # Für jede Analyse-Type (static, dynamic, combined)
        for analysis_key in ['static', 'dynamic', 'combined']:
            analysis_name = self.analysis_types[analysis_key]
            base_path = self.algorithm_dir / analysis_name

            # Level 1: Cluster Quality
            level1_dir = base_path / '1_cluster_quality'
            (level1_dir / 'plots').mkdir(parents=True, exist_ok=True)

            # Level 2: Algorithm Congruence
            level2_dir = base_path / '2_algorithm_congruence'
            (level2_dir / 'data').mkdir(parents=True, exist_ok=True)
            (level2_dir / 'plots').mkdir(parents=True, exist_ok=True)

            # Level 3: External Validation
            level3_dir = base_path / '3_external_validation'
            (level3_dir / 'data').mkdir(parents=True, exist_ok=True)
            (level3_dir / 'plots').mkdir(parents=True, exist_ok=True)
            (level3_dir / 'contingency_tables').mkdir(parents=True, exist_ok=True)

            # Level 4: Company Insights
            level4_dir = base_path / '4_company_insights'
            (level4_dir / 'data').mkdir(parents=True, exist_ok=True)
            (level4_dir / 'plots').mkdir(parents=True, exist_ok=True)
            (level4_dir / 'rankings').mkdir(parents=True, exist_ok=True)
            (level4_dir / 'plots' / 'radar_charts').mkdir(parents=True, exist_ok=True)

            # Level 5: PCA Analysis (optional)
            level5_dir = base_path / '5_pca_analysis'
            (level5_dir / 'data').mkdir(parents=True, exist_ok=True)
            (level5_dir / 'plots').mkdir(parents=True, exist_ok=True)

            # Summary
            summary_dir = base_path / 'summary'
            summary_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"  ✓ Created 5-level structure for {analysis_name}")

        logger.info(f"✓ 5-level output structure created successfully")

    def get_analysis_level_dir(self, level: int, analysis_type: str = 'static') -> Path:
        """
        Liefert Pfad für Analyse-Ebene

        Args:
            level: 1-5 (cluster_quality, algorithm_congruence, external_validation,
                        company_insights, pca_analysis)
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für die gewünschte Ebene

        Example:
            >>> handler.get_analysis_level_dir(4, 'static')
            Path('output/germany/kmeans_comparative/static/4_company_insights')
        """
        level_names = {
            1: '1_cluster_quality',
            2: '2_algorithm_congruence',
            3: '3_external_validation',
            4: '4_company_insights',
            5: '5_pca_analysis'
        }

        if level not in level_names:
            raise ValueError(f"Invalid level: {level}. Must be 1-5.")

        if analysis_type not in self.analysis_types:
            raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'static', 'dynamic', or 'combined'.")

        analysis_name = self.analysis_types[analysis_type]
        level_dir = self.algorithm_dir / analysis_name / level_names[level]

        return level_dir

    def get_summary_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert Summary-Verzeichnis für eine Analyse

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für Summary-Verzeichnis
        """
        if analysis_type not in self.analysis_types:
            raise ValueError(f"Invalid analysis_type: {analysis_type}")

        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'summary'

    def get_pca_analysis_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert PCA Analysis Verzeichnis (Level 5)

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für PCA Analysis Verzeichnis
        """
        return self.get_analysis_level_dir(5, analysis_type)

    def get_cluster_quality_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert Cluster Quality Verzeichnis (Level 1)

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für Cluster Quality Verzeichnis
        """
        return self.get_analysis_level_dir(1, analysis_type)

    def get_algorithm_congruence_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert Algorithm Congruence Verzeichnis (Level 2)

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für Algorithm Congruence Verzeichnis
        """
        return self.get_analysis_level_dir(2, analysis_type)

    def get_external_validation_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert External Validation Verzeichnis (Level 3)

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für External Validation Verzeichnis
        """
        return self.get_analysis_level_dir(3, analysis_type)

    def get_company_insights_dir(self, analysis_type: str = 'static') -> Path:
        """
        Liefert Company Insights Verzeichnis (Level 4)

        Args:
            analysis_type: 'static', 'dynamic', oder 'combined'

        Returns:
            Path-Objekt für Company Insights Verzeichnis
        """
        return self.get_analysis_level_dir(4, analysis_type)

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
        analysis_name = self.analysis_types[analysis_type]
        data_dir = self.algorithm_dir / analysis_name / 'data'

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
            import numpy as np

            # Filter out non-serializable objects
            metrics_clean = {}
            for k, v in metrics.items():
                if k not in ['scaler', 'model', 'pca_metadata', 'transformer']:
                    # Convert numpy types to Python native types
                    if isinstance(v, (np.integer, np.int64, np.int32)):
                        metrics_clean[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64, np.float32)):
                        metrics_clean[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        metrics_clean[k] = v.tolist()
                    elif isinstance(v, list):
                        # Convert list elements
                        metrics_clean[k] = [
                            str(item) if isinstance(item, (np.integer, np.floating))
                            else item for item in v
                        ]
                    else:
                        metrics_clean[k] = v

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
        analysis_name = self.analysis_types[analysis_type]
        clusters_dir = self.algorithm_dir / analysis_name / 'reports' / 'clusters'

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
        analysis_name = self.analysis_types[analysis_type]
        models_dir = self.algorithm_dir / analysis_name / 'models'

        joblib.dump(scaler, models_dir / 'scaler.pkl')
        joblib.dump(model, models_dir / 'model.pkl')

        if pca_model is not None:
            joblib.dump(pca_model, models_dir / 'pca_model.pkl')
            logger.info(f"    ✓ Models: scaler, model, pca_model")
        else:
            logger.info(f"    ✓ Models: scaler, model")

    def get_plots_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Plots-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'plots'

    def get_reports_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Reports-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'reports'

    def save_processed_features(self, df: pd.DataFrame):
        """Speichert verarbeitete Features in 01_data/"""
        path = self.data_dir / 'processed_features.csv'
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
        comp_dir = self.comparisons_dir / comp_type
        comp_dir.mkdir(parents=True, exist_ok=True)
        path = comp_dir / filename
        data.to_csv(path, index=False)
        logger.info(f"    ✓ {comp_type}/{filename}")

    def get_comparison_plots_dir(self, comp_type: str) -> Path:
        """Gibt Comparison Plots Dir zurück (direkt in comp_type/)"""
        # Plots direkt in comp_type/ speichern (keine plots/ Unterordner)
        comp_dir = self.comparisons_dir / comp_type
        comp_dir.mkdir(parents=True, exist_ok=True)
        return comp_dir

    def create_readme(self):
        """Erstellt README in 99_summary/"""
        readme_content = f"""# Clustering Analysis Results - {self.market.upper()}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Structure

### 01_data/
Processed feature data used for clustering

### 02_algorithms/
Clustering results per algorithm:
- `{self.algorithm_name}/` - {self.algorithm.upper()} results
  - `{self.analysis_types['static']}/` - Static features analysis
  - `{self.analysis_types['dynamic']}/` - Dynamic features analysis
  - `{self.analysis_types['combined']}/` - Combined analysis

### 03_comparisons/
Cross-algorithm comparisons:
- `algorithms/` - Performance metrics comparison
- `gics/` - Independence from GICS sectors
- `features/` - Feature importance analysis
- `temporal/` - Temporal stability

### 99_summary/
Executive summaries and interpretation reports

## Mode

This analysis uses **{self.mode.upper()} Mode**:
"""

        if self.mode == 'comparative':
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

        readme_path = self.summary_dir / 'README.md'
        readme_path.write_text(readme_content)
        logger.info(f"✓ README: {readme_path}")

    # =========================================================================
    # ENHANCED EXCEL OUTPUT with Multi-Sheet Structure
    # =========================================================================

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
        if hasattr(self, 'get_analysis_level_dir'):
            # New 5-level structure
            output_dir = self.get_analysis_level_dir(4, analysis_type) / 'data'
        else:
            # Legacy structure
            analysis_name = self.analysis_types[analysis_type]
            output_dir = self.algorithm_dir / analysis_name / 'reports'

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
        df_enhanced = self._calculate_cluster_averages(df_enhanced, profiles, feature_columns)
        df_enhanced = self._calculate_relative_values(df_enhanced, feature_columns)

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
            self._apply_excel_formatting(writer, 'Overview', df_enhanced, 'overview', feature_columns, score_columns)

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
                self._apply_excel_formatting(writer, sheet_name, cluster_df, 'cluster', feature_columns, score_columns)

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
                self._apply_excel_formatting(writer, 'Score_Evolution', evolution_merged, 'evolution', [], [])

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
                self._apply_excel_formatting(writer, 'Dimensional_Scores', dim_df, 'dimensional', [], dim_score_cols)

            # Sheet N+3: Summary
            logger.info(f"    ✓ Summary")
            summary_df = self._create_cluster_summary_table(df_enhanced, primary_score if primary_score else 'cluster')
            summary_df.to_excel(writer, sheet_name='Summary', index=True)
            self._apply_excel_formatting(writer, 'Summary', summary_df, 'summary', [], score_columns)

        logger.info(f"\n✓ Excel saved: {excel_path}")
        logger.info(f"  Sheets: {len(df_enhanced['cluster'].unique()) + 2 + (1 if evolution_data is not None else 0) + (1 if dimensional_scores is not None else 0)}")
        logger.info(f"{'='*80}\n")

        return excel_path

    def _calculate_cluster_averages(
        self,
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

    def _calculate_relative_values(
        self,
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

    def _apply_excel_formatting(
        self,
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

    def _create_cluster_summary_table(
        self,
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

    def __repr__(self):
        return f"OutputHandler(market='{self.market}', algorithm='{self.algorithm_name}', mode='{self.mode}')"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("OUTPUT HANDLER TEST - Option B")
    print("="*80)

    # Test Comparative Mode (K-Means)
    print("\n1. K-Means Comparative Mode:")
    handler_km = OutputHandler(market='germany', algorithm='kmeans', mode='comparative')
    print(f"   Algorithm Dir: {handler_km.algorithm_dir}")
    print(f"   Analysis Types: {handler_km.analysis_types}")

    # Test Hierarchical Mode
    print("\n2. Hierarchical Mode:")
    handler_hc = OutputHandler(market='germany', algorithm='hierarchical')
    print(f"   Algorithm Dir: {handler_hc.algorithm_dir}")
    print(f"   Analysis Types: {handler_hc.analysis_types}")

    # Test DBSCAN Mode
    print("\n3. DBSCAN Mode:")
    handler_db = OutputHandler(market='germany', algorithm='dbscan')
    print(f"   Algorithm Dir: {handler_db.algorithm_dir}")
    print(f"   Analysis Types: {handler_db.analysis_types}")

    # Create README
    handler_km.create_readme()

    print("\n✓ Output Handler Test erfolgreich!")
    print("="*80)
