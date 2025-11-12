"""
Output Coordinator - Main Facade for OutputHandler
Coordinates between PathManager, FileWriter, and DataFormatter modules
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .path_manager import PathManager
from .file_writer import FileWriter
from .data_formatter import DataFormatter

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Verwaltet alle Outputs in finaler Option B Struktur

    This is the main facade that coordinates between:
    - PathManager: Directory structure and path generation
    - FileWriter: File I/O operations
    - DataFormatter: Data formatting and transformation

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

        # Initialize specialized modules
        self.path_manager = PathManager(market, algorithm, self.mode, base_dir)
        self.data_formatter = DataFormatter()
        self.file_writer = FileWriter(self.path_manager, self.data_formatter)

        # Expose commonly used attributes for backward compatibility
        self.market_dir = self.path_manager.market_dir
        self.data_dir = self.path_manager.data_dir
        self.algorithm_name = self.path_manager.algorithm_name
        self.algorithm_dir = self.path_manager.algorithm_dir
        self.comparisons_dir = self.path_manager.comparisons_dir
        self.summary_dir = self.path_manager.summary_dir
        self.analysis_types = self.path_manager.analysis_types

        # Create directory structure
        self.path_manager._create_directories()

        logger.info(f"✓ OutputHandler: {market} / {self.algorithm_name} ({self.mode} mode)")

    # =========================================================================
    # PATH MANAGER DELEGATION - Directory Structure & Paths
    # =========================================================================

    def create_output_structure(self):
        """Erstellt neue 5-Ebenen Output-Struktur"""
        return self.path_manager.create_output_structure()

    def get_analysis_level_dir(self, level: int, analysis_type: str = 'static') -> Path:
        """Liefert Pfad für Analyse-Ebene (1-5)"""
        return self.path_manager.get_analysis_level_dir(level, analysis_type)

    def get_summary_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert Summary-Verzeichnis für eine Analyse"""
        return self.path_manager.get_summary_dir(analysis_type)

    def get_pca_analysis_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert PCA Analysis Verzeichnis (Level 5)"""
        return self.path_manager.get_pca_analysis_dir(analysis_type)

    def get_cluster_quality_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert Cluster Quality Verzeichnis (Level 1)"""
        return self.path_manager.get_cluster_quality_dir(analysis_type)

    def get_algorithm_congruence_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert Algorithm Congruence Verzeichnis (Level 2)"""
        return self.path_manager.get_algorithm_congruence_dir(analysis_type)

    def get_external_validation_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert External Validation Verzeichnis (Level 3)"""
        return self.path_manager.get_external_validation_dir(analysis_type)

    def get_company_insights_dir(self, analysis_type: str = 'static') -> Path:
        """Liefert Company Insights Verzeichnis (Level 4)"""
        return self.path_manager.get_company_insights_dir(analysis_type)

    def get_plots_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Plots-Verzeichnis zurück"""
        return self.path_manager.get_plots_dir(analysis_type)

    def get_reports_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Reports-Verzeichnis zurück"""
        return self.path_manager.get_reports_dir(analysis_type)

    def get_comparison_plots_dir(self, comp_type: str) -> Path:
        """Gibt Comparison Plots Dir zurück (direkt in comp_type/)"""
        return self.path_manager.get_comparison_plots_dir(comp_type)

    # =========================================================================
    # FILE WRITER DELEGATION - File I/O Operations
    # =========================================================================

    def save_cluster_data(
        self,
        df: pd.DataFrame,
        cluster_profiles: pd.DataFrame,
        analysis_type: str = 'static',
        metrics: Dict = None
    ):
        """Speichert Cluster-Daten"""
        return self.file_writer.save_cluster_data(df, cluster_profiles, analysis_type, metrics)

    def save_cluster_lists(
        self,
        df: pd.DataFrame,
        analysis_type: str = 'static',
        sort_by: str = 'roa'
    ):
        """Erstellt separate CSV pro Cluster"""
        return self.file_writer.save_cluster_lists(df, analysis_type, sort_by)

    def save_models(
        self,
        scaler,
        model,
        analysis_type: str = 'static',
        pca_model = None
    ):
        """Speichert ML-Modelle"""
        return self.file_writer.save_models(scaler, model, analysis_type, pca_model)

    def save_processed_features(self, df: pd.DataFrame):
        """Speichert verarbeitete Features in 01_data/"""
        return self.file_writer.save_processed_features(df)

    def save_comparison_data(
        self,
        comp_type: str,
        data: pd.DataFrame,
        filename: str
    ):
        """Speichert Comparison-Daten direkt in comp_type/ Ordner"""
        return self.file_writer.save_comparison_data(comp_type, data, filename)

    def create_readme(self):
        """Erstellt README in 99_summary/"""
        return self.file_writer.create_readme()

    def save_enhanced_company_analysis(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        analysis_type: str,
        score_columns: Optional[List[str]] = None,
        dimensional_scores: Optional[pd.DataFrame] = None,
        evolution_data: Optional[pd.DataFrame] = None
    ) -> Path:
        """Saves enhanced multi-sheet Excel file with comprehensive score information"""
        return self.file_writer.save_enhanced_company_analysis(
            df, profiles, analysis_type, score_columns, dimensional_scores, evolution_data
        )

    # =========================================================================
    # DATA FORMATTER DELEGATION - Data Formatting (exposed for advanced usage)
    # =========================================================================

    def _calculate_cluster_averages(
        self,
        df: pd.DataFrame,
        profiles: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """Adds cluster average columns to DataFrame"""
        return self.data_formatter.calculate_cluster_averages(df, profiles, features)

    def _calculate_relative_values(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """Calculates relative values vs cluster average"""
        return self.data_formatter.calculate_relative_values(df, features)

    def _apply_excel_formatting(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        df: pd.DataFrame,
        style: str,
        feature_columns: List[str] = None,
        score_columns: List[str] = None
    ):
        """Applies formatting to Excel sheet"""
        return self.data_formatter.apply_excel_formatting(
            writer, sheet_name, df, style, feature_columns, score_columns
        )

    def _create_cluster_summary_table(
        self,
        df: pd.DataFrame,
        score_column: str
    ) -> pd.DataFrame:
        """Creates cluster summary statistics table"""
        return self.data_formatter.create_cluster_summary_table(df, score_column)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def __repr__(self):
        return f"OutputHandler(market='{self.market}', algorithm='{self.algorithm_name}', mode='{self.mode}')"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("OUTPUT HANDLER TEST - Option B (Refactored)")
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
