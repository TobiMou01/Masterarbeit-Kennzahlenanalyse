"""
Path Manager Module
Handles all path generation and directory structure management for OutputHandler
"""

from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PathManager:
    """
    Manages all path generation and directory structure

    Responsible for:
    - Creating directory structures
    - Providing path objects for various output locations
    - Managing directory hierarchy
    """

    def __init__(
        self,
        market: str,
        algorithm: str,
        mode: str,
        base_dir: str = 'output'
    ):
        """
        Initialize PathManager

        Args:
            market: Market designation (germany, usa, etc.)
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan')
            mode: 'comparative' or 'hierarchical'
            base_dir: Base directory for output
        """
        self.market = market
        self.algorithm = algorithm
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

    def get_plots_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Plots-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'plots'

    def get_reports_dir(self, analysis_type: str = 'static') -> Path:
        """Gibt Reports-Verzeichnis zurück"""
        analysis_name = self.analysis_types[analysis_type]
        return self.algorithm_dir / analysis_name / 'reports'

    def get_comparison_plots_dir(self, comp_type: str) -> Path:
        """Gibt Comparison Plots Dir zurück (direkt in comp_type/)"""
        # Plots direkt in comp_type/ speichern (keine plots/ Unterordner)
        comp_dir = self.comparisons_dir / comp_type
        comp_dir.mkdir(parents=True, exist_ok=True)
        return comp_dir
