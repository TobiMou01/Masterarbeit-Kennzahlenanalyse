"""
Research Excel Coordinator - Main orchestrator for creating research Excel file

Coordinates all section writers to create a comprehensive Excel analysis file
aligned with the 4 core research questions:
1. Homogenität (Homogeneity) - Internal cluster quality
2. Kongruenz (Congruence) - Agreement with classifications
3. Treiber (Drivers) - Key financial ratios
4. Stabilität & Kontext (Stability & Context) - Temporal & size effects
"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional
from openpyxl import Workbook

# Import comparison analyzers
from src._04_comparison.gics_analyzer import GICSComparison
from src._04_comparison.algorithm_analyzer import AlgorithmComparison
from src._04_comparison.feature_analyzer import FeatureImportance
from src._03_clustering.cluster_naming import ClusterNamer

# Import section writers
from .section_0_config_writer import Section0ConfigWriter
from .section_1_homogeneity_writer import Section1HomogeneityWriter
from .section_2_congruence_writer import Section2CongruenceWriter
from .section_3_drivers_writer import Section3DriversWriter
from .section_4_stability_writer import Section4StabilityWriter

logger = logging.getLogger(__name__)


class ResearchExcelCoordinator:
    """
    Main coordinator for creating research-oriented Excel file

    Structure:
    - Section 0: Config & Overview (1 sheet)
    - Section 1: Homogenität (4 sheets)
    - Section 2: Kongruenz (3 sheets)
    - Section 3: Treiber (3 sheets)
    - Section 4: Stabilität (3 sheets)
    """

    def __init__(self, algorithm_results: Dict, market: str = 'germany', config_path: str = 'config.yaml'):
        """
        Initialize research Excel coordinator

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

        # Initialize comparison analyzers
        self.gics_analyzer = GICSComparison()
        self.algo_analyzer = AlgorithmComparison()
        self.feature_analyzer = FeatureImportance()
        self.cluster_namer = ClusterNamer()

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
        section0_writer = Section0ConfigWriter(self.config, self.colors, self.score_columns)
        section0_writer.create_section(wb, overview_df)
        logger.info("  ✓ Section 0 complete")

        # Section 1: Homogenität
        logger.info("\n→ Step 5: Creating Section 1 (Homogenität)...")
        section1_writer = Section1HomogeneityWriter(self.config, self.colors, self.score_columns, self.market)
        section1_writer.create_section(wb, overview_df)
        logger.info("  ✓ Section 1 complete")

        # Section 2: Kongruenz
        logger.info("\n→ Step 6: Creating Section 2 (Kongruenz)...")
        section2_writer = Section2CongruenceWriter(self.config, self.colors, self.score_columns)
        section2_writer.create_section(wb, overview_df)
        logger.info("  ✓ Section 2 complete")

        # Section 3: Treiber
        logger.info("\n→ Step 7: Creating Section 3 (Treiber)...")
        section3_writer = Section3DriversWriter(
            self.config, self.colors, self.score_columns, self.market,
            self.feature_analyzer, self.cluster_namer
        )
        section3_writer.create_section(wb, overview_df)
        logger.info("  ✓ Section 3 complete")

        # Section 4: Stabilität
        logger.info("\n→ Step 8: Creating Section 4 (Stabilität)...")
        section4_writer = Section4StabilityWriter(self.config, self.colors, self.score_columns)
        section4_writer.create_section(wb, overview_df)
        logger.info("  ✓ Section 4 complete")

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
    coordinator = ResearchExcelCoordinator(algorithm_results, market=market)
    output_path = output_dir / 'research_analysis_master.xlsx'
    return coordinator.create_research_excel(output_path)
