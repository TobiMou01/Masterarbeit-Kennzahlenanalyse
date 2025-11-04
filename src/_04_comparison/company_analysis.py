"""
Company-Level Cluster Analysis - Extended Multi-Sheet Version
Generates comprehensive Excel workbook with company insights, cluster analysis, and score evolution
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

logger = logging.getLogger(__name__)


class CompanyClusterAnalyzer:
    """
    Erweiterte Company-Level Cluster Analysis

    Features:
    - Overview Sheet mit allen Unternehmen, Features, Cluster-Durchschnitten und Scores
    - Separate Sheets pro Cluster mit Ranking
    - Score Evolution Sheet (bei combined analysis)
    - Erweiterte Summary-Statistiken
    """

    def __init__(self, algorithm_results: Dict, market: str = 'germany'):
        """
        Initialize analyzer

        Args:
            algorithm_results: Dict mit Ergebnissen aller Algorithmen
            market: Market-Bezeichnung
        """
        self.algorithm_results = algorithm_results
        self.market = market
        self.base_features = self._get_base_features()
        self.score_columns = [
            'proximity_score',
            'profitability_score',
            'leverage_score',
            'efficiency_score',
            'growth_score',
            'relative_score',
            'overall_score'
        ]

    def _get_base_features(self) -> List[str]:
        """Get base features for analysis"""
        return [
            'roa', 'roe', 'ebit_margin', 'debt_to_equity',
            'current_ratio', 'asset_turnover', 'revenue_growth',
            'net_profit_margin', 'quick_ratio', 'interest_coverage',
            'fcf_margin', 'asset_growth'
        ]

    def _extract_base_data(self) -> pd.DataFrame:
        """
        Extract base company data from first available dataset

        Returns:
            DataFrame with gvkey, company_name, sector info
        """
        logger.info("→ Extracting base company data...")

        base_df = None
        for algo_name, results in self.algorithm_results.items():
            if results.get('static') and 'df' in results['static']:
                df = results['static']['df']

                # Extract base columns
                base_cols = ['gvkey', 'conm', 'gsector', 'ggroup', 'gind', 'gsubind']
                available_cols = [col for col in base_cols if col in df.columns]

                base_df = df[available_cols].copy()
                base_df = base_df.rename(columns={'conm': 'company_name'})
                base_df = base_df.drop_duplicates(subset=['gvkey'])
                break

        if base_df is None:
            raise ValueError("❌ No data found for company analysis")

        logger.info(f"  ✓ Found {len(base_df)} companies")
        return base_df

    def _get_algorithm_data(self, algo_name: str, stage: str) -> Optional[pd.DataFrame]:
        """
        Get data for specific algorithm and stage

        Args:
            algo_name: Algorithm name (kmeans, hierarchical, dbscan)
            stage: Stage name (static, dynamic, combined)

        Returns:
            DataFrame or None if not available
        """
        if algo_name not in self.algorithm_results:
            return None

        results = self.algorithm_results[algo_name]
        if stage not in results or 'df' not in results[stage]:
            return None

        return results[stage]['df'].copy()

    def create_overview_dataframe(self, algo_name: str, stage: str) -> pd.DataFrame:
        """
        Create comprehensive overview DataFrame with all features and scores

        Args:
            algo_name: Algorithm name
            stage: Stage name

        Returns:
            DataFrame with overview data
        """
        logger.info(f"\n→ Creating Overview DataFrame ({algo_name} - {stage})...")

        # Get algorithm data
        df = self._get_algorithm_data(algo_name, stage)
        if df is None:
            logger.warning(f"  ⚠️  No data available for {algo_name} - {stage}")
            return None

        # Start with base info - handle both 'conm' and 'company_name'
        base_cols = ['gvkey', 'cluster']

        # Check for company name column
        if 'conm' in df.columns:
            base_cols.append('conm')
            result_df = df[base_cols].copy()
            result_df = result_df.rename(columns={'conm': 'company_name'})
        elif 'company_name' in df.columns:
            base_cols.append('company_name')
            result_df = df[base_cols].copy()
        else:
            # Fallback: just use gvkey and cluster
            logger.warning("  ⚠️  No company name column found, using only gvkey")
            result_df = df[base_cols].copy()
            result_df['company_name'] = result_df['gvkey']  # Use gvkey as name

        # Add cluster name
        result_df['cluster_name'] = result_df['cluster'].apply(
            lambda x: f"Cluster_{x}" if x >= 0 else "Noise"
        )

        # Add original features
        logger.info("  → Adding original features...")
        available_features = [f for f in self.base_features if f in df.columns]
        for feature in available_features:
            result_df[feature] = df[feature]

        logger.info(f"    ✓ Added {len(available_features)} features")

        # Calculate cluster averages
        logger.info("  → Calculating cluster averages...")
        cluster_avgs = {}

        for cluster_id in result_df['cluster'].unique():
            if cluster_id < 0:  # Skip noise
                continue

            cluster_mask = result_df['cluster'] == cluster_id
            cluster_data = result_df[cluster_mask]

            for feature in available_features:
                avg_col = f"{feature}_cluster_avg"
                cluster_avg = cluster_data[feature].mean()

                if avg_col not in result_df.columns:
                    result_df[avg_col] = np.nan

                result_df.loc[cluster_mask, avg_col] = cluster_avg

        logger.info(f"    ✓ Calculated averages for {len(available_features)} features")

        # Calculate relative values (absolute difference)
        logger.info("  → Calculating relative values (absolute)...")
        for feature in available_features:
            avg_col = f"{feature}_cluster_avg"
            rel_col = f"{feature}_vs_cluster"

            if avg_col in result_df.columns:
                result_df[rel_col] = result_df[feature] - result_df[avg_col]

        # Calculate relative values (percentage)
        logger.info("  → Calculating relative values (%)...")
        for feature in available_features:
            avg_col = f"{feature}_cluster_avg"
            pct_col = f"{feature}_rel_pct"

            if avg_col in result_df.columns:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    result_df[pct_col] = (
                        (result_df[feature] / result_df[avg_col] - 1) * 100
                    )
                    # Replace inf and -inf with NaN
                    result_df[pct_col] = result_df[pct_col].replace([np.inf, -np.inf], np.nan)

        # Add all scores
        logger.info("  → Adding scores...")
        available_scores = [s for s in self.score_columns if s in df.columns]
        for score in available_scores:
            result_df[score] = df[score]

        logger.info(f"    ✓ Added {len(available_scores)} score columns")

        # Sort by overall_score (descending) if available
        if 'overall_score' in result_df.columns:
            result_df = result_df.sort_values('overall_score', ascending=False)

        logger.info(f"  ✓ Overview DataFrame created: {len(result_df)} companies, {len(result_df.columns)} columns")

        return result_df

    def create_cluster_dataframes(self, overview_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Create separate DataFrames for each cluster with ranking

        Args:
            overview_df: Overview DataFrame

        Returns:
            Dict mapping cluster_id to cluster DataFrame
        """
        logger.info("\n→ Creating cluster-specific DataFrames...")

        cluster_dfs = {}

        for cluster_id in sorted(overview_df['cluster'].unique()):
            if cluster_id < 0:  # Skip noise
                continue

            # Filter cluster data
            cluster_df = overview_df[overview_df['cluster'] == cluster_id].copy()

            # Add cluster rank based on overall_score
            if 'overall_score' in cluster_df.columns:
                cluster_df['cluster_rank'] = cluster_df['overall_score'].rank(
                    method='min', ascending=False
                ).astype(int)

                # Sort by rank
                cluster_df = cluster_df.sort_values('cluster_rank')

            cluster_dfs[cluster_id] = cluster_df
            logger.info(f"  ✓ Cluster {cluster_id}: {len(cluster_df)} companies")

        return cluster_dfs

    def create_score_evolution_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Create Score Evolution DataFrame comparing static, dynamic, combined scores

        Only applicable when all three stages are available

        Returns:
            DataFrame with score evolution or None
        """
        logger.info("\n→ Creating Score Evolution DataFrame...")

        # Check if we have all three stages for kmeans
        if 'kmeans' not in self.algorithm_results:
            logger.info("  ⚠️  KMeans not available, skipping Score Evolution")
            return None

        kmeans = self.algorithm_results['kmeans']
        stages = ['static', 'dynamic', 'combined']

        # Check all stages available
        if not all(stage in kmeans and 'df' in kmeans[stage] for stage in stages):
            logger.info("  ⚠️  Not all stages available, skipping Score Evolution")
            return None

        # Get data from each stage
        static_source = kmeans['static']['df']

        # Handle both 'conm' and 'company_name'
        if 'conm' in static_source.columns:
            static_df = static_source[['gvkey', 'conm', 'cluster']].copy()
            static_df = static_df.rename(columns={'conm': 'company_name', 'cluster': 'static_cluster'})
        elif 'company_name' in static_source.columns:
            static_df = static_source[['gvkey', 'company_name', 'cluster']].copy()
            static_df = static_df.rename(columns={'cluster': 'static_cluster'})
        else:
            static_df = static_source[['gvkey', 'cluster']].copy()
            static_df = static_df.rename(columns={'cluster': 'static_cluster'})
            static_df['company_name'] = static_df['gvkey']

        dynamic_df = kmeans['dynamic']['df'][['gvkey', 'cluster']].copy()
        dynamic_df = dynamic_df.rename(columns={'cluster': 'dynamic_cluster'})

        combined_df = kmeans['combined']['df'][['gvkey', 'cluster']].copy()
        combined_df = combined_df.rename(columns={'cluster': 'combined_cluster'})

        # Merge all stages
        evolution_df = static_df.merge(dynamic_df, on='gvkey', how='outer')
        evolution_df = evolution_df.merge(combined_df, on='gvkey', how='outer')

        # Add scores from each stage
        score_cols = ['proximity_score', 'overall_score']

        for stage in stages:
            stage_df = kmeans[stage]['df']
            for score in score_cols:
                if score in stage_df.columns:
                    col_name = f"{stage}_{score}"
                    score_data = stage_df.set_index('gvkey')[score]
                    evolution_df[col_name] = evolution_df['gvkey'].map(score_data)

        # Calculate score changes
        logger.info("  → Calculating score changes...")

        # Static -> Dynamic
        if 'static_overall_score' in evolution_df.columns and 'dynamic_overall_score' in evolution_df.columns:
            evolution_df['score_change_static_dynamic'] = (
                evolution_df['dynamic_overall_score'] - evolution_df['static_overall_score']
            )

        # Dynamic -> Combined
        if 'dynamic_overall_score' in evolution_df.columns and 'combined_overall_score' in evolution_df.columns:
            evolution_df['score_change_dynamic_combined'] = (
                evolution_df['combined_overall_score'] - evolution_df['dynamic_overall_score']
            )

        # Static -> Combined (total change)
        if 'static_overall_score' in evolution_df.columns and 'combined_overall_score' in evolution_df.columns:
            evolution_df['score_change_total'] = (
                evolution_df['combined_overall_score'] - evolution_df['static_overall_score']
            )

        # Cluster changes
        evolution_df['cluster_changed'] = (
            (evolution_df['static_cluster'] != evolution_df['dynamic_cluster']) |
            (evolution_df['dynamic_cluster'] != evolution_df['combined_cluster']) |
            (evolution_df['static_cluster'] != evolution_df['combined_cluster'])
        )

        # Evolution pattern
        def determine_pattern(row):
            if pd.isna(row.get('score_change_total')):
                return 'unknown'

            total_change = row['score_change_total']

            if total_change > 10:
                return 'strong_improvement'
            elif total_change > 5:
                return 'moderate_improvement'
            elif total_change > -5:
                return 'stable'
            elif total_change > -10:
                return 'moderate_decline'
            else:
                return 'strong_decline'

        evolution_df['evolution_pattern'] = evolution_df.apply(determine_pattern, axis=1)

        # Sort by total score change (descending)
        if 'score_change_total' in evolution_df.columns:
            evolution_df = evolution_df.sort_values('score_change_total', ascending=False)

        logger.info(f"  ✓ Score Evolution DataFrame created: {len(evolution_df)} companies")

        return evolution_df

    def create_summary_dataframe(self, overview_df: pd.DataFrame,
                                 cluster_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Create enhanced summary statistics

        Args:
            overview_df: Overview DataFrame
            cluster_dfs: Dict of cluster DataFrames

        Returns:
            Summary DataFrame
        """
        logger.info("\n→ Creating Summary DataFrame...")

        summary_data = []

        # Overall statistics
        summary_data.append({
            'Category': 'General',
            'Metric': 'Total Companies',
            'Value': len(overview_df)
        })

        summary_data.append({
            'Category': 'General',
            'Metric': 'Number of Clusters',
            'Value': len(cluster_dfs)
        })

        # Noise points
        noise_count = (overview_df['cluster'] == -1).sum()
        if noise_count > 0:
            summary_data.append({
                'Category': 'General',
                'Metric': 'Noise Points',
                'Value': noise_count
            })

        # Cluster sizes
        for cluster_id, cluster_df in sorted(cluster_dfs.items()):
            summary_data.append({
                'Category': f'Cluster {cluster_id}',
                'Metric': 'Size',
                'Value': len(cluster_df)
            })

        # Score statistics
        if 'overall_score' in overview_df.columns:
            summary_data.append({
                'Category': 'Scores',
                'Metric': 'Overall Score (Mean)',
                'Value': f"{overview_df['overall_score'].mean():.2f}"
            })

            summary_data.append({
                'Category': 'Scores',
                'Metric': 'Overall Score (Std)',
                'Value': f"{overview_df['overall_score'].std():.2f}"
            })

            # Top 10 companies
            top_10 = overview_df.nlargest(10, 'overall_score')[['company_name', 'overall_score']]
            for idx, row in enumerate(top_10.itertuples(), 1):
                summary_data.append({
                    'Category': 'Top 10 Companies',
                    'Metric': f"{idx}. {row.company_name}",
                    'Value': f"{row.overall_score:.2f}"
                })

            # Bottom 10 companies
            bottom_10 = overview_df.nsmallest(10, 'overall_score')[['company_name', 'overall_score']]
            for idx, row in enumerate(bottom_10.itertuples(), 1):
                summary_data.append({
                    'Category': 'Bottom 10 Companies',
                    'Metric': f"{idx}. {row.company_name}",
                    'Value': f"{row.overall_score:.2f}"
                })

        summary_df = pd.DataFrame(summary_data)
        logger.info(f"  ✓ Summary DataFrame created: {len(summary_df)} rows")

        return summary_df

    def save_to_excel(self, output_path: Path, algo_name: str, stage: str):
        """
        Save all DataFrames to multi-sheet Excel file

        Args:
            output_path: Path for output Excel file
            algo_name: Algorithm name
            stage: Stage name
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"CREATING MULTI-SHEET EXCEL: {algo_name} - {stage}")
        logger.info(f"{'='*80}")

        # Create overview DataFrame
        overview_df = self.create_overview_dataframe(algo_name, stage)
        if overview_df is None:
            logger.error("❌ Failed to create overview DataFrame")
            return None

        # Create cluster DataFrames
        cluster_dfs = self.create_cluster_dataframes(overview_df)

        # Create score evolution (if applicable)
        evolution_df = None
        if stage == 'combined':
            evolution_df = self.create_score_evolution_dataframe()

        # Create summary
        summary_df = self.create_summary_dataframe(overview_df, cluster_dfs)

        # Write to Excel
        logger.info(f"\n→ Writing to Excel: {output_path}")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Overview
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            logger.info(f"  ✓ Sheet 'Overview': {len(overview_df)} rows, {len(overview_df.columns)} columns")

            # Sheets 2-N: Cluster sheets
            for cluster_id, cluster_df in sorted(cluster_dfs.items()):
                sheet_name = f"Cluster_{cluster_id}"
                cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"  ✓ Sheet '{sheet_name}': {len(cluster_df)} rows")

            # Sheet N+1: Score Evolution (if applicable)
            if evolution_df is not None:
                evolution_df.to_excel(writer, sheet_name='Score_Evolution', index=False)
                logger.info(f"  ✓ Sheet 'Score_Evolution': {len(evolution_df)} rows")

            # Sheet N+2: Summary
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            logger.info(f"  ✓ Sheet 'Summary': {len(summary_df)} rows")

        logger.info(f"\n✓ Excel file created successfully: {output_path}")
        logger.info(f"{'='*80}\n")

        return str(output_path)


def create_company_cluster_excel(
    algorithm_results: Dict,
    market: str = 'germany',
    output_dir: str = None
) -> List[str]:
    """
    Erstellt erweiterte Multi-Sheet Excel-Dateien für Company Cluster Analysis

    Creates separate Excel files for each algorithm and stage combination

    Args:
        algorithm_results: Dict mit Ergebnissen aller Algorithmen
        market: Market-Bezeichnung
        output_dir: Output-Verzeichnis (optional)

    Returns:
        List of paths to created Excel files
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING COMPANY-LEVEL CLUSTER ANALYSIS")
    logger.info("=" * 80)

    # Output path
    if output_dir is None:
        output_dir = f'output/{market}'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create analyzer
    analyzer = CompanyClusterAnalyzer(algorithm_results, market)

    created_files = []

    # Create Excel for each algorithm + stage combination
    # For kmeans: create separate files for static, dynamic, combined
    if 'kmeans' in algorithm_results:
        kmeans_results = algorithm_results['kmeans']

        for stage in ['static', 'dynamic', 'combined']:
            if stage in kmeans_results and 'df' in kmeans_results[stage]:
                excel_file = output_path / f'company_cluster_analysis_kmeans_{stage}.xlsx'
                result = analyzer.save_to_excel(excel_file, 'kmeans', stage)
                if result:
                    created_files.append(result)

    # For hierarchical and dbscan: create single file (uses static as master)
    for algo_name in ['hierarchical', 'dbscan']:
        if algo_name in algorithm_results:
            results = algorithm_results[algo_name]

            if 'static' in results and 'df' in results['static']:
                excel_file = output_path / f'company_cluster_analysis_{algo_name}.xlsx'
                result = analyzer.save_to_excel(excel_file, algo_name, 'static')
                if result:
                    created_files.append(result)

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Company Cluster Analysis Complete")
    logger.info(f"  Created {len(created_files)} Excel files")
    for file in created_files:
        logger.info(f"    • {file}")
    logger.info("=" * 80 + "\n")

    return created_files


if __name__ == "__main__":
    # Test function
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("COMPANY ANALYSIS MODULE TEST")
    print("=" * 80)
    print("\nThis module needs to be called from ComparisonPipeline with actual results.")
    print("See ComparisonPipeline.run_full_comparison_pipeline() for integration.")
    print("=" * 80)
