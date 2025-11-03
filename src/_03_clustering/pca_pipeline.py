"""
PCA Pipeline - Parallel Clustering Analysis
Performs clustering in both original and PCA space for comparison
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from src._02_preprocessing.pca_transformer import PCATransformer
from src._03_clustering.cluster_engine import ClusteringEngine
from src._03_clustering.cluster_naming import ClusterNamer
from src._01_setup.feature_selector import FeatureSelector
from src._01_setup.output_handler import OutputHandler
from src._01_setup import config_loader as config

logger = logging.getLogger(__name__)


class PCAPipeline:
    """
    PCA-based clustering pipeline for extended feature sets

    Workflow:
    1. Run clustering in original space (base features)
    2. Run clustering in PCA space (extended features)
    3. Compare results (ARI, confusion matrix)
    4. Map PCA profiles back to original features
    5. Save comprehensive comparison
    """

    def __init__(
        self,
        config_dict: dict,
        market: str,
        algorithm: str = 'kmeans',
        n_components: float = 0.85,
        features_config_path: str = 'features_config.yaml'
    ):
        """
        Initialize PCA Pipeline

        Args:
            config_dict: Configuration dictionary
            market: Market name (e.g., 'germany')
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan')
            n_components: PCA components (int or variance threshold)
            features_config_path: Path to features_config.yaml
        """
        self.config = config_dict
        self.market = market
        self.algorithm = algorithm
        self.n_components = n_components

        # Initialize components
        self.feature_selector = FeatureSelector(config_path=features_config_path)
        self.pca_transformer = PCATransformer(n_components=n_components)
        self.cluster_engine = ClusteringEngine(config_dict=config_dict)
        self.output = OutputHandler(market=market, algorithm=algorithm)
        self.namer = ClusterNamer(feature_selector=self.feature_selector)

        # Results storage
        self.results = {}

        logger.info(f"✓ PCAPipeline initialized (n_components={n_components})")

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run_parallel_analysis(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5,
        analysis_type: str = 'static'
    ) -> Dict:
        """
        Run parallel clustering analysis in original and PCA space

        Args:
            df: DataFrame with all features
            n_clusters: Number of clusters
            analysis_type: 'static', 'dynamic', or 'combined'

        Returns:
            Dictionary with comprehensive results
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔬 PCA PARALLEL ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"  Market: {self.market}")
        logger.info(f"  Algorithm: {self.algorithm}")
        logger.info(f"  Analysis Type: {analysis_type}")
        logger.info(f"  Companies: {len(df)}")
        logger.info(f"  Target Clusters: {n_clusters}\n")

        # 1. Clustering in original space (base features)
        logger.info("📊 Step 1/4: Clustering in Original Space")
        df_original, profiles_original, metrics_original = self._run_clustering_in_original_space(
            df, n_clusters, analysis_type
        )

        # 2. Clustering in PCA space (extended features)
        logger.info("\n📊 Step 2/4: Clustering in PCA Space")
        df_pca, profiles_pca, metrics_pca, pca_metadata = self._run_clustering_in_pca_space(
            df, n_clusters, analysis_type
        )

        # 3. Compare clusterings
        logger.info("\n📊 Step 3/4: Comparing Clusterings")
        comparison = self._compare_clusterings(df_original, df_pca)

        # 4. Map PCA profiles to original features
        logger.info("\n📊 Step 4/4: Mapping PCA Profiles to Original Features")
        profiles_pca_mapped = self._map_pca_profiles_to_original_features(
            profiles_pca, pca_metadata
        )

        # Store results
        self.results = {
            'original': {
                'df': df_original,
                'profiles': profiles_original,
                'metrics': metrics_original,
                'n_features': len(metrics_original['features'])
            },
            'pca': {
                'df': df_pca,
                'profiles': profiles_pca,
                'profiles_mapped': profiles_pca_mapped,
                'metrics': metrics_pca,
                'metadata': pca_metadata,
                'n_components': self.pca_transformer.pca.n_components_,
                'variance_explained': self.pca_transformer.pca.explained_variance_ratio_.sum()
            },
            'comparison': comparison,
            'config': {
                'n_clusters': n_clusters,
                'algorithm': self.algorithm,
                'analysis_type': analysis_type
            }
        }

        # 5. Save results
        logger.info("\n📊 Saving PCA Analysis Results...")
        self.save_pca_results(analysis_type)

        # Summary
        self._print_summary()

        return self.results

    # =========================================================================
    # CLUSTERING METHODS
    # =========================================================================

    def _run_clustering_in_original_space(
        self,
        df: pd.DataFrame,
        n_clusters: int,
        analysis_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run clustering using base features (12) in original space

        Args:
            df: DataFrame with features
            n_clusters: Number of clusters
            analysis_type: Analysis type

        Returns:
            Tuple of (df_with_clusters, profiles, metrics)
        """
        # Get base features (12)
        base_features = self.feature_selector.get_base_features()

        # Validate features exist
        available_features = [f for f in base_features if f in df.columns]

        if len(available_features) < len(base_features):
            missing = set(base_features) - set(available_features)
            logger.warning(f"  ⚠️  Missing features: {missing}")

        logger.info(f"  Features: {len(available_features)} (base set)")

        # Run clustering
        df_result, profiles, metrics = self.cluster_engine.perform_clustering(
            df=df,
            features=available_features,
            n_clusters=n_clusters,
            analysis_type=analysis_type
        )

        # Store features in metrics
        metrics['features'] = available_features

        logger.info(f"  ✓ Clustering complete")
        logger.info(f"    Clusters: {n_clusters}")
        logger.info(f"    Silhouette: {metrics.get('silhouette_score', 0):.3f}")

        return df_result, profiles, metrics

    def _run_clustering_in_pca_space(
        self,
        df: pd.DataFrame,
        n_clusters: int,
        analysis_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Run clustering using extended features (25+) in PCA space

        Workflow:
        1. Get extended features (25+)
        2. Apply PCA transformation
        3. Run clustering on PCA components
        4. Return results

        Args:
            df: DataFrame with features
            n_clusters: Number of clusters
            analysis_type: Analysis type

        Returns:
            Tuple of (df_with_clusters, profiles, metrics, pca_metadata)
        """
        # Get extended features (25+)
        extended_features = self.feature_selector.get_extended_features()

        # Validate features exist
        available_features = [f for f in extended_features if f in df.columns]

        if len(available_features) < len(extended_features):
            missing = set(extended_features) - set(available_features)
            logger.warning(f"  ⚠️  Missing features: {missing}")

        logger.info(f"  Features: {len(available_features)} (extended set)")

        # Apply PCA transformation
        X_pca, variance_summary = self.pca_transformer.fit_transform(
            df=df,
            features=available_features
        )

        n_components = self.pca_transformer.pca.n_components_
        variance_explained = self.pca_transformer.pca.explained_variance_ratio_.sum()

        logger.info(f"  ✓ PCA transformation complete")
        logger.info(f"    {len(available_features)} features → {n_components} components")
        logger.info(f"    Variance explained: {variance_explained:.1%}")

        # Create DataFrame with PCA components
        df_pca = df[['gvkey']].copy()

        # Add company name if available
        if 'company_name' in df.columns:
            df_pca['company_name'] = df['company_name']
        elif 'conm' in df.columns:
            df_pca['company_name'] = df['conm']

        # Add PCA components
        pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
        for i, col in enumerate(pca_feature_names):
            df_pca[col] = X_pca[:, i]

        # Run clustering on PCA components
        df_result, profiles, metrics = self.cluster_engine.perform_clustering(
            df=df_pca,
            features=pca_feature_names,
            n_clusters=n_clusters,
            analysis_type=analysis_type
        )

        # Store PCA metadata
        pca_metadata = {
            'original_features': available_features,
            'n_components': n_components,
            'variance_explained': variance_explained,
            'variance_summary': variance_summary,
            'component_loadings': self.pca_transformer.get_component_loadings(),
            'component_interpretation': self.pca_transformer.interpret_components(top_n=5)
        }

        # Store in metrics
        metrics['features'] = pca_feature_names
        metrics['pca_metadata'] = pca_metadata

        logger.info(f"  ✓ Clustering complete")
        logger.info(f"    Clusters: {n_clusters}")
        logger.info(f"    Silhouette: {metrics.get('silhouette_score', 0):.3f}")

        return df_result, profiles, metrics, pca_metadata

    # =========================================================================
    # COMPARISON METHODS
    # =========================================================================

    def _compare_clusterings(
        self,
        df_original: pd.DataFrame,
        df_pca: pd.DataFrame
    ) -> Dict:
        """
        Compare clustering results between original and PCA space

        Computes:
        - Adjusted Rand Index (ARI)
        - Confusion Matrix
        - Cluster size differences
        - Agreement statistics

        Args:
            df_original: Results from original space
            df_pca: Results from PCA space

        Returns:
            Dictionary with comparison metrics
        """
        # Merge on gvkey to align companies
        df_merged = df_original[['gvkey', 'cluster']].merge(
            df_pca[['gvkey', 'cluster']],
            on='gvkey',
            suffixes=('_original', '_pca')
        )

        # Calculate ARI
        ari = adjusted_rand_score(
            df_merged['cluster_original'],
            df_merged['cluster_pca']
        )

        logger.info(f"  Adjusted Rand Index (ARI): {ari:.3f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(
            df_merged['cluster_original'],
            df_merged['cluster_pca']
        )

        # Convert to DataFrame for easier interpretation
        original_clusters = sorted(df_merged['cluster_original'].unique())
        pca_clusters = sorted(df_merged['cluster_pca'].unique())

        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f'Original_C{c}' for c in original_clusters],
            columns=[f'PCA_C{c}' for c in pca_clusters]
        )

        # Agreement statistics
        total_companies = len(df_merged)
        perfect_agreement = (df_merged['cluster_original'] == df_merged['cluster_pca']).sum()
        agreement_rate = perfect_agreement / total_companies

        logger.info(f"  Perfect Agreement: {perfect_agreement}/{total_companies} ({agreement_rate:.1%})")

        # Cluster size comparison
        size_original = df_merged['cluster_original'].value_counts().sort_index()
        size_pca = df_merged['cluster_pca'].value_counts().sort_index()

        size_comparison = pd.DataFrame({
            'Original': size_original,
            'PCA': size_pca,
            'Difference': size_pca - size_original
        })

        # Find best cluster mapping (Hungarian algorithm approximation)
        cluster_mapping = self._find_best_cluster_mapping(conf_matrix_df)

        return {
            'ari': ari,
            'confusion_matrix': conf_matrix_df,
            'agreement_rate': agreement_rate,
            'perfect_agreement': perfect_agreement,
            'total_companies': total_companies,
            'size_comparison': size_comparison,
            'cluster_mapping': cluster_mapping,
            'df_merged': df_merged
        }

    def _find_best_cluster_mapping(self, conf_matrix: pd.DataFrame) -> Dict[int, int]:
        """
        Find best 1:1 mapping between original and PCA clusters

        Uses greedy approach: assign each PCA cluster to the original cluster
        with the highest overlap.

        Args:
            conf_matrix: Confusion matrix (original x PCA)

        Returns:
            Dict mapping PCA cluster → Original cluster
        """
        mapping = {}
        conf_array = conf_matrix.values

        # Extract cluster IDs from column names (e.g., 'PCA_C0' -> 0)
        pca_clusters = [int(col.split('_C')[1]) for col in conf_matrix.columns]
        original_clusters = [int(idx.split('_C')[1]) for idx in conf_matrix.index]

        # For each PCA cluster, find best match in original space
        for pca_idx, pca_cluster in enumerate(pca_clusters):
            # Find original cluster with highest overlap
            best_original_idx = conf_array[:, pca_idx].argmax()
            best_original_cluster = original_clusters[best_original_idx]
            overlap = conf_array[best_original_idx, pca_idx]

            mapping[pca_cluster] = {
                'best_match': best_original_cluster,
                'overlap': int(overlap),
                'total_in_pca': int(conf_array[:, pca_idx].sum())
            }

        return mapping

    def _map_pca_profiles_to_original_features(
        self,
        profiles_pca: pd.DataFrame,
        pca_metadata: Dict
    ) -> pd.DataFrame:
        """
        Map PCA cluster profiles back to original feature space

        Uses inverse PCA transformation to interpret what each PCA cluster
        "means" in terms of original financial ratios.

        Args:
            profiles_pca: Cluster profiles in PCA space (PC1, PC2, ...)
            pca_metadata: PCA metadata with loadings

        Returns:
            DataFrame with profiles in original feature space
        """
        # Use PCATransformer's inverse transform
        profiles_mapped = self.pca_transformer.map_pca_profiles_to_original(profiles_pca)

        logger.info(f"  ✓ Profiles mapped to {len(profiles_mapped.columns)} original features")

        return profiles_mapped

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================

    def save_pca_results(self, analysis_type: str):
        """
        Save all PCA analysis results to 5_pca_analysis/ directory

        Saves:
        - Original space: clusters, profiles
        - PCA space: clusters, profiles, profiles_mapped
        - Comparison: ARI, confusion matrix, cluster mapping
        - PCA metadata: loadings, variance explained
        - Visualizations (if enabled)

        Args:
            analysis_type: Analysis type ('static', 'dynamic', 'combined')
        """
        # Get PCA analysis directory
        pca_dir = self.output.get_pca_analysis_dir(analysis_type)

        # Create subdirectories
        original_dir = pca_dir / 'original_space'
        pca_space_dir = pca_dir / 'pca_space'
        comparison_dir = pca_dir / 'comparison'
        metadata_dir = pca_dir / 'metadata'

        for d in [original_dir, pca_space_dir, comparison_dir, metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Save original space results
        self.results['original']['df'].to_csv(
            original_dir / f'{analysis_type}_clusters.csv', index=False
        )
        self.results['original']['profiles'].to_csv(
            original_dir / f'{analysis_type}_profiles.csv'
        )

        # 2. Save PCA space results
        self.results['pca']['df'].to_csv(
            pca_space_dir / f'{analysis_type}_clusters_pca.csv', index=False
        )
        self.results['pca']['profiles'].to_csv(
            pca_space_dir / f'{analysis_type}_profiles_pca.csv'
        )
        self.results['pca']['profiles_mapped'].to_csv(
            pca_space_dir / f'{analysis_type}_profiles_mapped.csv'
        )

        # 3. Save comparison results
        comparison = self.results['comparison']

        # ARI and summary
        summary = pd.DataFrame([{
            'ARI': comparison['ari'],
            'Agreement_Rate': comparison['agreement_rate'],
            'Perfect_Agreement': comparison['perfect_agreement'],
            'Total_Companies': comparison['total_companies'],
            'N_Clusters': self.results['config']['n_clusters'],
            'Algorithm': self.algorithm
        }])
        summary.to_csv(comparison_dir / 'ari_summary.csv', index=False)

        # Confusion matrix
        comparison['confusion_matrix'].to_csv(
            comparison_dir / 'confusion_matrix.csv'
        )

        # Cluster mapping
        mapping_df = pd.DataFrame.from_dict(
            comparison['cluster_mapping'], orient='index'
        )
        mapping_df.index.name = 'PCA_Cluster'
        mapping_df.to_csv(comparison_dir / 'cluster_mapping.csv')

        # Size comparison
        comparison['size_comparison'].to_csv(
            comparison_dir / 'cluster_size_comparison.csv'
        )

        # Company-level comparison
        comparison['df_merged'].to_csv(
            comparison_dir / 'company_cluster_assignments.csv', index=False
        )

        # 4. Save PCA metadata
        pca_meta = self.results['pca']['metadata']

        # Variance explained
        pca_meta['variance_summary'].to_csv(
            metadata_dir / 'variance_explained.csv', index=False
        )

        # Component loadings
        pca_meta['component_loadings'].to_csv(
            metadata_dir / 'component_loadings.csv'
        )

        # Component interpretation
        interpretation_data = []
        for pc, features in pca_meta['component_interpretation'].items():
            for rank, (feature, loading) in enumerate(features, 1):
                interpretation_data.append({
                    'Component': pc,
                    'Rank': rank,
                    'Feature': feature,
                    'Loading': loading
                })

        interpretation_df = pd.DataFrame(interpretation_data)
        interpretation_df.to_csv(
            metadata_dir / 'component_interpretation.csv', index=False
        )

        # Original features list
        pd.DataFrame({
            'Feature': pca_meta['original_features']
        }).to_csv(metadata_dir / 'original_features.csv', index=False)

        # 5. Save comprehensive Excel report
        self._save_excel_report(pca_dir, analysis_type)

        logger.info(f"\n  ✓ Results saved to: {pca_dir}/")
        logger.info(f"    ├── original_space/")
        logger.info(f"    ├── pca_space/")
        logger.info(f"    ├── comparison/")
        logger.info(f"    ├── metadata/")
        logger.info(f"    └── pca_analysis_report.xlsx")

    def _save_excel_report(self, pca_dir: Path, analysis_type: str):
        """
        Create comprehensive Excel report with all PCA analysis results

        Sheets:
        1. Summary
        2. Original Clusters
        3. PCA Clusters
        4. PCA Profiles (Mapped)
        5. Comparison
        6. Component Loadings
        7. Variance Explained

        Args:
            pca_dir: PCA analysis directory
            analysis_type: Analysis type
        """
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows

        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        # 1. Summary Sheet
        ws_summary = wb.create_sheet("Summary")
        summary_data = [
            ["PCA CLUSTERING ANALYSIS REPORT"],
            [""],
            ["Configuration", ""],
            ["Market", self.market],
            ["Algorithm", self.algorithm],
            ["Analysis Type", analysis_type],
            ["N Clusters", self.results['config']['n_clusters']],
            [""],
            ["Original Space", ""],
            ["N Features", self.results['original']['n_features']],
            ["Silhouette Score", f"{self.results['original']['metrics'].get('silhouette_score', 0):.3f}"],
            [""],
            ["PCA Space", ""],
            ["N Original Features", len(self.results['pca']['metadata']['original_features'])],
            ["N Components", self.results['pca']['n_components']],
            ["Variance Explained", f"{self.results['pca']['variance_explained']:.1%}"],
            ["Silhouette Score", f"{self.results['pca']['metrics'].get('silhouette_score', 0):.3f}"],
            [""],
            ["Comparison", ""],
            ["Adjusted Rand Index", f"{self.results['comparison']['ari']:.3f}"],
            ["Agreement Rate", f"{self.results['comparison']['agreement_rate']:.1%}"],
            ["Perfect Agreement", f"{self.results['comparison']['perfect_agreement']}/{self.results['comparison']['total_companies']}"]
        ]

        for row in summary_data:
            ws_summary.append(row)

        # Format summary
        ws_summary['A1'].font = Font(size=14, bold=True)
        for row in [3, 9, 13, 19]:
            ws_summary[f'A{row}'].font = Font(bold=True)

        # 2. Original Clusters
        ws_orig = wb.create_sheet("Original Clusters")
        for r in dataframe_to_rows(self.results['original']['profiles'], index=True, header=True):
            ws_orig.append(r)

        # 3. PCA Clusters
        ws_pca = wb.create_sheet("PCA Clusters")
        for r in dataframe_to_rows(self.results['pca']['profiles'], index=True, header=True):
            ws_pca.append(r)

        # 4. PCA Profiles (Mapped)
        ws_mapped = wb.create_sheet("PCA Profiles (Mapped)")
        for r in dataframe_to_rows(self.results['pca']['profiles_mapped'], index=True, header=True):
            ws_mapped.append(r)

        # 5. Comparison
        ws_comp = wb.create_sheet("Comparison")
        ws_comp.append(["Confusion Matrix"])
        ws_comp.append([])
        for r in dataframe_to_rows(self.results['comparison']['confusion_matrix'], index=True, header=True):
            ws_comp.append(r)

        ws_comp.append([])
        ws_comp.append(["Cluster Size Comparison"])
        ws_comp.append([])
        for r in dataframe_to_rows(self.results['comparison']['size_comparison'], index=True, header=True):
            ws_comp.append(r)

        # 6. Component Loadings
        ws_loadings = wb.create_sheet("Component Loadings")
        for r in dataframe_to_rows(
            self.results['pca']['metadata']['component_loadings'],
            index=True,
            header=True
        ):
            ws_loadings.append(r)

        # 7. Variance Explained
        ws_variance = wb.create_sheet("Variance Explained")
        for r in dataframe_to_rows(
            self.results['pca']['metadata']['variance_summary'],
            index=False,
            header=True
        ):
            ws_variance.append(r)

        # Save workbook
        excel_path = pca_dir / 'pca_analysis_report.xlsx'
        wb.save(excel_path)

        logger.info(f"  ✓ Excel report saved: {excel_path.name}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _print_summary(self):
        """Print comprehensive summary of PCA analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("PCA ANALYSIS COMPLETE")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("✓ PCA Parallel Analysis Complete")
        print("=" * 80)

        print("\n📊 ORIGINAL SPACE")
        print(f"   Features: {self.results['original']['n_features']}")
        print(f"   Silhouette: {self.results['original']['metrics'].get('silhouette_score', 0):.3f}")

        print("\n📊 PCA SPACE")
        print(f"   Original Features: {len(self.results['pca']['metadata']['original_features'])}")
        print(f"   PCA Components: {self.results['pca']['n_components']}")
        print(f"   Variance Explained: {self.results['pca']['variance_explained']:.1%}")
        print(f"   Silhouette: {self.results['pca']['metrics'].get('silhouette_score', 0):.3f}")

        print("\n📊 COMPARISON")
        print(f"   Adjusted Rand Index: {self.results['comparison']['ari']:.3f}")
        print(f"   Agreement Rate: {self.results['comparison']['agreement_rate']:.1%}")

        # Interpretation
        if self.results['comparison']['ari'] > 0.7:
            interpretation = "Very similar clusterings"
        elif self.results['comparison']['ari'] > 0.5:
            interpretation = "Moderately similar clusterings"
        elif self.results['comparison']['ari'] > 0.3:
            interpretation = "Somewhat different clusterings"
        else:
            interpretation = "Very different clusterings"

        print(f"   Interpretation: {interpretation}")

        print("\n📂 Output:")
        print(f"   {self.output.get_pca_analysis_dir()}/")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test PCAPipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("PCA PIPELINE TEST")
    print("=" * 80)

    # Mock configuration
    mock_config = {
        'features': {
            'base_features': ['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'current_ratio'],
            'extended_features': ['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'current_ratio',
                                 'net_profit_margin', 'quick_ratio', 'asset_turnover',
                                 'interest_coverage', 'fcf_margin']
        },
        'clustering': {
            'algorithm': 'kmeans'
        }
    }

    # Create mock data
    np.random.seed(42)
    n_samples = 200

    # Base features (12)
    base_features = ['roa', 'roe', 'ebit_margin', 'net_profit_margin',
                     'debt_to_equity', 'current_ratio', 'quick_ratio',
                     'asset_turnover', 'interest_coverage', 'fcf_margin',
                     'revenue_growth', 'asset_growth']

    # Extended features (25) - add more
    extended_features = base_features + [
        'gross_margin', 'operating_margin', 'roc',
        'cash_ratio', 'working_capital_ratio',
        'equity_ratio', 'debt_ratio', 'net_debt_to_ebitda',
        'revenue_per_employee', 'days_sales_outstanding', 'capital_intensity',
        'capex_to_revenue', 'reinvestment_rate'
    ]

    # Generate correlated data
    base_data = np.random.randn(n_samples, 3)

    df_data = {}
    df_data['gvkey'] = [f'G{i:05d}' for i in range(n_samples)]
    df_data['company_name'] = [f'Company {i}' for i in range(n_samples)]

    # Create features with some correlation structure
    for i, feat in enumerate(extended_features):
        if i < 5:  # Profitability - correlated
            df_data[feat] = base_data[:, 0] * 5 + np.random.randn(n_samples) * 2
        elif i < 10:  # Leverage - correlated
            df_data[feat] = base_data[:, 1] * 3 + np.random.randn(n_samples) * 1.5
        else:  # Others - more independent
            df_data[feat] = base_data[:, 2] * 2 + np.random.randn(n_samples)

    df_mock = pd.DataFrame(df_data)

    print(f"\nMock data created:")
    print(f"  Companies: {len(df_mock)}")
    print(f"  Features: {len(extended_features)}")

    # Test PCAPipeline
    print("\n" + "-" * 80)
    print("Testing PCAPipeline")
    print("-" * 80)

    try:
        pipeline = PCAPipeline(
            config_dict=mock_config,
            market='test_market',
            algorithm='kmeans',
            n_components=0.85
        )

        results = pipeline.run_parallel_analysis(
            df=df_mock,
            n_clusters=4,
            analysis_type='static'
        )

        print("\n✓ PCAPipeline test successful!")
        print(f"\nResults keys: {list(results.keys())}")
        print(f"ARI: {results['comparison']['ari']:.3f}")
        print(f"Agreement Rate: {results['comparison']['agreement_rate']:.1%}")

    except Exception as e:
        print(f"\n❌ PCAPipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80 + "\n")
