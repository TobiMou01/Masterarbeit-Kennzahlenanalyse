"""
Score Analyzer - Analyzes Score Quality and Distributions

Provides:
1. Cluster Homogeneity Analysis - How consistent are scores within clusters
2. Outlier Detection - Companies with unusually low scores
3. Score Distribution Comparison - Statistical comparisons across clusters
4. Score Summary - Overall statistics and rankings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class ScoreAnalyzer:
    """
    Analyzes score quality, distributions, and identifies anomalies

    Use cases:
    - Assess cluster quality via score homogeneity
    - Find misclassified companies (outliers)
    - Compare score distributions across clusters
    - Generate summary statistics and rankings
    """

    def __init__(self):
        """Initialize Score Analyzer"""
        logger.info("✓ ScoreAnalyzer initialized")

    # =========================================================================
    # CLUSTER HOMOGENEITY ANALYSIS
    # =========================================================================

    def analyze_cluster_homogeneity(
        self,
        df: pd.DataFrame,
        score_column: str,
        cluster_column: str = 'cluster'
    ) -> pd.DataFrame:
        """
        Analyzes score homogeneity within clusters

        Homogeneous clusters have:
        - Low standard deviation in scores
        - Low coefficient of variation (CV = std/mean)
        - Small range (max - min)

        Args:
            df: DataFrame with scores and cluster assignments
            score_column: Name of score column to analyze
            cluster_column: Name of cluster column

        Returns:
            DataFrame with homogeneity metrics per cluster:
                - cluster_id
                - count
                - mean_score
                - std_score
                - min_score
                - max_score
                - range
                - coefficient_of_variation
        """
        logger.info(f"\nAnalyzing Cluster Homogeneity...")
        logger.info(f"  Score Column: {score_column}")

        if score_column not in df.columns:
            logger.error(f"  ❌ Score column '{score_column}' not found!")
            return pd.DataFrame()

        homogeneity_data = []

        for cluster_id in sorted(df[cluster_column].unique()):
            if cluster_id < 0:  # Skip noise points
                continue

            cluster_data = df[df[cluster_column] == cluster_id][score_column]

            if len(cluster_data) == 0:
                continue

            mean_score = cluster_data.mean()
            std_score = cluster_data.std()
            min_score = cluster_data.min()
            max_score = cluster_data.max()
            score_range = max_score - min_score

            # Coefficient of Variation (CV = std/mean)
            # Lower CV = more homogeneous
            cv = (std_score / mean_score * 100) if mean_score != 0 else np.inf

            homogeneity_data.append({
                'cluster': cluster_id,
                'count': len(cluster_data),
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'range': score_range,
                'coefficient_of_variation': cv
            })

        homogeneity_df = pd.DataFrame(homogeneity_data)

        logger.info(f"\n  Homogeneity Summary:")
        logger.info(f"    Avg Std Dev: {homogeneity_df['std_score'].mean():.1f}")
        logger.info(f"    Avg CV: {homogeneity_df['coefficient_of_variation'].mean():.1f}%")
        logger.info(f"    Most Homogeneous: Cluster {homogeneity_df.loc[homogeneity_df['std_score'].idxmin(), 'cluster']}")
        logger.info(f"    Least Homogeneous: Cluster {homogeneity_df.loc[homogeneity_df['std_score'].idxmax(), 'cluster']}")

        return homogeneity_df

    # =========================================================================
    # OUTLIER DETECTION
    # =========================================================================

    def identify_outliers(
        self,
        df: pd.DataFrame,
        score_column: str,
        cluster_column: str = 'cluster',
        threshold: float = 30,
        method: str = 'absolute'
    ) -> pd.DataFrame:
        """
        Identifies companies with unusually low scores (potential outliers)

        Two methods:
        - 'absolute': Score < threshold (e.g., < 30)
        - 'iqr': Score < Q1 - 1.5*IQR (within cluster)

        Args:
            df: DataFrame with scores and cluster assignments
            score_column: Name of score column
            cluster_column: Name of cluster column
            threshold: Threshold for absolute method (default: 30)
            method: 'absolute' or 'iqr'

        Returns:
            DataFrame with outliers and diagnostic info:
                - All original columns
                - outlier_reason
                - distance_to_cluster_mean
        """
        logger.info(f"\nIdentifying Score Outliers...")
        logger.info(f"  Method: {method}")
        logger.info(f"  Threshold: {threshold if method == 'absolute' else 'Q1 - 1.5*IQR'}")

        if score_column not in df.columns:
            logger.error(f"  ❌ Score column '{score_column}' not found!")
            return pd.DataFrame()

        outliers_list = []

        if method == 'absolute':
            # Method 1: Simple threshold
            outlier_mask = df[score_column] < threshold

            for idx, row in df[outlier_mask].iterrows():
                cluster_id = row[cluster_column]
                score = row[score_column]

                # Calculate distance to cluster mean
                cluster_mean = df[df[cluster_column] == cluster_id][score_column].mean()
                distance = cluster_mean - score

                outlier_row = row.to_dict()
                outlier_row['outlier_reason'] = f'Score {score:.1f} < {threshold}'
                outlier_row['distance_to_cluster_mean'] = distance

                outliers_list.append(outlier_row)

        elif method == 'iqr':
            # Method 2: IQR method (within cluster)
            for cluster_id in df[cluster_column].unique():
                if cluster_id < 0:
                    continue

                cluster_data = df[df[cluster_column] == cluster_id]
                cluster_scores = cluster_data[score_column]

                # Calculate IQR bounds
                Q1 = cluster_scores.quantile(0.25)
                Q3 = cluster_scores.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR

                # Find outliers
                outlier_mask = cluster_scores < lower_bound

                for idx, row in cluster_data[outlier_mask].iterrows():
                    score = row[score_column]
                    cluster_mean = cluster_scores.mean()
                    distance = cluster_mean - score

                    outlier_row = row.to_dict()
                    outlier_row['outlier_reason'] = f'Score {score:.1f} < {lower_bound:.1f} (Q1 - 1.5*IQR)'
                    outlier_row['distance_to_cluster_mean'] = distance

                    outliers_list.append(outlier_row)

        outliers_df = pd.DataFrame(outliers_list)

        logger.info(f"  ✓ Found {len(outliers_df)} outliers")

        if len(outliers_df) > 0:
            logger.info(f"    Mean Score: {outliers_df[score_column].mean():.1f}")
            logger.info(f"    Mean Distance to Cluster: {outliers_df['distance_to_cluster_mean'].mean():.1f}")

            # Log worst outliers
            worst_outliers = outliers_df.nsmallest(5, score_column)
            logger.info(f"    Worst 5 Outliers:")
            for idx, row in worst_outliers.iterrows():
                company_name = row.get('company_name', row.get('conm', idx))
                logger.info(f"      {company_name}: {row[score_column]:.1f} (Cluster {row[cluster_column]})")

        return outliers_df

    # =========================================================================
    # SCORE DISTRIBUTION COMPARISON
    # =========================================================================

    def compare_score_distributions(
        self,
        df: pd.DataFrame,
        score_column: str,
        cluster_column: str = 'cluster'
    ) -> dict:
        """
        Compares score distributions across clusters

        Statistical measures per cluster:
        - Median, IQR
        - Skewness (symmetry of distribution)
        - Kurtosis (tail heaviness)
        - Number of outliers

        Args:
            df: DataFrame with scores and cluster assignments
            score_column: Name of score column
            cluster_column: Name of cluster column

        Returns:
            Dict with distribution statistics per cluster
        """
        logger.info(f"\nComparing Score Distributions...")

        if score_column not in df.columns:
            logger.error(f"  ❌ Score column '{score_column}' not found!")
            return {}

        distribution_stats = {}

        for cluster_id in sorted(df[cluster_column].unique()):
            if cluster_id < 0:
                continue

            cluster_scores = df[df[cluster_column] == cluster_id][score_column]

            if len(cluster_scores) < 3:  # Need at least 3 points for statistics
                continue

            # Calculate statistics
            median = cluster_scores.median()
            Q1 = cluster_scores.quantile(0.25)
            Q3 = cluster_scores.quantile(0.75)
            IQR = Q3 - Q1

            # Skewness: 0 = symmetric, >0 = right-tailed, <0 = left-tailed
            skewness = cluster_scores.skew()

            # Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails
            kurtosis = cluster_scores.kurtosis()

            # Outliers (IQR method)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = ((cluster_scores < lower_bound) | (cluster_scores > upper_bound)).sum()

            distribution_stats[cluster_id] = {
                'count': len(cluster_scores),
                'median': median,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'outliers_count': outliers_count,
                'outliers_pct': outliers_count / len(cluster_scores) * 100
            }

            logger.info(f"  Cluster {cluster_id}:")
            logger.info(f"    Median: {median:.1f}, IQR: {IQR:.1f}")
            logger.info(f"    Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
            logger.info(f"    Outliers: {outliers_count} ({outliers_count/len(cluster_scores)*100:.1f}%)")

        return distribution_stats

    # =========================================================================
    # SCORE SUMMARY
    # =========================================================================

    def generate_score_summary(
        self,
        df: pd.DataFrame,
        score_columns: List[str],
        cluster_column: str = 'cluster'
    ) -> dict:
        """
        Generates comprehensive score summary

        Includes:
        - Correlation matrix between scores
        - Top 10 / Bottom 10 companies overall
        - Score distribution (histogram data)
        - Overall statistics

        Args:
            df: DataFrame with scores
            score_columns: List of score column names
            cluster_column: Name of cluster column

        Returns:
            Dict with summary statistics
        """
        logger.info(f"\nGenerating Score Summary...")
        logger.info(f"  Score Columns: {score_columns}")

        # Validate score columns
        available_scores = [col for col in score_columns if col in df.columns]
        if len(available_scores) == 0:
            logger.error("  ❌ No valid score columns found!")
            return {}

        summary = {}

        # 1. Correlation Matrix
        if len(available_scores) > 1:
            correlation_matrix = df[available_scores].corr()
            summary['correlation_matrix'] = correlation_matrix.to_dict()
            logger.info(f"\n  Score Correlations:")
            for i, col1 in enumerate(available_scores):
                for col2 in available_scores[i+1:]:
                    corr = correlation_matrix.loc[col1, col2]
                    logger.info(f"    {col1} ↔ {col2}: {corr:.3f}")

        # 2. Overall Statistics (for first score column)
        primary_score = available_scores[0]
        summary['primary_score'] = primary_score
        summary['overall_stats'] = {
            'mean': df[primary_score].mean(),
            'median': df[primary_score].median(),
            'std': df[primary_score].std(),
            'min': df[primary_score].min(),
            'max': df[primary_score].max(),
            'Q1': df[primary_score].quantile(0.25),
            'Q3': df[primary_score].quantile(0.75)
        }

        logger.info(f"\n  Overall Statistics ({primary_score}):")
        logger.info(f"    Mean: {summary['overall_stats']['mean']:.1f}")
        logger.info(f"    Median: {summary['overall_stats']['median']:.1f}")
        logger.info(f"    Std: {summary['overall_stats']['std']:.1f}")
        logger.info(f"    Range: [{summary['overall_stats']['min']:.1f}, {summary['overall_stats']['max']:.1f}]")

        # 3. Top 10 / Bottom 10 Companies
        top_10 = df.nlargest(10, primary_score)
        bottom_10 = df.nsmallest(10, primary_score)

        summary['top_10'] = top_10[[col for col in ['gvkey', 'company_name', 'conm', cluster_column] + available_scores if col in df.columns]].to_dict('records')
        summary['bottom_10'] = bottom_10[[col for col in ['gvkey', 'company_name', 'conm', cluster_column] + available_scores if col in df.columns]].to_dict('records')

        logger.info(f"\n  Top 10 Companies:")
        for i, row in enumerate(top_10.head(5).itertuples(), 1):
            company_name = getattr(row, 'company_name', getattr(row, 'conm', 'Unknown'))
            score = getattr(row, primary_score)
            logger.info(f"    {i}. {company_name}: {score:.1f}")

        # 4. Distribution (Histogram data)
        hist, bin_edges = np.histogram(df[primary_score], bins=10, range=(0, 100))
        summary['distribution_histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }

        # 5. Per-Cluster Summary
        cluster_summary = {}
        for cluster_id in sorted(df[cluster_column].unique()):
            if cluster_id < 0:
                continue

            cluster_data = df[df[cluster_column] == cluster_id]
            cluster_summary[cluster_id] = {
                'count': len(cluster_data),
                'mean_score': cluster_data[primary_score].mean(),
                'median_score': cluster_data[primary_score].median(),
                'std_score': cluster_data[primary_score].std()
            }

        summary['cluster_summary'] = cluster_summary

        logger.info(f"\n  ✓ Score Summary generated successfully")

        return summary

    # =========================================================================
    # ADDITIONAL ANALYSIS
    # =========================================================================

    def find_misclassified_candidates(
        self,
        df: pd.DataFrame,
        features: List[str],
        score_column: str,
        cluster_column: str = 'cluster',
        low_score_threshold: float = 30
    ) -> pd.DataFrame:
        """
        Finds companies that might be misclassified

        Criteria for misclassification:
        - Low proximity score (< threshold)
        - Closer to a different cluster center

        Args:
            df: DataFrame with scores, features, and cluster assignments
            features: List of features used for clustering
            score_column: Name of score column (proximity_score)
            cluster_column: Name of cluster column
            low_score_threshold: Threshold for low scores

        Returns:
            DataFrame with potential misclassifications and suggested cluster
        """
        logger.info(f"\nFinding Misclassified Candidates...")
        logger.info(f"  Low Score Threshold: {low_score_threshold}")

        # Find companies with low scores
        low_score_companies = df[df[score_column] < low_score_threshold].copy()

        if len(low_score_companies) == 0:
            logger.info("  ✓ No low-score companies found")
            return pd.DataFrame()

        logger.info(f"  Found {len(low_score_companies)} low-score companies")

        # Calculate cluster centers
        available_features = [f for f in features if f in df.columns]
        if len(available_features) == 0:
            logger.warning("  ⚠️  No features available for distance calculation")
            return low_score_companies

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[available_features].fillna(0))

        # Calculate cluster centers
        cluster_centers = {}
        for cluster_id in df[cluster_column].unique():
            if cluster_id >= 0:
                cluster_mask = df[cluster_column] == cluster_id
                cluster_centers[cluster_id] = X_scaled[cluster_mask].mean(axis=0)

        # For each low-score company, find closest cluster
        suggested_clusters = []
        for idx, row in low_score_companies.iterrows():
            current_cluster = row[cluster_column]
            company_idx = df.index.get_loc(idx)
            company_vector = X_scaled[company_idx].reshape(1, -1)

            # Calculate distances to all clusters
            distances = {}
            for cluster_id, center in cluster_centers.items():
                center_vector = center.reshape(1, -1)
                distance = euclidean_distances(company_vector, center_vector)[0, 0]
                distances[cluster_id] = distance

            # Find closest cluster
            closest_cluster = min(distances, key=distances.get)
            closest_distance = distances[closest_cluster]
            current_distance = distances.get(current_cluster, np.inf)

            suggested_clusters.append({
                'suggested_cluster': closest_cluster,
                'current_distance': current_distance,
                'suggested_distance': closest_distance,
                'improvement': current_distance - closest_distance
            })

        # Add suggestions to dataframe
        suggestion_df = pd.DataFrame(suggested_clusters, index=low_score_companies.index)
        result = pd.concat([low_score_companies, suggestion_df], axis=1)

        # Filter to only real improvements
        result = result[result['suggested_cluster'] != result[cluster_column]]
        result = result[result['improvement'] > 0.5]  # Significant improvement

        logger.info(f"  ✓ Found {len(result)} potential misclassifications")

        if len(result) > 0:
            logger.info(f"    Top 5 Candidates:")
            for idx, row in result.nlargest(5, 'improvement').iterrows():
                company_name = row.get('company_name', row.get('conm', idx))
                logger.info(f"      {company_name}: Cluster {row[cluster_column]} → {row['suggested_cluster']} "
                           f"(improvement: {row['improvement']:.2f})")

        return result


if __name__ == "__main__":
    # Test ScoreAnalyzer
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("SCORE ANALYZER TEST")
    print("="*80)

    # Create mock data
    np.random.seed(42)
    n_companies = 100

    mock_df = pd.DataFrame({
        'gvkey': [f'{i:03d}' for i in range(n_companies)],
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'cluster': np.random.choice([0, 1, 2, 3], n_companies),
        'proximity_score': np.random.normal(60, 20, n_companies),
        'profitability_score': np.random.normal(55, 18, n_companies),
        'overall_score': np.random.normal(58, 19, n_companies),
        'roa': np.random.normal(0.1, 0.05, n_companies),
        'roe': np.random.normal(0.15, 0.08, n_companies)
    })

    # Clip scores to [0, 100]
    for col in ['proximity_score', 'profitability_score', 'overall_score']:
        mock_df[col] = np.clip(mock_df[col], 0, 100)

    # Test Analyzer
    analyzer = ScoreAnalyzer()

    # Test 1: Cluster Homogeneity
    print("\n1. Cluster Homogeneity:")
    homogeneity = analyzer.analyze_cluster_homogeneity(mock_df, 'proximity_score', 'cluster')
    print(homogeneity.head())

    # Test 2: Outlier Detection
    print("\n2. Outlier Detection:")
    outliers = analyzer.identify_outliers(mock_df, 'proximity_score', 'cluster', threshold=30)
    print(f"Found {len(outliers)} outliers")

    # Test 3: Distribution Comparison
    print("\n3. Distribution Comparison:")
    distributions = analyzer.compare_score_distributions(mock_df, 'proximity_score', 'cluster')

    # Test 4: Score Summary
    print("\n4. Score Summary:")
    summary = analyzer.generate_score_summary(
        mock_df,
        ['proximity_score', 'profitability_score', 'overall_score'],
        'cluster'
    )
    print(f"Summary Keys: {summary.keys()}")

    print("\n✓ ScoreAnalyzer Test erfolgreich!")
    print("="*80 + "\n")
