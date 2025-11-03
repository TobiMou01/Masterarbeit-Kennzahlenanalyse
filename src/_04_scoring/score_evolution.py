"""
Score Evolution Tracker - Tracks Score Changes Across Analysis Phases

Tracks how scores evolve from:
- Static Analysis (current state)
- Dynamic Analysis (trends)
- Combined Analysis (integrated view)

Provides pattern classification and biggest movers identification.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ScoreEvolutionTracker:
    """
    Tracks score evolution across Static → Dynamic → Combined analysis phases

    Pattern Classification:
    - "Consistent Excellence": High scores in both static and dynamic
    - "Eroding Position": High static but declining in dynamic
    - "Improving Trend": Low static but improving in dynamic
    - "Challenged": Low combined score
    - "Mixed": Everything else
    """

    def __init__(self):
        """Initialize Score Evolution Tracker"""
        logger.info("✓ ScoreEvolutionTracker initialized")

    # =========================================================================
    # EVOLUTION TRACKING
    # =========================================================================

    def track_evolution(
        self,
        df_static: pd.DataFrame,
        df_dynamic: pd.DataFrame,
        df_combined: pd.DataFrame,
        score_column: str = 'proximity_score',
        company_id: str = 'gvkey'
    ) -> pd.DataFrame:
        """
        Tracks score evolution across 3 analysis phases

        Args:
            df_static: DataFrame from static analysis with scores
            df_dynamic: DataFrame from dynamic analysis with scores
            df_combined: DataFrame from combined analysis with scores
            score_column: Name of score column to track
            company_id: Column name for company identifier

        Returns:
            DataFrame with columns:
                - gvkey, company_name
                - static_score, dynamic_score, combined_score
                - score_change_static_to_dynamic
                - score_change_static_to_combined
                - total_score_change
        """
        logger.info(f"\nTracking Score Evolution...")
        logger.info(f"  Score Column: {score_column}")

        # Extract relevant columns from each phase
        static_scores = self._extract_scores(df_static, score_column, company_id, 'static')
        dynamic_scores = self._extract_scores(df_dynamic, score_column, company_id, 'dynamic')
        combined_scores = self._extract_scores(df_combined, score_column, company_id, 'combined')

        # Find common companies across all phases
        common_companies = set(static_scores.index).intersection(
            set(dynamic_scores.index)
        ).intersection(
            set(combined_scores.index)
        )

        logger.info(f"  Companies in static: {len(static_scores)}")
        logger.info(f"  Companies in dynamic: {len(dynamic_scores)}")
        logger.info(f"  Companies in combined: {len(combined_scores)}")
        logger.info(f"  Common companies: {len(common_companies)}")

        if len(common_companies) == 0:
            logger.warning("  ⚠️  No common companies found across all phases!")
            return pd.DataFrame()

        # Create evolution dataframe
        evolution_df = pd.DataFrame(index=list(common_companies))
        evolution_df.index.name = company_id

        # Add company name if available
        if 'company_name' in df_static.columns:
            name_map = df_static.set_index(company_id)['company_name'].to_dict()
            evolution_df['company_name'] = evolution_df.index.map(name_map)
        elif 'conm' in df_static.columns:
            name_map = df_static.set_index(company_id)['conm'].to_dict()
            evolution_df['company_name'] = evolution_df.index.map(name_map)

        # Add scores
        evolution_df['static_score'] = static_scores
        evolution_df['dynamic_score'] = dynamic_scores
        evolution_df['combined_score'] = combined_scores

        # Calculate changes
        evolution_df['score_change_static_to_dynamic'] = (
            evolution_df['dynamic_score'] - evolution_df['static_score']
        )
        evolution_df['score_change_static_to_combined'] = (
            evolution_df['combined_score'] - evolution_df['static_score']
        )
        evolution_df['total_score_change'] = (
            evolution_df['combined_score'] - evolution_df['static_score']
        )

        # Add absolute change for sorting
        evolution_df['abs_total_change'] = evolution_df['total_score_change'].abs()

        logger.info(f"\n  Score Changes:")
        logger.info(f"    Static → Dynamic: Mean={evolution_df['score_change_static_to_dynamic'].mean():.1f}")
        logger.info(f"    Static → Combined: Mean={evolution_df['score_change_static_to_combined'].mean():.1f}")
        logger.info(f"    Max Improvement: {evolution_df['total_score_change'].max():.1f}")
        logger.info(f"    Max Decline: {evolution_df['total_score_change'].min():.1f}")

        return evolution_df

    def _extract_scores(
        self,
        df: pd.DataFrame,
        score_column: str,
        company_id: str,
        phase_name: str
    ) -> pd.Series:
        """Extracts scores from a dataframe for a specific phase"""
        if score_column not in df.columns:
            logger.warning(f"  ⚠️  {score_column} not found in {phase_name} data")
            return pd.Series(dtype=float)

        scores = df.set_index(company_id)[score_column]
        return scores

    # =========================================================================
    # PATTERN CLASSIFICATION
    # =========================================================================

    def classify_evolution_pattern(
        self,
        evolution_df: pd.DataFrame,
        excellence_threshold: float = 70,
        challenged_threshold: float = 40,
        improvement_threshold: float = 20
    ) -> pd.Series:
        """
        Classifies evolution patterns based on score combinations

        Patterns:
        - "Consistent Excellence": static > 70 AND dynamic > 70
        - "Eroding Position": static > 70 AND dynamic < 50
        - "Improving Trend": static < 50 AND (combined - static) > 20
        - "Challenged": combined < 40
        - "Mixed": Everything else

        Args:
            evolution_df: DataFrame from track_evolution()
            excellence_threshold: Threshold for high scores (default: 70)
            challenged_threshold: Threshold for low scores (default: 40)
            improvement_threshold: Minimum improvement for "Improving" (default: 20)

        Returns:
            pd.Series with pattern labels
        """
        logger.info(f"\nClassifying Evolution Patterns...")

        patterns = pd.Series(index=evolution_df.index, dtype=str)

        for idx, row in evolution_df.iterrows():
            static_score = row['static_score']
            dynamic_score = row['dynamic_score']
            combined_score = row['combined_score']
            total_change = row['total_score_change']

            # Pattern 1: Consistent Excellence
            if static_score >= excellence_threshold and dynamic_score >= excellence_threshold:
                pattern = "Consistent Excellence"

            # Pattern 2: Eroding Position
            elif static_score >= excellence_threshold and dynamic_score < 50:
                pattern = "Eroding Position"

            # Pattern 3: Improving Trend
            elif static_score < 50 and total_change >= improvement_threshold:
                pattern = "Improving Trend"

            # Pattern 4: Challenged
            elif combined_score < challenged_threshold:
                pattern = "Challenged"

            # Pattern 5: Mixed (default)
            else:
                pattern = "Mixed"

            patterns[idx] = pattern

        # Log pattern distribution
        pattern_counts = patterns.value_counts()
        logger.info(f"\n  Pattern Distribution:")
        for pattern, count in pattern_counts.items():
            pct = count / len(patterns) * 100
            logger.info(f"    {pattern}: {count} ({pct:.1f}%)")

        return patterns

    # =========================================================================
    # BIGGEST MOVERS
    # =========================================================================

    def get_biggest_movers(
        self,
        evolution_df: pd.DataFrame,
        top_n: int = 10,
        direction: str = 'both'
    ) -> pd.DataFrame:
        """
        Identifies companies with biggest score changes

        Args:
            evolution_df: DataFrame from track_evolution()
            top_n: Number of companies to return
            direction: 'both', 'improvers', or 'decliners'

        Returns:
            DataFrame with biggest movers, sorted by absolute change
        """
        logger.info(f"\nIdentifying Biggest Movers (top {top_n})...")

        if direction == 'improvers':
            # Top improvers (positive change)
            movers = evolution_df.nlargest(top_n, 'total_score_change')
            logger.info(f"  Top {top_n} Improvers:")

        elif direction == 'decliners':
            # Top decliners (negative change)
            movers = evolution_df.nsmallest(top_n, 'total_score_change')
            logger.info(f"  Top {top_n} Decliners:")

        else:  # 'both'
            # Biggest absolute changes
            movers = evolution_df.nlargest(top_n, 'abs_total_change')
            logger.info(f"  Top {top_n} Biggest Changes (absolute):")

        # Log results
        for idx, row in movers.iterrows():
            company_name = row.get('company_name', idx)
            change = row['total_score_change']
            logger.info(f"    {company_name}: {change:+.1f} points "
                       f"({row['static_score']:.1f} → {row['combined_score']:.1f})")

        return movers

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze_score_stability(
        self,
        evolution_df: pd.DataFrame
    ) -> dict:
        """
        Analyzes overall score stability across phases

        Args:
            evolution_df: DataFrame from track_evolution()

        Returns:
            Dict with stability metrics
        """
        logger.info(f"\nAnalyzing Score Stability...")

        # Calculate correlations between phases
        corr_static_dynamic = evolution_df['static_score'].corr(evolution_df['dynamic_score'])
        corr_static_combined = evolution_df['static_score'].corr(evolution_df['combined_score'])
        corr_dynamic_combined = evolution_df['dynamic_score'].corr(evolution_df['combined_score'])

        # Calculate volatility (std of changes)
        volatility = evolution_df['total_score_change'].std()

        # Percentage of stable companies (change < 10 points)
        stable_companies = (evolution_df['abs_total_change'] < 10).sum()
        stable_pct = stable_companies / len(evolution_df) * 100

        stability_metrics = {
            'correlation_static_dynamic': corr_static_dynamic,
            'correlation_static_combined': corr_static_combined,
            'correlation_dynamic_combined': corr_dynamic_combined,
            'volatility_std': volatility,
            'stable_companies_count': stable_companies,
            'stable_companies_pct': stable_pct,
            'mean_change': evolution_df['total_score_change'].mean(),
            'median_change': evolution_df['total_score_change'].median()
        }

        logger.info(f"  Correlations:")
        logger.info(f"    Static ↔ Dynamic: {corr_static_dynamic:.3f}")
        logger.info(f"    Static ↔ Combined: {corr_static_combined:.3f}")
        logger.info(f"    Dynamic ↔ Combined: {corr_dynamic_combined:.3f}")
        logger.info(f"  Volatility (Std): {volatility:.1f}")
        logger.info(f"  Stable Companies (<10pts change): {stable_companies} ({stable_pct:.1f}%)")

        return stability_metrics

    def get_evolution_summary(
        self,
        evolution_df: pd.DataFrame,
        patterns: pd.Series
    ) -> dict:
        """
        Generates comprehensive evolution summary

        Args:
            evolution_df: DataFrame from track_evolution()
            patterns: Series from classify_evolution_pattern()

        Returns:
            Dict with summary statistics
        """
        summary = {
            'total_companies': len(evolution_df),
            'mean_static_score': evolution_df['static_score'].mean(),
            'mean_dynamic_score': evolution_df['dynamic_score'].mean(),
            'mean_combined_score': evolution_df['combined_score'].mean(),
            'mean_total_change': evolution_df['total_score_change'].mean(),
            'std_total_change': evolution_df['total_score_change'].std(),
            'max_improvement': evolution_df['total_score_change'].max(),
            'max_decline': evolution_df['total_score_change'].min(),
            'pattern_distribution': patterns.value_counts().to_dict(),
            'improving_companies': (evolution_df['total_score_change'] > 0).sum(),
            'declining_companies': (evolution_df['total_score_change'] < 0).sum(),
            'stable_companies': (evolution_df['abs_total_change'] < 5).sum()
        }

        return summary


if __name__ == "__main__":
    # Test ScoreEvolutionTracker
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("SCORE EVOLUTION TRACKER TEST")
    print("="*80)

    # Create mock evolution data
    np.random.seed(42)
    n_companies = 50

    gvkeys = [f'{i:03d}' for i in range(n_companies)]

    # Simulate 3 phases with some correlation
    static_scores = np.random.normal(60, 15, n_companies)
    # Dynamic scores correlated with static but with noise
    dynamic_scores = static_scores + np.random.normal(0, 10, n_companies)
    # Combined is average with some adjustments
    combined_scores = (static_scores + dynamic_scores) / 2 + np.random.normal(0, 5, n_companies)

    # Clip to [0, 100]
    static_scores = np.clip(static_scores, 0, 100)
    dynamic_scores = np.clip(dynamic_scores, 0, 100)
    combined_scores = np.clip(combined_scores, 0, 100)

    # Create mock dataframes
    df_static = pd.DataFrame({
        'gvkey': gvkeys,
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'proximity_score': static_scores
    })

    df_dynamic = pd.DataFrame({
        'gvkey': gvkeys,
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'proximity_score': dynamic_scores
    })

    df_combined = pd.DataFrame({
        'gvkey': gvkeys,
        'company_name': [f'Company_{i}' for i in range(n_companies)],
        'proximity_score': combined_scores
    })

    # Test Tracker
    tracker = ScoreEvolutionTracker()

    # Test evolution tracking
    evolution = tracker.track_evolution(df_static, df_dynamic, df_combined)
    print(f"\nEvolution DataFrame Shape: {evolution.shape}")
    print(f"Columns: {evolution.columns.tolist()}")

    # Test pattern classification
    patterns = tracker.classify_evolution_pattern(evolution)
    print(f"\nPatterns: {patterns.value_counts().to_dict()}")

    # Test biggest movers
    improvers = tracker.get_biggest_movers(evolution, top_n=5, direction='improvers')
    decliners = tracker.get_biggest_movers(evolution, top_n=5, direction='decliners')

    # Test stability analysis
    stability = tracker.analyze_score_stability(evolution)
    print(f"\nStability Metrics:")
    print(f"  Volatility: {stability['volatility_std']:.1f}")
    print(f"  Stable Companies: {stability['stable_companies_pct']:.1f}%")

    print("\n✓ ScoreEvolutionTracker Test erfolgreich!")
    print("="*80 + "\n")
