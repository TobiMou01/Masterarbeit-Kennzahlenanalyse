"""
Score Integration Module
Provides scoring functionality for clustering pipelines
"""

import logging
import pandas as pd
from typing import Dict, Tuple, Optional

from src._04_scoring.score_calculator import ScoreCalculator

logger = logging.getLogger(__name__)


def apply_scoring(
    df: pd.DataFrame,
    features: list,
    cluster_column: str,
    profiles: pd.DataFrame,
    analysis_type: str,
    config: dict,
    scoring_enabled: bool = True
) -> pd.DataFrame:
    """
    Apply scoring to clustered data
    
    Integrates ScoreCalculator to add proximity, dimensional, and relative scores
    
    Args:
        df: DataFrame with cluster assignments
        features: List of features used for clustering
        cluster_column: Name of cluster column
        profiles: Cluster profiles (means/medians)
        analysis_type: Type of analysis ('static', 'dynamic', 'combined', 'unified')
        config: Configuration dictionary
        scoring_enabled: Whether scoring is enabled
        
    Returns:
        DataFrame with added score columns
    """
    if not scoring_enabled:
        logger.info(f"  Scoring disabled, skipping...")
        return df
    
    logger.info(f"\n  📊 Applying Scoring ({analysis_type})...")

    try:
        # Initialize score calculator (no config needed, uses FeatureSelector internally)
        calculator = ScoreCalculator()
        
        # Calculate all scores
        df_scored = calculator.calculate_all_scores(
            df=df,
            features=features,
            cluster_column=cluster_column,
            profiles=profiles
        )
        
        # Log score statistics
        if 'composite_score' in df_scored.columns:
            score_stats = df_scored['composite_score'].describe()
            logger.info(f"     ✓ Scores calculated:")
            logger.info(f"       Mean: {score_stats['mean']:.3f}, Median: {score_stats['50%']:.3f}")
            logger.info(f"       Range: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]")
        
        return df_scored
        
    except Exception as e:
        logger.error(f"     ❌ Scoring failed: {e}", exc_info=True)
        return df


def track_score_evolution(
    df_static: Optional[pd.DataFrame],
    df_dynamic: Optional[pd.DataFrame],
    df_combined: Optional[pd.DataFrame],
    config: dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Track score evolution across analysis stages
    
    Analyzes how scores change from static → dynamic → combined
    
    Args:
        df_static: Static analysis results with scores
        df_dynamic: Dynamic analysis results with scores
        df_combined: Combined analysis results with scores
        config: Configuration dictionary
        
    Returns:
        Tuple of (evolution_df, evolution_stats)
    """
    logger.info(f"\n  📈 Tracking Score Evolution...")
    
    try:
        from src._04_scoring.score_evolution import ScoreEvolutionTracker

        # Initialize evolution tracker
        tracker = ScoreEvolutionTracker()
        
        # Prepare data
        stage_dfs = {}
        if df_static is not None and 'composite_score' in df_static.columns:
            stage_dfs['static'] = df_static
        if df_dynamic is not None and 'composite_score' in df_dynamic.columns:
            stage_dfs['dynamic'] = df_dynamic
        if df_combined is not None and 'composite_score' in df_combined.columns:
            stage_dfs['combined'] = df_combined
        
        if len(stage_dfs) < 2:
            logger.warning(f"     ⚠️  Need at least 2 stages for evolution tracking")
            return pd.DataFrame(), {}
        
        # Track evolution
        evolution_df, stats = tracker.analyze_evolution(stage_dfs)
        
        # Log key findings
        if stats:
            logger.info(f"     ✓ Evolution tracked:")
            if 'score_changes' in stats:
                changes = stats['score_changes']
                logger.info(f"       Average change: {changes.get('mean_change', 0):.3f}")
                logger.info(f"       Companies improved: {changes.get('improved_pct', 0):.1f}%")
        
        return evolution_df, stats
        
    except Exception as e:
        logger.error(f"     ❌ Score evolution tracking failed: {e}", exc_info=True)
        return pd.DataFrame(), {}
