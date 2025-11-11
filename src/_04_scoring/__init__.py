"""
Scoring System for Clustering Analysis

Provides comprehensive scoring mechanisms:
1. Proximity Score - Distance to cluster center (0-100)
2. Dimensional Scores - Per category (Profitability, Leverage, Efficiency, Growth)
3. Relative Score - Performance vs. cluster average (Z-Score based)
4. Score Evolution - Tracking across Static → Dynamic → Combined
5. Score Analysis - Homogeneity, Outlier Detection, Pattern Classification
"""

from src._04_scoring.score_calculator import ScoreCalculator
from src._04_scoring.score_evolution import ScoreEvolutionTracker
from src._04_scoring.score_analyzer import ScoreAnalyzer
from src._04_scoring.score_integrator import apply_scoring, track_score_evolution

__all__ = [
    'ScoreCalculator',
    'ScoreEvolutionTracker',
    'ScoreAnalyzer',
    'apply_scoring',
    'track_score_evolution'
]
