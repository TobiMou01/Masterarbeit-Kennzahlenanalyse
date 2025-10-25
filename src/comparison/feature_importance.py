"""
Feature Importance
Analysiert welche Features die Cluster am besten trennen
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureImportance:
    """Berechnet Feature Importance für Clustering-Ergebnisse"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators: Anzahl Trees im Random Forest
            random_state: Random State für Reproduzierbarkeit
        """
        self.n_estimators = n_estimators
        self.random_state = random_state

    def compute_importance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        cluster_col: str = 'cluster',
        algorithm_name: str = 'algorithm'
    ) -> pd.DataFrame:
        """
        Berechnet Feature Importance mittels Random Forest

        Args:
            df: DataFrame mit Features und Clustern
            feature_cols: Liste der Feature-Spalten
            cluster_col: Name der Cluster-Spalte
            algorithm_name: Name des Algorithmus

        Returns:
            DataFrame mit Feature Importance
        """
        logger.info(f"\n  Berechne Feature Importance für {algorithm_name}...")

        # Filtere gültige Cluster (keine Noise)
        df_valid = df[df[cluster_col] >= 0].copy()

        if len(df_valid) == 0:
            logger.warning(f"    ⚠ Keine gültigen Cluster")
            return None

        # Prüfe verfügbare Features
        available_features = [f for f in feature_cols if f in df_valid.columns]
        if not available_features:
            logger.warning(f"    ⚠ Keine Features verfügbar")
            return None

        # Prepare Data
        X = df_valid[available_features].copy()
        y = df_valid[cluster_col].copy()

        # Handle NaN values
        X = X.fillna(X.median())

        # Check for infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Prüfe ob genug Samples und Cluster
        if len(y.unique()) < 2:
            logger.warning(f"    ⚠ Nur {len(y.unique())} Cluster - zu wenig für Importance")
            return None

        if len(df_valid) < 10:
            logger.warning(f"    ⚠ Nur {len(df_valid)} Samples - zu wenig für RF")
            return None

        # Standardisierung
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X_scaled, y)

        # Feature Importance
        importances = rf.feature_importances_

        # Als DataFrame
        importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': importances,
            'algorithm': algorithm_name
        }).sort_values('importance', ascending=False)

        # Log Top 5
        logger.info(f"    Top 5 Features:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"      {row['feature']:<30} {row['importance']:.4f}")

        return importance_df

    def compute_all_algorithms(
        self,
        results_dict: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        analysis_type: str = 'static'
    ) -> Dict[str, pd.DataFrame]:
        """
        Berechnet Feature Importance für alle Algorithmen

        Args:
            results_dict: Dict mit {algorithm_name: df_with_clusters}
            feature_cols: Liste der Feature-Spalten
            analysis_type: Analyse-Typ

        Returns:
            Dict mit {algorithm_name: importance_df}
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE IMPORTANCE - {analysis_type.upper()}")
        logger.info(f"{'='*80}")

        importance_dict = {}

        for algorithm, df in results_dict.items():
            if 'cluster' not in df.columns:
                logger.warning(f"  ⚠ {algorithm}: Keine Cluster-Spalte")
                continue

            importance_df = self.compute_importance(
                df=df,
                feature_cols=feature_cols,
                cluster_col='cluster',
                algorithm_name=algorithm
            )

            if importance_df is not None:
                importance_dict[algorithm] = importance_df

        # Zusammenfassung
        if importance_dict:
            self._print_summary(importance_dict)

        return importance_dict

    def _print_summary(self, importance_dict: Dict[str, pd.DataFrame]):
        """Print Feature Importance Summary"""
        logger.info(f"\n{'='*80}")
        logger.info("=== FEATURE IMPORTANCE SUMMARY ===")
        logger.info(f"{'='*80}")

        # Kombiniere alle
        all_importance = pd.concat(importance_dict.values())

        # Top Features über alle Algorithmen
        top_features = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(5)

        logger.info("\nTop 5 Features across all algorithms:")
        for feature, importance in top_features.items():
            logger.info(f"  {feature:<30} (avg importance: {importance:.4f})")

    def plot_importance(
        self,
        importance_df: pd.DataFrame,
        algorithm_name: str,
        top_n: int = 15
    ) -> plt.Figure:
        """
        Erstellt Bar Chart für Feature Importance

        Args:
            importance_df: DataFrame mit Importance
            algorithm_name: Name des Algorithmus
            top_n: Anzahl Top-Features

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = importance_df.head(top_n)

        bars = ax.barh(
            range(len(top_features)),
            top_features['importance'],
            color='steelblue'
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()

        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {algorithm_name.upper()}\nTop {top_n} Features',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Werte anzeigen
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'], i, f" {row['importance']:.4f}",
                    va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_combined_importance(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        top_n: int = 10
    ) -> plt.Figure:
        """
        Erstellt kombinierten Plot für alle Algorithmen

        Args:
            importance_dict: Dict mit Importance DataFrames
            top_n: Anzahl Top-Features

        Returns:
            Matplotlib Figure
        """
        # Kombiniere und finde Top Features
        all_importance = pd.concat(importance_dict.values())
        top_features = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(top_n)

        # Erstelle DataFrame für Plot
        plot_data = []
        for feature in top_features.index:
            for algorithm, df in importance_dict.items():
                if feature in df['feature'].values:
                    importance = df[df['feature'] == feature]['importance'].values[0]
                else:
                    importance = 0
                plot_data.append({
                    'feature': feature,
                    'algorithm': algorithm,
                    'importance': importance
                })

        plot_df = pd.DataFrame(plot_data)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Grouped Bar Chart
        algorithms = list(importance_dict.keys())
        x = np.arange(len(top_features))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for i, algorithm in enumerate(algorithms):
            algo_data = plot_df[plot_df['algorithm'] == algorithm]
            values = [algo_data[algo_data['feature'] == f]['importance'].values[0] if f in algo_data['feature'].values else 0
                      for f in top_features.index]
            ax.bar(x + i*width, values, width, label=algorithm.upper(), color=colors[i])

        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance Comparison\nTop {top_n} Features Across All Algorithms',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(top_features.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
