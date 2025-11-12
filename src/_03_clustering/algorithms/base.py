"""
Base Clusterer Interface
Abstract Base Class für alle Clustering-Algorithmen
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import logging

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):
    """
    Abstract Base Class für Clustering-Algorithmen

    Alle Clusterer müssen diese Interface implementieren, um
    einheitliche Verwendung in der Pipeline zu garantieren.
    """

    def __init__(self, config: Dict, random_state: int = 42):
        """
        Initialisiert den Clusterer

        Args:
            config: Algorithmus-spezifische Konfiguration
            random_state: Random State für Reproduzierbarkeit
        """
        self.config = config
        self.random_state = random_state
        self.model = None
        self.scaler = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """
        Trainiert den Clustering-Algorithmus

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            self für Method Chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weist Cluster zu

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Kombiniert fit() und predict()

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        self.fit(X)
        return self.predict(X)

    @abstractmethod
    def get_n_clusters(self) -> int:
        """
        Gibt Anzahl der Cluster zurück

        Returns:
            Anzahl Cluster
        """
        pass

    @abstractmethod
    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Berechnet Qualitätsmetriken für das Clustering

        Args:
            X: Feature-Matrix (n_samples, n_features)
            labels: Cluster-Labels (n_samples,)

        Returns:
            Dictionary mit Metriken (z.B. silhouette_score, davies_bouldin)
        """
        pass

    def preprocess_data(
        self,
        df: pd.DataFrame,
        features: list,
        use_winsorization: bool = True,
        winsorize_limits: tuple = (0.01, 0.01)
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Standard-Preprocessing: Winsorization, Missing Values, Scaling

        Kann von spezifischen Clustern überschrieben werden für
        custom preprocessing.

        Args:
            df: DataFrame mit Features
            features: Liste der zu verwendenden Features
            use_winsorization: Use Winsorization for outlier handling (default: True)
            winsorize_limits: Percentiles to winsorize (default: 1% and 99%)

        Returns:
            Tuple (scaled_features, valid_indices)
        """
        df_subset = df[features].copy()
        initial = len(df_subset)

        # Remove rows with too many missing values BEFORE winsorization
        max_missing = 0.5
        missing_per_row = df_subset.isna().sum(axis=1) / len(features)
        df_subset = df_subset[missing_per_row <= max_missing]

        # Impute missing values with median
        df_subset = df_subset.fillna(df_subset.median())

        # Remove infinite values
        df_subset = df_subset.replace([np.inf, -np.inf], np.nan)
        df_subset = df_subset.dropna()

        # Winsorization: Cap extreme values at 1st/99th percentile
        if use_winsorization:
            n_outliers_total = 0
            for col in features:
                if col in df_subset.columns:
                    # Count outliers before winsorization
                    q_low = df_subset[col].quantile(winsorize_limits[0])
                    q_high = df_subset[col].quantile(1 - winsorize_limits[1])
                    n_outliers = ((df_subset[col] < q_low) | (df_subset[col] > q_high)).sum()
                    n_outliers_total += n_outliers

                    # Apply winsorization (cap values)
                    df_subset[col] = winsorize(df_subset[col].values,
                                              limits=winsorize_limits,
                                              nan_policy='omit')

            if n_outliers_total > 0:
                logger.info(f"  Winsorized {n_outliers_total} outlier values across {len(features)} features "
                          f"(capped at {winsorize_limits[0]:.1%}/{1-winsorize_limits[1]:.1%} percentiles)")

        removed = initial - len(df_subset)
        if removed > 0:
            logger.info(f"  Removed {removed} rows ({removed/initial*100:.1f}%) due to missing/invalid values")

        # Standardization (Z-score)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df_subset)

        return X_scaled, df_subset.index

    def get_model(self):
        """
        Gibt das trainierte Modell zurück (für Speicherung)

        Returns:
            Trainiertes Modell-Objekt
        """
        return self.model

    def get_scaler(self):
        """
        Gibt den Scaler zurück (für Speicherung)

        Returns:
            Fitted StandardScaler
        """
        return self.scaler

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        Gibt Namen des Algorithmus zurück (für Output-Ordner)

        Returns:
            Algorithmus-Name (lowercase, z.B. 'kmeans', 'hierarchical')
        """
        pass

    def get_algorithm_params(self) -> Dict:
        """
        Gibt Algorithmus-Parameter zurück (für Dokumentation)

        Returns:
            Dictionary mit Parametern
        """
        return self.config.copy()
