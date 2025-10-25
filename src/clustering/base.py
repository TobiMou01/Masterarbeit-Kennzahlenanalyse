"""
Base Clusterer Interface
Abstract Base Class für alle Clustering-Algorithmen
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        features: list
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Standard-Preprocessing: Outlier-Removal, Missing Values, Scaling

        Kann von spezifischen Clustern überschrieben werden für
        custom preprocessing.

        Args:
            df: DataFrame mit Features
            features: Liste der zu verwendenden Features

        Returns:
            Tuple (scaled_features, valid_indices)
        """
        df_subset = df[features].copy()
        initial = len(df_subset)

        # Outlier entfernen (Quantile-Methode)
        for col in features:
            if col in df_subset.columns:
                q_low = df_subset[col].quantile(0.001)
                q_high = df_subset[col].quantile(0.999)
                df_subset.loc[(df_subset[col] < q_low) | (df_subset[col] > q_high), col] = np.nan

        # Zeilen mit zu vielen NaNs entfernen
        max_missing = 0.5
        missing_per_row = df_subset.isna().sum(axis=1) / len(features)
        df_subset = df_subset[missing_per_row <= max_missing]

        # Impute mit Median
        df_subset = df_subset.fillna(df_subset.median())

        # Inf entfernen
        df_subset = df_subset.replace([np.inf, -np.inf], np.nan).dropna()

        removed = initial - len(df_subset)

        # Standardisierung
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
