"""
K-Means Clusterer
Implementierung des K-Means Clustering-Algorithmus
"""

import numpy as np
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

from src._03_clustering.algorithms.base import BaseClusterer

logger = logging.getLogger(__name__)


class KMeansClusterer(BaseClusterer):
    """
    K-Means Clustering Implementation

    Verwendet scikit-learn's KMeans für robustes Clustering.
    """

    def __init__(self, config: Dict, random_state: int = 42):
        """
        Initialisiert K-Means Clusterer

        Args:
            config: Konfiguration mit mindestens 'n_clusters'
            random_state: Random State für Reproduzierbarkeit
        """
        super().__init__(config, random_state)

        # K-Means spezifische Parameter
        self.n_clusters = config.get('n_clusters', 5)
        self.n_init = config.get('n_init', 20)
        self.max_iter = config.get('max_iter', 300)
        self.algorithm = config.get('algorithm', 'lloyd')

        # Model initialisieren
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm=self.algorithm
        )

    def fit(self, X: np.ndarray) -> 'KMeansClusterer':
        """
        Trainiert K-Means auf den Daten

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            self für Method Chaining
        """
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weist Cluster zu

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Kombiniert fit() und predict() - optimiert für K-Means

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        return self.model.fit_predict(X)

    def get_n_clusters(self) -> int:
        """
        Gibt Anzahl der Cluster zurück

        Returns:
            Anzahl Cluster
        """
        return self.n_clusters

    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Berechnet K-Means spezifische Metriken

        Args:
            X: Feature-Matrix (n_samples, n_features)
            labels: Cluster-Labels (n_samples,)

        Returns:
            Dictionary mit Metriken
        """
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'inertia': self.model.inertia_,
            'n_iter': self.model.n_iter_,
            'n_clusters': self.n_clusters
        }

        return metrics

    def get_algorithm_name(self) -> str:
        """
        Gibt Namen des Algorithmus zurück

        Returns:
            'kmeans'
        """
        return 'kmeans'

    def get_cluster_centers(self) -> np.ndarray:
        """
        Gibt Cluster-Zentren zurück (K-Means spezifisch)

        Returns:
            Cluster-Zentren (n_clusters, n_features)
        """
        return self.model.cluster_centers_

    def get_algorithm_params(self) -> Dict:
        """
        Gibt K-Means Parameter zurück

        Returns:
            Dictionary mit Parametern
        """
        return {
            'algorithm': 'K-Means',
            'n_clusters': self.n_clusters,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'sklearn_algorithm': self.algorithm
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== K-MEANS CLUSTERER ===")
    print("K-Means Clustering Implementierung")
    print("\nVerwendung:")
    print("  clusterer = KMeansClusterer({'n_clusters': 5})")
    print("  labels = clusterer.fit_predict(X)")
