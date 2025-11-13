"""
DBSCAN Clusterer
Implementierung des Density-Based Spatial Clustering (DBSCAN)
"""

import numpy as np
from typing import Dict, Tuple, List
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

from src._03_clustering.algorithms.base import BaseClusterer

logger = logging.getLogger(__name__)


def find_optimal_dbscan_params(
    X: np.ndarray,
    eps_range: List[float] = None,
    min_samples_range: List[int] = None
) -> Tuple[Tuple[float, int], List[Dict]]:
    """
    Grid search for optimal DBSCAN parameters.

    Args:
        X: Scaled feature matrix
        eps_range: List of eps values to test
        min_samples_range: List of min_samples values to test

    Returns:
        Tuple of (best_params, all_results)
        best_params: (eps, min_samples)
        all_results: List of dicts with all tested combinations
    """
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]

    if min_samples_range is None:
        min_samples_range = [2, 3, 5, 7, 10]

    logger.info(f"\n🔍 DBSCAN Grid Search")
    logger.info(f"  Eps range: {eps_range}")
    logger.info(f"  Min_samples range: {min_samples_range}")
    logger.info(f"  Total combinations: {len(eps_range) * len(min_samples_range)}\n")

    best_score = -1
    best_params = None
    results = []

    for eps, min_samples in product(eps_range, min_samples_range):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_pct = n_noise / len(X) * 100

        # Only evaluate valid clusterings
        # Requirements: ≥2 clusters, <50% noise
        if n_clusters >= 2 and n_noise < 0.5 * len(X):
            try:
                # Calculate silhouette for non-noise samples
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > n_clusters:
                    score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])

                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_pct': noise_pct,
                        'silhouette': score
                    }
                    results.append(result)

                    logger.info(f"  eps={eps:.2f}, min_samples={min_samples}: "
                              f"{n_clusters} clusters, {noise_pct:.1f}% noise, "
                              f"silhouette={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
            except Exception as e:
                logger.debug(f"  eps={eps:.2f}, min_samples={min_samples}: Failed ({e})")
        else:
            logger.debug(f"  eps={eps:.2f}, min_samples={min_samples}: "
                       f"Invalid ({n_clusters} clusters, {noise_pct:.1f}% noise)")

    if best_params is None:
        logger.warning("  ⚠️  No valid parameter combination found!")
        logger.warning("  Falling back to default: eps=1.5, min_samples=5")
        best_params = (1.5, 5)
    else:
        logger.info(f"\n  ✓ Optimal: eps={best_params[0]}, min_samples={best_params[1]} "
                  f"(silhouette={best_score:.3f})")

    return best_params, results


class DBSCANClusterer(BaseClusterer):
    """
    DBSCAN Clustering Implementation

    Verwendet scikit-learn's DBSCAN für Density-Based Clustering.

    Besonderheiten:
    - Findet Cluster-Anzahl automatisch
    - Erkennt Ausreißer/Noise (Label -1)
    - Cluster können beliebige Form haben
    - Keine Initialisierung nötig (deterministisch bei festen Daten)

    Parameter:
    - eps: Maximaler Abstand zwischen zwei Punkten (Radius)
    - min_samples: Minimale Anzahl Punkte für Core-Punkt
    """

    def __init__(self, config: Dict, random_state: int = 42):
        """
        Initialisiert DBSCAN Clusterer

        Args:
            config: Konfiguration mit 'eps' und 'min_samples'
            random_state: Nicht verwendet (DBSCAN ist deterministisch)
        """
        super().__init__(config, random_state)

        # DBSCAN spezifische Parameter
        self.eps = config.get('eps', 0.5)
        self.min_samples = config.get('min_samples', 5)
        self.metric = config.get('metric', 'euclidean')
        self.algorithm = config.get('algorithm', 'auto')
        self.leaf_size = config.get('leaf_size', 30)

        # Hinweis: n_clusters wird bei DBSCAN automatisch bestimmt
        # und ist daher nicht in der Config
        self._n_clusters = None
        self._n_noise = None

        # Model initialisieren
        # Note: DBSCAN hat keinen random_state Parameter
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=-1  # Parallele Verarbeitung
        )

        logger.info(f"DBSCAN initialisiert: eps={self.eps}, min_samples={self.min_samples}")

    def fit(self, X: np.ndarray) -> 'DBSCANClusterer':
        """
        Trainiert DBSCAN auf den Daten

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            self für Method Chaining
        """
        self.model.fit(X)

        # Cluster-Anzahl berechnen (ohne Noise)
        labels = self.model.labels_
        self._n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self._n_noise = np.sum(labels == -1)

        # Warnung wenn keine Cluster gefunden
        if self._n_clusters == 0:
            logger.warning(
                "DBSCAN hat keine Cluster gefunden! "
                f"Alle {len(X)} Punkte wurden als Noise klassifiziert. "
                f"Empfehlung: Erhöhe eps (aktuell: {self.eps}) oder "
                f"reduziere min_samples (aktuell: {self.min_samples})"
            )
        # Warnung wenn zu viele Cluster
        elif self._n_clusters > 20:
            logger.warning(
                f"DBSCAN hat {self._n_clusters} Cluster gefunden. "
                f"Das ist sehr viel! "
                f"Empfehlung: Erhöhe eps (aktuell: {self.eps}) oder "
                f"erhöhe min_samples (aktuell: {self.min_samples})"
            )
        # Info über Noise
        if self._n_noise > 0:
            noise_pct = (self._n_noise / len(X)) * 100
            logger.info(
                f"  DBSCAN: {self._n_clusters} Cluster gefunden, "
                f"{self._n_noise} Noise-Punkte ({noise_pct:.1f}%)"
            )
        else:
            logger.info(f"  DBSCAN: {self._n_clusters} Cluster gefunden, keine Noise-Punkte")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Weist Cluster zu

        Note: DBSCAN hat kein predict() für neue Daten.
        Diese Methode gibt die Labels vom fit() zurück.
        Für neue Daten müsste ein separater Classifier trainiert werden.

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        if not hasattr(self.model, 'labels_'):
            logger.warning(
                "DBSCAN wurde noch nicht trainiert. Führe fit() aus..."
            )
            self.fit(X)

        return self.model.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Kombiniert fit() und predict() - optimal für DBSCAN

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
            Label -1 = Noise/Outlier
        """
        labels = self.model.fit_predict(X)

        # Cluster-Anzahl und Noise berechnen
        self._n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self._n_noise = np.sum(labels == -1)

        # Warnungen
        if self._n_clusters == 0:
            logger.warning(
                "DBSCAN hat keine Cluster gefunden! "
                f"Alle {len(X)} Punkte wurden als Noise klassifiziert."
            )
        elif self._n_clusters > 20:
            logger.warning(f"DBSCAN hat {self._n_clusters} Cluster gefunden (sehr viel!)")

        if self._n_noise > 0:
            noise_pct = (self._n_noise / len(X)) * 100
            logger.info(
                f"  DBSCAN: {self._n_clusters} Cluster, "
                f"{self._n_noise} Noise ({noise_pct:.1f}%)"
            )

        return labels

    def get_n_clusters(self) -> int:
        """
        Gibt Anzahl der gefundenen Cluster zurück (ohne Noise)

        Returns:
            Anzahl Cluster (automatisch bestimmt)
        """
        if self._n_clusters is None:
            logger.warning("DBSCAN wurde noch nicht trainiert. Returniere 0.")
            return 0
        return self._n_clusters

    def get_n_noise(self) -> int:
        """
        Gibt Anzahl der Noise-Punkte zurück

        Returns:
            Anzahl Noise-Punkte
        """
        if self._n_noise is None:
            return 0
        return self._n_noise

    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Berechnet DBSCAN spezifische Metriken

        Args:
            X: Feature-Matrix (n_samples, n_features)
            labels: Cluster-Labels (n_samples,), -1 = Noise

        Returns:
            Dictionary mit Metriken
        """
        # Noise-Maske
        noise_mask = labels == -1
        n_noise = np.sum(noise_mask)
        n_total = len(labels)

        # Cluster ohne Noise
        non_noise_mask = ~noise_mask
        X_non_noise = X[non_noise_mask]
        labels_non_noise = labels[non_noise_mask]

        # Anzahl tatsächlicher Cluster
        n_clusters = len(set(labels_non_noise)) if len(labels_non_noise) > 0 else 0

        metrics = {
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'noise_percentage': (n_noise / n_total * 100) if n_total > 0 else 0,
            'eps': self.eps,
            'min_samples': self.min_samples
        }

        # Silhouette Score nur wenn genug Cluster ohne Noise
        if n_clusters >= 2 and len(labels_non_noise) > n_clusters:
            try:
                metrics['silhouette'] = silhouette_score(X_non_noise, labels_non_noise)
                metrics['davies_bouldin'] = davies_bouldin_score(X_non_noise, labels_non_noise)
            except Exception as e:
                logger.warning(f"Konnte Metriken nicht berechnen: {e}")
                metrics['silhouette'] = -999  # Sentinel value
                metrics['davies_bouldin'] = -999
        else:
            logger.warning(
                f"Zu wenige Cluster ({n_clusters}) oder Punkte für Silhouette Score. "
                "Setze Metriken auf -999."
            )
            metrics['silhouette'] = -999
            metrics['davies_bouldin'] = -999

        # Core samples (Punkte die Core-Punkte sind)
        if hasattr(self.model, 'core_sample_indices_'):
            metrics['n_core_samples'] = len(self.model.core_sample_indices_)
            metrics['core_sample_percentage'] = (
                len(self.model.core_sample_indices_) / n_total * 100
            )

        return metrics

    def get_algorithm_name(self) -> str:
        """
        Gibt Namen des Algorithmus zurück

        Returns:
            'dbscan'
        """
        return 'dbscan'

    def get_algorithm_params(self) -> Dict:
        """
        Gibt DBSCAN Parameter zurück

        Returns:
            Dictionary mit Parametern
        """
        return {
            'algorithm': 'DBSCAN (Density-Based)',
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'n_clusters': self._n_clusters if self._n_clusters else 'automatic',
            'n_noise': self._n_noise if self._n_noise else 0,
            'feature': 'Automatic cluster count + outlier detection'
        }

    def get_noise_mask(self) -> np.ndarray:
        """
        Gibt Boolean-Maske für Noise-Punkte zurück

        Returns:
            Boolean array (True = Noise)
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model muss erst trainiert werden (fit())")

        return self.model.labels_ == -1

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        Gibt Verteilung der Cluster zurück

        Returns:
            Dictionary {cluster_id: count}
            -1 = Noise
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Model muss erst trainiert werden (fit())")

        labels = self.model.labels_
        unique, counts = np.unique(labels, return_counts=True)

        distribution = {}
        for label, count in zip(unique, counts):
            if label == -1:
                distribution['Noise'] = int(count)
            else:
                distribution[f'Cluster_{label}'] = int(count)

        return distribution

    def recommend_parameters(self, X: np.ndarray) -> Dict:
        """
        Empfiehlt Parameter basierend auf Daten (k-distance graph)

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Dictionary mit empfohlenen Parametern
        """
        from sklearn.neighbors import NearestNeighbors

        # K-Nearest Neighbors für eps-Schätzung
        k = self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # k-distance (Abstand zum k-nächsten Nachbarn)
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)

        # Empfohlener eps: Elbow in k-distance graph
        # Heuristik: 90. Perzentil der k-Distanzen
        recommended_eps = np.percentile(k_distances_sorted, 90)

        return {
            'recommended_eps': float(recommended_eps),
            'current_eps': self.eps,
            'min_k_distance': float(k_distances_sorted.min()),
            'max_k_distance': float(k_distances_sorted.max()),
            'median_k_distance': float(np.median(k_distances_sorted)),
            'note': 'Empfehlung: Visualisiere k-distance graph für optimales eps'
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== DBSCAN CLUSTERER ===")
    print("DBSCAN (Density-Based Spatial Clustering) Implementierung")
    print("\nVerwendung:")
    print("  clusterer = DBSCANClusterer({'eps': 0.5, 'min_samples': 5})")
    print("  labels = clusterer.fit_predict(X)")
    print("  # Label -1 = Noise/Outlier")
    print("\nVorteile:")
    print("  - Findet Cluster-Anzahl automatisch")
    print("  - Erkennt Ausreißer/Noise")
    print("  - Cluster können beliebige Form haben")
    print("  - Deterministisch")
    print("\nNachteile:")
    print("  - Parameter eps schwer zu wählen")
    print("  - Sensitiv auf Skalierung")
    print("  - Nicht für high-dimensional data optimal")
