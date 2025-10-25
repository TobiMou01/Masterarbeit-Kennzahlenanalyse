"""
Hierarchical Clusterer
Implementierung des Agglomerative Hierarchical Clustering mit Ward's Methode
"""

import numpy as np
from typing import Dict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

from src._03_clustering.algorithms.base import BaseClusterer

logger = logging.getLogger(__name__)


class HierarchicalClusterer(BaseClusterer):
    """
    Hierarchical Clustering Implementation (Ward's Methode)

    Verwendet scikit-learn's AgglomerativeClustering für robustes
    hierarchisches Clustering mit Ward's Linkage-Methode.

    Ward's Methode minimiert die Varianz innerhalb der Cluster und
    verwendet euklidische Distanz.
    """

    def __init__(self, config: Dict, random_state: int = 42):
        """
        Initialisiert Hierarchical Clusterer

        Args:
            config: Konfiguration mit mindestens 'n_clusters'
            random_state: Random State für Reproduzierbarkeit (nicht bei Hierarchical verwendet)
        """
        super().__init__(config, random_state)

        # Hierarchical Clustering spezifische Parameter
        self.n_clusters = config.get('n_clusters', 5)
        self.linkage = config.get('linkage', 'ward')
        self.distance_metric = config.get('distance_metric', 'euclidean')

        # Validierung: Ward benötigt euklidische Distanz
        if self.linkage == 'ward' and self.distance_metric != 'euclidean':
            logger.warning(
                f"Ward's Methode benötigt euklidische Distanz. "
                f"Setze distance_metric von '{self.distance_metric}' auf 'euclidean'"
            )
            self.distance_metric = 'euclidean'

        # Model initialisieren
        # Note: AgglomerativeClustering hat keinen random_state Parameter,
        # da der Algorithmus deterministisch ist
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.distance_metric
        )

    def fit(self, X: np.ndarray) -> 'HierarchicalClusterer':
        """
        Trainiert Hierarchical Clustering auf den Daten

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

        Note: Hierarchical Clustering hat kein separates predict(),
        daher muss fit_predict() verwendet oder das Modell neu trainiert werden.

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array mit Cluster-Labels (n_samples,)
        """
        if not hasattr(self.model, 'labels_'):
            # Model wurde noch nicht gefittet
            logger.warning(
                "Hierarchical Clustering wurde noch nicht trainiert. "
                "Führe fit() aus..."
            )
            self.fit(X)

        return self.model.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Kombiniert fit() und predict() - optimal für Hierarchical Clustering

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
        Berechnet Hierarchical Clustering spezifische Metriken

        Args:
            X: Feature-Matrix (n_samples, n_features)
            labels: Cluster-Labels (n_samples,)

        Returns:
            Dictionary mit Metriken
        """
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'distance_metric': self.distance_metric
        }

        # Hierarchical Clustering hat kein Inertia-Konzept wie K-Means
        # Aber wir können die Anzahl der Leaves und Children speichern
        if hasattr(self.model, 'n_leaves_'):
            metrics['n_leaves'] = self.model.n_leaves_

        if hasattr(self.model, 'n_connected_components_'):
            metrics['n_connected_components'] = self.model.n_connected_components_

        return metrics

    def get_algorithm_name(self) -> str:
        """
        Gibt Namen des Algorithmus zurück

        Returns:
            'hierarchical'
        """
        return 'hierarchical'

    def get_algorithm_params(self) -> Dict:
        """
        Gibt Hierarchical Clustering Parameter zurück

        Returns:
            Dictionary mit Parametern
        """
        return {
            'algorithm': 'Hierarchical Clustering (Ward\'s Method)',
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'distance_metric': self.distance_metric,
            'method': 'Agglomerative (Bottom-Up)'
        }

    def get_dendrogram_linkage_matrix(self) -> np.ndarray:
        """
        Erstellt Linkage-Matrix für Dendrogram-Visualisierung

        Returns:
            Linkage-Matrix für scipy.cluster.hierarchy.dendrogram

        Note:
            Diese Methode ist optional und kann für erweiterte
            Visualisierungen verwendet werden.
        """
        if not hasattr(self.model, 'children_'):
            raise ValueError(
                "Model muss erst trainiert werden (fit()) "
                "bevor Linkage-Matrix erstellt werden kann"
            )

        # Erstelle Linkage-Matrix aus children_ und distances_
        from scipy.cluster.hierarchy import linkage

        # Sklearn's children_ Format in Linkage-Matrix konvertieren
        n_samples = len(self.model.labels_)
        n_clusters = self.model.n_clusters_

        # Children enthält für jeden Merge die beiden Cluster-IDs
        children = self.model.children_

        # Distances (wenn verfügbar)
        if hasattr(self.model, 'distances_'):
            distances = self.model.distances_
        else:
            # Fallback: Berechne Distanzen manuell (approximiert)
            distances = np.arange(len(children))

        # Erstelle Linkage-Matrix im scipy-Format
        # Format: [idx1, idx2, distance, sample_count]
        counts = np.zeros(len(children) + n_samples)
        counts[:n_samples] = 1  # Jedes Sample hat Count 1

        linkage_matrix = np.zeros((len(children), 4))
        for i, (child1, child2) in enumerate(children):
            linkage_matrix[i, 0] = child1
            linkage_matrix[i, 1] = child2
            linkage_matrix[i, 2] = distances[i] if i < len(distances) else i
            linkage_matrix[i, 3] = counts[child1] + counts[child2]
            counts[n_samples + i] = linkage_matrix[i, 3]

        return linkage_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== HIERARCHICAL CLUSTERER ===")
    print("Hierarchical Clustering (Ward's Method) Implementierung")
    print("\nVerwendung:")
    print("  clusterer = HierarchicalClusterer({'n_clusters': 5, 'linkage': 'ward'})")
    print("  labels = clusterer.fit_predict(X)")
    print("\nVorteile:")
    print("  - Keine Initialisierung nötig (deterministisch)")
    print("  - Hierarchische Struktur sichtbar")
    print("  - Ward's Methode minimiert Varianz")
    print("  - Robuste Cluster-Bildung")
