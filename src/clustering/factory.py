"""
Clusterer Factory
Factory Pattern zum dynamischen Erstellen von Clustering-Algorithmen
"""

from typing import Dict, Type
import logging

from src.clustering.base import BaseClusterer
from src.clustering.kmeans_clusterer import KMeansClusterer
from src.clustering.hierarchical_clusterer import HierarchicalClusterer
from src.clustering.dbscan_clusterer import DBSCANClusterer

logger = logging.getLogger(__name__)


class ClustererFactory:
    """
    Factory zum Erstellen von Clustering-Algorithmen

    Automatische Registrierung verfügbarer Algorithmen und
    dynamische Instanziierung basierend auf Config.
    """

    # Registry aller verfügbaren Clusterer
    _registry: Dict[str, Type[BaseClusterer]] = {
        'kmeans': KMeansClusterer,
        'hierarchical': HierarchicalClusterer,
        'dbscan': DBSCANClusterer,
        # Später hinzufügen:
        # 'gaussian_mixture': GaussianMixtureClusterer,
    }

    @classmethod
    def create(
        cls,
        algorithm: str,
        config: Dict,
        random_state: int = 42
    ) -> BaseClusterer:
        """
        Erstellt einen Clusterer basierend auf Algorithmus-Namen

        Args:
            algorithm: Name des Algorithmus (z.B. 'kmeans', 'hierarchical')
            config: Algorithmus-spezifische Konfiguration
            random_state: Random State für Reproduzierbarkeit

        Returns:
            Instanziierter Clusterer

        Raises:
            ValueError: Wenn Algorithmus nicht verfügbar
        """
        algorithm = algorithm.lower()

        if algorithm not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unbekannter Clustering-Algorithmus: '{algorithm}'. "
                f"Verfügbar: {available}"
            )

        clusterer_class = cls._registry[algorithm]
        logger.info(f"Erstelle Clusterer: {clusterer_class.__name__}")

        return clusterer_class(config=config, random_state=random_state)

    @classmethod
    def register(cls, name: str, clusterer_class: Type[BaseClusterer]):
        """
        Registriert einen neuen Clustering-Algorithmus

        Erlaubt dynamisches Hinzufügen neuer Algorithmen zur Laufzeit.

        Args:
            name: Eindeutiger Name des Algorithmus
            clusterer_class: Klasse, die BaseClusterer implementiert
        """
        if not issubclass(clusterer_class, BaseClusterer):
            raise TypeError(
                f"{clusterer_class.__name__} muss von BaseClusterer erben"
            )

        name = name.lower()
        if name in cls._registry:
            logger.warning(f"Überschreibe existierenden Clusterer: {name}")

        cls._registry[name] = clusterer_class
        logger.info(f"Registriert: {name} -> {clusterer_class.__name__}")

    @classmethod
    def get_available_algorithms(cls) -> list:
        """
        Gibt Liste aller verfügbaren Algorithmen zurück

        Returns:
            Liste mit Algorithmus-Namen
        """
        return list(cls._registry.keys())

    @classmethod
    def is_available(cls, algorithm: str) -> bool:
        """
        Prüft ob Algorithmus verfügbar ist

        Args:
            algorithm: Name des Algorithmus

        Returns:
            True wenn verfügbar, sonst False
        """
        return algorithm.lower() in cls._registry


# Convenience function für direkten Import
def create_clusterer(
    algorithm: str = 'kmeans',
    config: Dict = None,
    random_state: int = 42
) -> BaseClusterer:
    """
    Convenience Function zum Erstellen eines Clusterers

    Args:
        algorithm: Name des Algorithmus (default: 'kmeans')
        config: Algorithmus-spezifische Konfiguration
        random_state: Random State

    Returns:
        Instanziierter Clusterer
    """
    if config is None:
        config = {'n_clusters': 5}  # Default config

    return ClustererFactory.create(algorithm, config, random_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== CLUSTERER FACTORY ===")
    print("\nVerfügbare Algorithmen:")
    for algo in ClustererFactory.get_available_algorithms():
        print(f"  - {algo}")

    print("\nBeispiel:")
    print("  clusterer = ClustererFactory.create('kmeans', {'n_clusters': 5})")
