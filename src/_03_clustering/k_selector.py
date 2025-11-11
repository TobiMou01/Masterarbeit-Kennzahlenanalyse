"""
K-Selector: Automatische Bestimmung der optimalen Clusterzahl
Implementiert: Elbow-Methode, Silhouette-Score, Gap-Statistic
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class KSelector:
    """Bestimmt optimale Anzahl Cluster mittels verschiedener Methoden"""

    def __init__(self, k_range: List[int] = None, random_state: int = 42):
        """
        Args:
            k_range: Range von k-Werten zum Testen (default: [2, 10])
            random_state: Random State für Reproduzierbarkeit
        """
        self.k_range = k_range if k_range else list(range(2, 11))
        self.random_state = random_state
        self.results = {}

    def find_optimal_k(
        self,
        X: np.ndarray,
        methods: List[str] = None
    ) -> Tuple[int, Dict]:
        """
        Findet optimales k mittels verschiedener Methoden.

        Args:
            X: Feature Matrix (scaled)
            methods: Liste der Methoden (['elbow', 'silhouette', 'gap'])

        Returns:
            Tuple von (optimal_k, results_dict)
        """
        if methods is None:
            methods = ['elbow', 'silhouette', 'gap']

        logger.info(f"\n{'='*80}")
        logger.info("AUTOMATISCHE K-BESTIMMUNG")
        logger.info(f"{'='*80}\n")
        logger.info(f"K-Range: {self.k_range}")
        logger.info(f"Methoden: {methods}")
        logger.info(f"Samples: {X.shape[0]}\n")

        results = {}

        # 1. Elbow-Methode
        if 'elbow' in methods:
            logger.info("📊 Elbow-Methode...")
            inertias = self._compute_elbow(X)
            elbow_k = self._detect_elbow_point(inertias)
            results['elbow'] = {
                'optimal_k': elbow_k,
                'inertias': inertias
            }
            logger.info(f"  ✓ Optimal k: {elbow_k}")

        # 2. Silhouette-Score
        if 'silhouette' in methods:
            logger.info("\n📊 Silhouette-Score...")
            silhouettes = self._compute_silhouette(X)
            silhouette_k = max(silhouettes, key=silhouettes.get)
            results['silhouette'] = {
                'optimal_k': silhouette_k,
                'scores': silhouettes
            }
            logger.info(f"  ✓ Optimal k: {silhouette_k} (Score: {silhouettes[silhouette_k]:.3f})")

        # 3. Gap-Statistic
        if 'gap' in methods:
            logger.info("\n📊 Gap-Statistic...")
            gaps = self._compute_gap_statistic(X)
            gap_k = max(gaps, key=gaps.get)
            results['gap'] = {
                'optimal_k': gap_k,
                'gaps': gaps
            }
            logger.info(f"  ✓ Optimal k: {gap_k} (Gap: {gaps[gap_k]:.3f})")

        # Konsens finden
        logger.info(f"\n{'='*80}")
        logger.info("KONSENS-BESTIMMUNG")
        logger.info(f"{'='*80}")

        optimal_ks = [results[m]['optimal_k'] for m in results]
        logger.info(f"\nEmpfehlungen: {optimal_ks}")

        # Häufigster Wert oder Median
        from collections import Counter
        counts = Counter(optimal_ks)
        most_common_k = counts.most_common(1)[0][0]

        logger.info(f"Konsens k: {most_common_k}")

        self.results = results
        results['consensus'] = most_common_k

        return most_common_k, results

    def _compute_elbow(self, X: np.ndarray) -> Dict[int, float]:
        """Berechnet Inertia für Elbow-Methode"""
        inertias = {}

        for k in self.k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X)
            inertias[k] = kmeans.inertia_

        return inertias

    def _detect_elbow_point(self, inertias: Dict[int, float]) -> int:
        """
        Findet Elbow-Punkt mittels Kneedle-Algorithmus (vereinfacht).

        Strategie: Finde Punkt mit maximaler Krümmung
        """
        k_values = sorted(inertias.keys())
        inertia_values = [inertias[k] for k in k_values]

        # Normalisieren auf [0, 1]
        x_norm = np.array(k_values, dtype=float)
        x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())

        y_norm = np.array(inertia_values, dtype=float)
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min())

        # Distanz zur Linie von Start zu Ende
        # Line: y = mx + b
        m = (y_norm[-1] - y_norm[0]) / (x_norm[-1] - x_norm[0]) if (x_norm[-1] - x_norm[0]) != 0 else 0
        b = y_norm[0] - m * x_norm[0]

        # Distanz jedes Punktes zur Linie
        distances = []
        for i in range(len(x_norm)):
            # Distance from point to line: |mx - y + b| / sqrt(m^2 + 1)
            dist = abs(m * x_norm[i] - y_norm[i] + b) / np.sqrt(m**2 + 1)
            distances.append(dist)

        # Punkt mit maximaler Distanz = Elbow
        elbow_idx = np.argmax(distances)
        elbow_k = k_values[elbow_idx]

        return elbow_k

    def _compute_silhouette(self, X: np.ndarray) -> Dict[int, float]:
        """Berechnet Silhouette-Scores"""
        silhouettes = {}

        for k in self.k_range:
            if k >= len(X):
                continue  # Zu wenig Samples

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(X)

            # Silhouette-Score
            score = silhouette_score(X, labels)
            silhouettes[k] = score

        return silhouettes

    def _compute_gap_statistic(
        self,
        X: np.ndarray,
        n_refs: int = 10
    ) -> Dict[int, float]:
        """
        Berechnet Gap-Statistic.

        Vergleicht Inertia mit zufälligen Referenz-Daten.

        Args:
            X: Feature Matrix
            n_refs: Anzahl Referenz-Datasets

        Returns:
            Dict mit Gap-Werten pro k
        """
        gaps = {}

        # Bounds für Referenz-Daten
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        for k in self.k_range:
            # Reale Inertia
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X)
            real_inertia = kmeans.inertia_

            # Referenz-Inertias
            ref_inertias = []
            for _ in range(n_refs):
                # Zufällige Daten im gleichen Wertebereich
                X_ref = np.random.uniform(
                    low=mins,
                    high=maxs,
                    size=X.shape
                )

                kmeans_ref = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans_ref.fit(X_ref)
                ref_inertias.append(kmeans_ref.inertia_)

            # Gap = log(E[ref_inertia]) - log(real_inertia)
            mean_ref_inertia = np.mean(ref_inertias)
            gap = np.log(mean_ref_inertia) - np.log(real_inertia)
            gaps[k] = gap

        return gaps

    def get_recommendation_summary(self) -> pd.DataFrame:
        """Erstellt Zusammenfassung der Empfehlungen"""
        if not self.results:
            return None

        summary_data = []
        for method, data in self.results.items():
            if method == 'consensus':
                continue

            summary_data.append({
                'method': method,
                'optimal_k': data['optimal_k']
            })

        summary_data.append({
            'method': 'CONSENSUS',
            'optimal_k': self.results.get('consensus')
        })

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test KSelector
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("K-SELECTOR TEST")
    print("=" * 80)

    # Mock-Daten
    np.random.seed(42)
    n_samples = 300

    # 4 Cluster (Ground Truth)
    centers = [
        [0, 0],
        [5, 5],
        [0, 5],
        [5, 0]
    ]

    X = []
    for center in centers:
        cluster_data = np.random.randn(n_samples // 4, 2) * 0.5 + center
        X.append(cluster_data)

    X = np.vstack(X)

    print(f"\nMock-Daten: {X.shape[0]} Samples, {X.shape[1]} Features")
    print(f"Ground Truth: 4 Cluster\n")

    # Test KSelector
    selector = KSelector(k_range=list(range(2, 8)))
    optimal_k, results = selector.find_optimal_k(X)

    print(f"\n{'='*80}")
    print(f"ERGEBNIS: Optimales k = {optimal_k}")
    print(f"{'='*80}")

    summary = selector.get_recommendation_summary()
    print("\n" + str(summary))

    print("\n✓ KSelector Test erfolgreich!")
    print("=" * 80 + "\n")
