"""
Feature Config Loader
Lädt und verwaltet Feature-Konfigurationen aus features_config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureConfigLoader:
    """Lädt und verwaltet Feature-Konfigurationen"""

    def __init__(self, config_path: str = 'features_config.yaml'):
        """
        Initialisiert den Feature Config Loader

        Args:
            config_path: Pfad zur features_config.yaml
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Lädt die YAML Config"""
        if not self.config_path.exists():
            logger.warning(f"Feature config nicht gefunden: {self.config_path}")
            return {}

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_all_static_features(self) -> List[str]:
        """
        Gibt alle aktivierten statischen Features zurück

        Returns:
            Liste aller aktivierten statischen Feature-Namen
        """
        features = []

        categories = ['profitability', 'liquidity', 'leverage', 'efficiency', 'cashflow', 'structure']

        for category in categories:
            if category not in self.config:
                continue

            cat_config = self.config[category]
            if not cat_config.get('enabled', True):
                continue

            metrics = cat_config.get('metrics', {})
            for metric_name, metric_config in metrics.items():
                if isinstance(metric_config, dict):
                    if metric_config.get('enabled', True):
                        features.append(metric_name)
                elif metric_config is True or metric_config is None:
                    features.append(metric_name)

        return features

    def get_all_dynamic_features(self) -> List[str]:
        """
        Gibt alle aktivierten dynamischen Features zurück

        Returns:
            Liste aller aktivierten dynamischen Feature-Namen
        """
        features = []

        if 'dynamic' not in self.config:
            return features

        dynamic_config = self.config['dynamic']
        if not dynamic_config.get('enabled', True):
            return features

        categories = ['growth', 'trends', 'volatility', 'quality']

        for category in categories:
            if category not in dynamic_config:
                continue

            cat_config = dynamic_config[category]
            if not cat_config.get('enabled', True):
                continue

            metrics = cat_config.get('metrics', {})
            for metric_name, metric_config in metrics.items():
                if isinstance(metric_config, dict):
                    if metric_config.get('enabled', True):
                        features.append(metric_name)
                elif metric_config is True or metric_config is None:
                    features.append(metric_name)

        return features

    def get_preset_features(self, preset_name: str) -> Dict[str, List[str]]:
        """
        Gibt Features für ein bestimmtes Preset zurück

        Args:
            preset_name: Name des Presets (minimal, standard, comprehensive, pca_optimized)

        Returns:
            Dict mit 'static' und 'dynamic' Feature-Listen
        """
        if 'presets' not in self.config:
            logger.warning("Keine Presets in Config gefunden")
            return {'static': [], 'dynamic': []}

        presets = self.config['presets']

        if preset_name not in presets:
            logger.warning(f"Preset '{preset_name}' nicht gefunden")
            return {'static': [], 'dynamic': []}

        preset = presets[preset_name]

        # Wenn use_all gesetzt ist, gib alle Features zurück
        if preset.get('use_all', False):
            return {
                'static': self.get_all_static_features(),
                'dynamic': self.get_all_dynamic_features()
            }

        # Ansonsten parse die Kategorien
        static_features = []
        dynamic_features = []

        categories = preset.get('categories', [])

        for category_item in categories:
            if isinstance(category_item, dict):
                # Format: {category_name: [feature1, feature2]}
                for cat_name, feature_list in category_item.items():
                    # Bestimme ob static oder dynamic
                    if cat_name in ['growth', 'trends', 'volatility', 'quality']:
                        dynamic_features.extend(feature_list)
                    else:
                        static_features.extend(feature_list)

        return {
            'static': static_features,
            'dynamic': dynamic_features
        }

    def get_feature_info(self, feature_name: str) -> Optional[Dict]:
        """
        Gibt Metadaten für ein Feature zurück

        Args:
            feature_name: Name des Features

        Returns:
            Dict mit formula, unit, interpretation etc.
        """
        # Suche in allen Kategorien
        all_categories = ['profitability', 'liquidity', 'leverage', 'efficiency', 'cashflow', 'structure']

        for category in all_categories:
            if category not in self.config:
                continue

            metrics = self.config[category].get('metrics', {})
            if feature_name in metrics:
                return metrics[feature_name]

        # Suche in dynamic
        if 'dynamic' in self.config:
            dynamic_cats = ['growth', 'trends', 'volatility', 'quality']
            for cat in dynamic_cats:
                if cat not in self.config['dynamic']:
                    continue

                metrics = self.config['dynamic'][cat].get('metrics', {})
                if feature_name in metrics:
                    return metrics[feature_name]

        return None

    def get_required_columns(self) -> List[str]:
        """
        Gibt alle benötigten Spalten aus den Rohdaten zurück

        Returns:
            Flache Liste aller benötigten Spalten
        """
        if 'required_columns' not in self.config:
            return []

        required = self.config['required_columns']
        all_columns = []

        for category, columns in required.items():
            if isinstance(columns, list):
                all_columns.extend(columns)

        return list(set(all_columns))  # Remove duplicates

    def get_outlier_config(self, feature_name: Optional[str] = None) -> Dict:
        """
        Gibt Outlier-Konfiguration zurück

        Args:
            feature_name: Optional - spezifisches Feature, sonst global

        Returns:
            Dict mit method und threshold
        """
        if 'outliers' not in self.config:
            return {'method': 'iqr', 'threshold': 3}

        outlier_config = self.config['outliers']

        # Wenn kein Feature spezifiziert, gib global zurück
        if feature_name is None:
            return {
                'method': outlier_config.get('global_method', 'iqr'),
                'threshold': outlier_config.get('global_threshold', 3)
            }

        # Prüfe ob es Category-Overrides gibt
        # Finde die Kategorie des Features
        feature_info = self.get_feature_info(feature_name)
        if feature_info and 'outlier_method' in feature_info:
            return {
                'method': feature_info.get('outlier_method', outlier_config.get('global_method', 'iqr')),
                'threshold': feature_info.get('outlier_threshold', outlier_config.get('global_threshold', 3))
            }

        return {
            'method': outlier_config.get('global_method', 'iqr'),
            'threshold': outlier_config.get('global_threshold', 3)
        }


def load_feature_config(config_path: str = 'features_config.yaml') -> FeatureConfigLoader:
    """
    Convenience function to load feature config

    Args:
        config_path: Pfad zur Config

    Returns:
        FeatureConfigLoader Instanz
    """
    return FeatureConfigLoader(config_path)


def get_preset_for_analysis(preset_name: str = 'pca_optimized') -> Dict[str, List[str]]:
    """
    Lädt Features für ein Preset - optimiert für Clustering

    Args:
        preset_name: Name des Presets

    Returns:
        Dict mit static und dynamic Features
    """
    loader = load_feature_config()
    return loader.get_preset_features(preset_name)


# Convenience functions für direkte Nutzung
def get_all_features() -> Dict[str, List[str]]:
    """Gibt alle verfügbaren Features zurück"""
    loader = load_feature_config()
    return {
        'static': loader.get_all_static_features(),
        'dynamic': loader.get_all_dynamic_features()
    }


if __name__ == "__main__":
    """Test der Feature Config"""

    print("=" * 80)
    print("FEATURE CONFIG LOADER TEST")
    print("=" * 80)

    loader = load_feature_config()

    print("\n📊 ALLE STATIC FEATURES:")
    static = loader.get_all_static_features()
    print(f"Total: {len(static)}")
    for feat in sorted(static):
        print(f"  - {feat}")

    print("\n📈 ALLE DYNAMIC FEATURES:")
    dynamic = loader.get_all_dynamic_features()
    print(f"Total: {len(dynamic)}")
    for feat in sorted(dynamic):
        print(f"  - {feat}")

    print("\n🎯 PRESETS:")
    for preset_name in ['minimal', 'standard', 'comprehensive', 'pca_optimized']:
        features = loader.get_preset_features(preset_name)
        total = len(features['static']) + len(features['dynamic'])
        print(f"\n  {preset_name.upper()} ({total} features):")
        print(f"    Static:  {len(features['static'])} features")
        print(f"    Dynamic: {len(features['dynamic'])} features")

    print("\n📋 REQUIRED DATA COLUMNS:")
    required = loader.get_required_columns()
    print(f"Total: {len(required)} columns needed from raw data")

    print("\n✓ Feature Config Loader Test abgeschlossen!")
