"""
Simplified Config Loader
Loads config.yaml with minimal overhead
"""

import yaml
from pathlib import Path
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load YAML config file

    Args:
        config_path: Path to config file (relative to project root)

    Returns:
        Config dictionary
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_value(config: dict, *keys, default=None) -> Any:
    """
    Get nested value from config with dot notation support

    Args:
        config: Config dictionary
        *keys: Keys to navigate (e.g., 'static_analysis', 'n_clusters')
        default: Default value if key not found

    Returns:
        Config value or default

    Examples:
        >>> get_value(config, 'static_analysis', 'n_clusters')
        5
        >>> get_value(config, 'static_analysis', 'features')
        ['roa', 'roe', ...]
    """
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def parse_algorithms_from_config(cfg: dict) -> list:
    """
    Parse algorithm(s) from config.
    Supports single or multiple algorithms (comma-separated).

    Examples:
        'kmeans' -> ['kmeans']
        'kmeans, hierarchical, dbscan' -> ['kmeans', 'hierarchical', 'dbscan']

    Returns:
        List of algorithm names
    """
    algorithm_str = get_value(cfg, 'classification', 'algorithm', default='kmeans')

    if isinstance(algorithm_str, str):
        # Split by comma and strip whitespace
        algorithms = [alg.strip() for alg in algorithm_str.split(',')]
        # Filter out empty strings
        algorithms = [alg for alg in algorithms if alg]
        return algorithms
    elif isinstance(algorithm_str, list):
        # Already a list
        return algorithm_str
    else:
        # Fallback
        return ['kmeans']


def get_features_for_analysis(cfg: dict, analysis_type: str, default_features: List[str] = None) -> List[str]:
    """
    Lädt Features für eine Analyse basierend auf feature_selection mode

    Args:
        cfg: Config dictionary
        analysis_type: 'static_analysis', 'dynamic_analysis', oder 'combined_analysis'
        default_features: Fallback Features falls nichts konfiguriert

    Returns:
        Liste von Feature-Namen

    Examples:
        >>> get_features_for_analysis(cfg, 'static_analysis')
        ['roa', 'roe', 'ebit_margin', 'gross_margin', ...]
    """
    if default_features is None:
        default_features = ['roa', 'roe', 'ebit_margin', 'debt_to_equity', 'current_ratio']

    # Prüfe ob feature_selection mode gesetzt ist
    mode = get_value(cfg, 'feature_selection', 'mode', default='manual')

    if mode == 'preset':
        # Nutze Preset aus features_config.yaml
        preset_name = get_value(cfg, 'feature_selection', 'preset', default='pca_optimized')

        try:
            from src._01_setup.feature_config_loader import FeatureConfigLoader

            feature_loader = FeatureConfigLoader()
            preset_features = feature_loader.get_preset_features(preset_name)

            if analysis_type == 'static_analysis':
                features = preset_features.get('static', [])
                logger.info(f"Using preset '{preset_name}' for static analysis: {len(features)} features")
                return features

            elif analysis_type == 'dynamic_analysis':
                features = preset_features.get('dynamic', [])
                logger.info(f"Using preset '{preset_name}' for dynamic analysis: {len(features)} features")
                return features

            elif analysis_type == 'combined_analysis':
                # Return both static and dynamic
                static_features = preset_features.get('static', [])
                dynamic_features = preset_features.get('dynamic', [])
                logger.info(f"Using preset '{preset_name}' for combined: {len(static_features)} static + {len(dynamic_features)} dynamic")
                return {
                    'static': static_features,
                    'dynamic': dynamic_features
                }

        except Exception as e:
            logger.warning(f"Failed to load preset '{preset_name}': {e}. Falling back to manual mode.")
            mode = 'manual'

    # Manual mode - load from config.yaml directly
    if analysis_type == 'combined_analysis':
        return {
            'static': get_value(cfg, 'combined_analysis', 'features_static', default=default_features),
            'dynamic': get_value(cfg, 'combined_analysis', 'features_dynamic', default=[])
        }
    else:
        return get_value(cfg, analysis_type, 'features', default=default_features)
