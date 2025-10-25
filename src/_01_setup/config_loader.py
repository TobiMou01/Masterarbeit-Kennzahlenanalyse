"""
Simplified Config Loader
Loads config.yaml with minimal overhead
"""

import yaml
from pathlib import Path
from typing import Any


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
