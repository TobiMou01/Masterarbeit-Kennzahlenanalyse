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
