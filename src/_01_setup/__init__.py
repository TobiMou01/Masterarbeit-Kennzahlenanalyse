"""
Setup Phase - Configuration and Output Management
"""

# Core infrastructure functions
from .logger import setup_logging
from . import config_loader  # Module with functions: load_config, get_value, etc.
from . import environment     # Module with functions: is_venv_active, check_environment

# Feature management
from .feature_config_loader import FeatureConfigLoader
from .feature_selector import FeatureSelector
from .interactive_menu import InteractiveMenu

# Output Handler - NEW modular structure
from .path_manager import PathManager
from .data_formatter import DataFormatter
from .file_writer import FileWriter
from .output_coordinator import OutputHandler  # Backward compatible name

# Reproducibility & Jupyter support
from . import config_exporter
from . import checkpoint_manager

__all__ = [
    'setup_logging',
    'config_loader',
    'environment',
    'FeatureConfigLoader',
    'FeatureSelector',
    'InteractiveMenu',
    'PathManager',
    'DataFormatter',
    'FileWriter',
    'OutputHandler',
    'config_exporter',
    'checkpoint_manager',
]
