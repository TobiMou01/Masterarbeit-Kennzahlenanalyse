"""
Setup Phase - Configuration and Output Management
"""

from .logger import setup_logger
from .config_loader import ConfigLoader
from .environment import Environment

# Output Handler - NEW modular structure
from .path_manager import PathManager
from .data_formatter import DataFormatter
from .file_writer import FileWriter
from .output_coordinator import OutputHandler  # Backward compatible name

# Legacy (kept for compatibility)
# from .output_handler import OutputHandler

__all__ = [
    'setup_logger',
    'ConfigLoader',
    'Environment',
    'PathManager',
    'DataFormatter',
    'FileWriter',
    'OutputHandler',
]
