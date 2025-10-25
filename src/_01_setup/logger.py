"""
Logging Setup
Extracted from main.py for cleaner structure
"""

import logging


def setup_logging(level=logging.INFO):
    """
    Setup logging configuration

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    return logging.getLogger(__name__)
