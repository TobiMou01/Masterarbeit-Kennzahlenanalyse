"""
Config Exporter - Reproduzierbarkeit für Masterarbeit

Exportiert die verwendete Config in den Output-Ordner für:
- Reproduzierbarkeit der Analysen
- Nachvollziehbarkeit für Jupyter Notebooks
- Dokumentation der Parameter-Einstellungen
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConfigExporter:
    """Exports configuration to output directory for reproducibility"""

    def __init__(self, config: dict, market: str = 'germany'):
        """
        Initialize config exporter

        Args:
            config: Configuration dictionary
            market: Market name
        """
        self.config = config
        self.market = market

    def export_config(self, output_dir: Path = None, format: str = 'yaml') -> str:
        """
        Export configuration to output directory

        Args:
            output_dir: Output directory (defaults to output/{market}/00_config/)
            format: Export format ('yaml' or 'json')

        Returns:
            Path to exported config file
        """
        if output_dir is None:
            output_dir = Path(f'output/{self.market}/00_config')

        output_dir.mkdir(parents=True, exist_ok=True)

        # Add metadata
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'market': self.market,
                'config_version': '1.0',
                'description': 'Configuration snapshot for reproducibility'
            },
            'config': self.config
        }

        # Export based on format
        if format == 'yaml':
            output_file = output_dir / 'analysis_config.yaml'
            with open(output_file, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            output_file = output_dir / 'analysis_config.json'
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"✓ Config exported: {output_file}")
        return str(output_file)

    def export_run_summary(self, output_dir: Path = None, **kwargs) -> str:
        """
        Export run summary with timestamps and metrics

        Args:
            output_dir: Output directory (defaults to output/{market}/00_config/)
            **kwargs: Additional run metadata (e.g., runtime, n_companies, etc.)

        Returns:
            Path to exported summary file
        """
        if output_dir is None:
            output_dir = Path(f'output/{self.market}/00_config')

        output_dir.mkdir(parents=True, exist_ok=True)

        summary_data = {
            'run_timestamp': datetime.now().isoformat(),
            'market': self.market,
            **kwargs
        }

        output_file = output_dir / 'run_summary.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False)

        logger.info(f"✓ Run summary exported: {output_file}")
        return str(output_file)


def export_config_snapshot(config: dict, market: str = 'germany', output_dir: Path = None) -> str:
    """
    Quick function to export config snapshot

    Args:
        config: Configuration dictionary
        market: Market name
        output_dir: Output directory (optional)

    Returns:
        Path to exported config file
    """
    exporter = ConfigExporter(config, market)
    return exporter.export_config(output_dir)


if __name__ == "__main__":
    # Test
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("CONFIG EXPORTER MODULE TEST")
    print("=" * 80)

    # Load example config
    from src._01_setup import config_loader
    config = config_loader.load()

    # Export
    exporter = ConfigExporter(config, market='germany')
    config_file = exporter.export_config()
    print(f"\n✓ Config exported to: {config_file}")

    summary_file = exporter.export_run_summary(
        runtime_seconds=123.45,
        n_companies=100,
        n_clusters=5
    )
    print(f"✓ Summary exported to: {summary_file}")
    print("=" * 80)
