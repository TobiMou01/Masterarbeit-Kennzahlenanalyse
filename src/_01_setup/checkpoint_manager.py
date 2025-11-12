"""
Checkpoint Manager - Für Jupyter Notebook Workflows

Ermöglicht:
- Zwischenspeichern von Pipeline-Ergebnissen
- Schnelles Laden für iterative Analysen
- On-the-fly Config-Änderungen ohne Neuberechnung
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for Jupyter notebook workflows"""

    def __init__(self, market: str = 'germany', checkpoint_dir: Path = None):
        """
        Initialize checkpoint manager

        Args:
            market: Market name
            checkpoint_dir: Directory for checkpoints (defaults to output/{market}/01_data/checkpoints/)
        """
        self.market = market
        if checkpoint_dir is None:
            self.checkpoint_dir = Path(f'output/{market}/01_data/checkpoints')
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, data: Any, name: str, metadata: Dict = None) -> str:
        """
        Save checkpoint

        Args:
            data: Data to checkpoint (can be dict, dataframe, model, etc.)
            name: Checkpoint name (e.g., 'preprocessing_complete', 'kmeans_results')
            metadata: Optional metadata dictionary

        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f'{name}_{timestamp}.pkl'

        # Prepare checkpoint package
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'market': self.market,
            'metadata': metadata or {},
            'data': data
        }

        # Save with pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save metadata as JSON for readability
        metadata_file = checkpoint_file.with_suffix('.json')
        meta = {
            'timestamp': checkpoint['timestamp'],
            'name': name,
            'market': self.market,
            'file': str(checkpoint_file),
            'metadata': metadata or {}
        }
        with open(metadata_file, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✓ Checkpoint saved: {checkpoint_file.name}")
        return str(checkpoint_file)

    def load_checkpoint(self, name: str, latest: bool = True) -> Optional[Any]:
        """
        Load checkpoint

        Args:
            name: Checkpoint name
            latest: If True, load latest checkpoint with this name

        Returns:
            Checkpoint data or None if not found
        """
        # Find checkpoint files with this name
        checkpoint_files = sorted(self.checkpoint_dir.glob(f'{name}_*.pkl'))

        if not checkpoint_files:
            logger.warning(f"⚠ No checkpoint found for: {name}")
            return None

        # Load latest or specific
        if latest:
            checkpoint_file = checkpoint_files[-1]  # Last = latest
        else:
            checkpoint_file = checkpoint_files[0]

        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            logger.info(f"✓ Checkpoint loaded: {checkpoint_file.name}")
            return checkpoint['data']

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return None

    def list_checkpoints(self) -> Dict[str, list]:
        """
        List all available checkpoints

        Returns:
            Dict mapping checkpoint names to list of timestamps
        """
        checkpoints = {}

        for pkl_file in sorted(self.checkpoint_dir.glob('*.pkl')):
            # Extract name and timestamp from filename
            # Format: {name}_{timestamp}.pkl
            parts = pkl_file.stem.rsplit('_', 2)
            if len(parts) >= 2:
                name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                timestamp = '_'.join(parts[-2:])

                if name not in checkpoints:
                    checkpoints[name] = []
                checkpoints[name].append(timestamp)

        return checkpoints

    def delete_checkpoint(self, name: str, all: bool = False) -> int:
        """
        Delete checkpoints

        Args:
            name: Checkpoint name
            all: If True, delete all checkpoints with this name

        Returns:
            Number of deleted checkpoints
        """
        checkpoint_files = sorted(self.checkpoint_dir.glob(f'{name}_*.pkl'))

        if not all and checkpoint_files:
            checkpoint_files = [checkpoint_files[-1]]  # Only delete latest

        deleted = 0
        for pkl_file in checkpoint_files:
            # Delete pickle file
            pkl_file.unlink()
            deleted += 1

            # Delete metadata JSON if exists
            json_file = pkl_file.with_suffix('.json')
            if json_file.exists():
                json_file.unlink()

        logger.info(f"✓ Deleted {deleted} checkpoint(s) for: {name}")
        return deleted

    def get_checkpoint_info(self, name: str) -> Optional[Dict]:
        """
        Get checkpoint metadata without loading full data

        Args:
            name: Checkpoint name

        Returns:
            Metadata dictionary or None
        """
        checkpoint_files = sorted(self.checkpoint_dir.glob(f'{name}_*.pkl'))

        if not checkpoint_files:
            return None

        # Load metadata JSON (faster than pickle)
        json_file = checkpoint_files[-1].with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)

        return None


def save_checkpoint(data: Any, name: str, market: str = 'germany', metadata: Dict = None) -> str:
    """
    Quick function to save checkpoint

    Args:
        data: Data to checkpoint
        name: Checkpoint name
        market: Market name
        metadata: Optional metadata

    Returns:
        Path to checkpoint file
    """
    manager = CheckpointManager(market)
    return manager.save_checkpoint(data, name, metadata)


def load_checkpoint(name: str, market: str = 'germany') -> Optional[Any]:
    """
    Quick function to load checkpoint

    Args:
        name: Checkpoint name
        market: Market name

    Returns:
        Checkpoint data or None
    """
    manager = CheckpointManager(market)
    return manager.load_checkpoint(name)


if __name__ == "__main__":
    # Test
    import logging
    import pandas as pd
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("CHECKPOINT MANAGER MODULE TEST")
    print("=" * 80)

    # Create test data
    test_data = pd.DataFrame({
        'company': ['A', 'B', 'C'],
        'cluster': [0, 1, 0],
        'score': [85, 72, 91]
    })

    # Save checkpoint
    manager = CheckpointManager(market='germany')
    checkpoint_file = manager.save_checkpoint(
        test_data,
        name='test_results',
        metadata={'n_companies': 3, 'n_clusters': 2}
    )
    print(f"\n✓ Saved: {checkpoint_file}")

    # List checkpoints
    print("\n→ Available checkpoints:")
    checkpoints = manager.list_checkpoints()
    for name, timestamps in checkpoints.items():
        print(f"  • {name}: {len(timestamps)} checkpoint(s)")

    # Load checkpoint
    loaded_data = manager.load_checkpoint('test_results')
    print(f"\n✓ Loaded checkpoint:")
    print(loaded_data)

    # Cleanup
    manager.delete_checkpoint('test_results', all=True)
    print("\n✓ Cleanup complete")
    print("=" * 80)
