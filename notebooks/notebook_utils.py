"""
Shared utilities for interactive Jupyter notebooks
Provides common functions for loading, saving, and visualizing data
"""

import sys
from pathlib import Path
import pickle
import json
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Default figure size for notebooks
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


class NotebookState:
    """Manages state between notebooks for seamless workflow"""

    def __init__(self, market: str = 'germany'):
        self.market = market
        self.state_dir = PROJECT_ROOT / 'notebooks' / 'state'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / f'{market}_state.pkl'
        self.metadata_file = self.state_dir / f'{market}_metadata.json'

    def save(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Save a value to the state"""
        # Load existing state
        state = self.load_all()
        state[key] = value

        # Save state
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)

        # Update metadata
        meta = self.load_metadata()
        meta[key] = {
            'timestamp': datetime.now().isoformat(),
            'type': type(value).__name__,
            **(metadata or {})
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved '{key}' to state (type: {type(value).__name__})")

    def load(self, key: str, default: Any = None) -> Any:
        """Load a value from the state"""
        state = self.load_all()
        value = state.get(key, default)

        if value is None and default is None:
            print(f"⚠️  Warning: '{key}' not found in state")
        else:
            print(f"✓ Loaded '{key}' from state")

        return value

    def load_all(self) -> Dict:
        """Load entire state"""
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def load_metadata(self) -> Dict:
        """Load state metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def list_keys(self) -> None:
        """List all keys in the state"""
        meta = self.load_metadata()
        if not meta:
            print("No state saved yet")
            return

        print("\n📦 Saved State:")
        print("-" * 80)
        for key, info in meta.items():
            print(f"  {key:30s} | {info['type']:15s} | {info['timestamp']}")
        print("-" * 80)

    def clear(self):
        """Clear all state"""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("✓ State cleared")


def setup_notebook(title: str, market: str = 'germany') -> Tuple[Any, NotebookState]:
    """
    Setup notebook environment

    Returns:
        config: Loaded configuration
        state: NotebookState instance for sharing data between notebooks
    """
    # Load config
    from src._01_setup import config_loader
    cfg = config_loader.load_config(PROJECT_ROOT / 'config.yaml')

    # Create state manager
    state = NotebookState(market)

    # Print header
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"Market: {market}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    return cfg, state


def display_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """Display a nice summary of a dataframe"""
    print(f"\n📊 {name} Summary")
    print("-" * 80)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if 'company_id' in df.columns:
        print(f"Companies: {df['company_id'].nunique():,}")

    if 'year' in df.columns:
        years = df['year'].unique()
        print(f"Years: {min(years)} - {max(years)}")

    print(f"\nColumns: {', '.join(df.columns[:10].tolist())}", end='')
    if len(df.columns) > 10:
        print(f" ... (+{len(df.columns) - 10} more)")
    else:
        print()

    print("-" * 80)


def plot_cluster_distribution(df: pd.DataFrame, cluster_col: str = 'cluster',
                              title: str = "Cluster Distribution"):
    """Plot cluster size distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    cluster_counts = df[cluster_col].value_counts().sort_index()
    ax1.bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Absolute Counts')
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels
    for i, v in enumerate(cluster_counts.values):
        ax1.text(cluster_counts.index[i], v + 5, str(v),
                ha='center', va='bottom', fontsize=9)

    # Pie chart
    ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("husl", len(cluster_counts)))
    ax2.set_title(f'{title} - Proportions')

    plt.tight_layout()
    plt.show()

    return cluster_counts


def plot_feature_distributions(df: pd.DataFrame, features: list,
                               cluster_col: str = 'cluster', max_features: int = 6):
    """Plot feature distributions across clusters"""
    features = features[:max_features]
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Box plot for each cluster
        df.boxplot(column=feature, by=cluster_col, ax=ax)
        ax.set_title(f'{feature} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature)
        plt.sca(ax)
        plt.xticks(rotation=0)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_cluster_profiles(df_profiles: pd.DataFrame, title: str = "Cluster Profiles"):
    """Plot cluster profiles as heatmap"""
    # Normalize for better visualization
    df_norm = df_profiles.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    plt.figure(figsize=(14, max(6, len(df_profiles) * 0.5)))
    sns.heatmap(df_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Normalized Value'})
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.show()


def save_results(output_dir: Path, df: pd.DataFrame,
                prefix: str = "results", formats: list = ['csv', 'excel']):
    """Save results in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if 'csv' in formats:
        csv_path = output_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)
        saved_files.append(csv_path)

    if 'excel' in formats:
        excel_path = output_dir / f"{prefix}.xlsx"
        df.to_excel(excel_path, index=False, engine='openpyxl')
        saved_files.append(excel_path)

    print(f"\n✓ Saved results:")
    for f in saved_files:
        print(f"  → {f}")

    return saved_files


def create_progress_bar(total: int, desc: str = "Processing"):
    """Create a simple text-based progress indicator"""
    from IPython.display import display, HTML
    import time

    class ProgressBar:
        def __init__(self, total, desc):
            self.total = total
            self.current = 0
            self.desc = desc
            self.start_time = time.time()

        def update(self, n=1):
            self.current += n
            pct = self.current / self.total * 100
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0

            bar_length = 40
            filled = int(bar_length * self.current / self.total)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f'\r{self.desc}: [{bar}] {pct:.1f}% ({self.current}/{self.total}) '
                  f'[{elapsed:.1f}s, {rate:.1f}it/s]', end='', flush=True)

            if self.current >= self.total:
                print()  # New line when done

    return ProgressBar(total, desc)
