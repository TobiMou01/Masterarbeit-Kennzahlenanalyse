"""
Shared utilities for interactive Jupyter notebooks
Provides common functions for loading, saving, and visualizing data
"""

import sys
from pathlib import Path
import pickle
import json
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Plotly imports for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ipywidgets for interactive controls
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️ ipywidgets not available - some interactive features disabled")

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


# ============================================================================
# INTERACTIVE PLOTLY VISUALIZATIONS
# ============================================================================

def create_interactive_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: Optional[str] = None,
    color: str = 'cluster',
    hover_data: Optional[List[str]] = None,
    title: str = "Interactive Scatter Plot",
    size: Optional[str] = None,
    color_discrete_map: Optional[Dict] = None,
    width: int = 900,
    height: int = 600
) -> go.Figure:
    """
    Create interactive 2D or 3D scatter plot with Plotly

    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        z: Column name for z-axis (optional, creates 3D plot if provided)
        color: Column name for color coding (default: 'cluster')
        hover_data: List of additional columns to show on hover
        title: Plot title
        size: Column name for marker size (optional)
        color_discrete_map: Custom color mapping for clusters
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    # Default hover data includes company_id if available
    if hover_data is None:
        hover_data = ['company_id'] if 'company_id' in df.columns else []

    # Create 3D plot if z is provided
    if z is not None:
        fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color=color,
            hover_data=hover_data,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Bold,
            color_discrete_map=color_discrete_map,
            size=size
        )

        # Update 3D layout
        fig.update_layout(
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=width,
            height=height,
            hovermode='closest'
        )
    else:
        # Create 2D plot
        fig = px.scatter(
            df, x=x, y=y,
            color=color,
            hover_data=hover_data,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Bold,
            color_discrete_map=color_discrete_map,
            size=size
        )

        fig.update_layout(
            width=width,
            height=height,
            hovermode='closest'
        )

    # Update marker styling
    fig.update_traces(
        marker=dict(size=8 if size is None else None, opacity=0.7, line=dict(width=0.5, color='white'))
    )

    return fig


def create_interactive_cluster_distribution(
    df: pd.DataFrame,
    cluster_col: str = 'cluster',
    title: str = "Cluster Distribution",
    show_pie: bool = True,
    width: int = 1200,
    height: int = 500
) -> go.Figure:
    """
    Create interactive cluster distribution with bar chart and optional pie chart

    Args:
        df: DataFrame with cluster assignments
        cluster_col: Column name containing cluster labels
        title: Plot title
        show_pie: Whether to show pie chart alongside bar chart
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    cluster_counts = df[cluster_col].value_counts().sort_index()
    total = len(df)
    percentages = (cluster_counts / total * 100).round(1)

    if show_pie:
        # Create subplots with bar and pie
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=(f"{title} - Counts", f"{title} - Proportions")
        )

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                text=[f"{count}<br>({pct}%)" for count, pct in zip(cluster_counts.values, percentages.values)],
                textposition='auto',
                marker=dict(color=cluster_counts.index, colorscale='Viridis'),
                hovertemplate='<b>Cluster %{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>',
                name='Cluster Size'
            ),
            row=1, col=1
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=[f"Cluster {i}" for i in cluster_counts.index],
                values=cluster_counts.values,
                text=[f"{pct}%" for pct in percentages.values],
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                marker=dict(colors=px.colors.qualitative.Bold[:len(cluster_counts)])
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Cluster", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

    else:
        # Single bar chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                text=[f"{count} ({pct}%)" for count, pct in zip(cluster_counts.values, percentages.values)],
                textposition='auto',
                marker=dict(color=cluster_counts.index, colorscale='Viridis'),
                hovertemplate='<b>Cluster %{x}</b><br>Count: %{y}<br><extra></extra>'
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Cluster",
            yaxis_title="Count"
        )

    fig.update_layout(width=width, height=height, showlegend=False)
    return fig


def create_interactive_heatmap(
    data: pd.DataFrame,
    title: str = "Heatmap",
    x_label: str = "Features",
    y_label: str = "Clusters",
    colorscale: str = "RdBu_r",
    show_values: bool = True,
    width: int = 1000,
    height: int = 600
) -> go.Figure:
    """
    Create interactive heatmap with hover information

    Args:
        data: DataFrame with values (rows=clusters, columns=features)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        colorscale: Plotly colorscale name
        show_values: Whether to show values on cells
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale,
        text=data.values.round(3) if show_values else None,
        texttemplate='%{text}' if show_values else None,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Value")
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height
    )

    return fig


def create_parallel_coordinates(
    df: pd.DataFrame,
    features: List[str],
    cluster_col: str = 'cluster',
    title: str = "Parallel Coordinates Plot",
    width: int = 1200,
    height: int = 600
) -> go.Figure:
    """
    Create parallel coordinates plot for multi-dimensional feature exploration

    Args:
        df: DataFrame with features and cluster assignments
        features: List of feature columns to include
        cluster_col: Column name containing cluster labels
        title: Plot title
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    # Prepare dimensions for parallel coordinates
    dimensions = []
    for feature in features:
        dimensions.append(
            dict(
                label=feature,
                values=df[feature],
                range=[df[feature].min(), df[feature].max()]
            )
        )

    # Create color mapping for clusters
    clusters = df[cluster_col].unique()
    colors = px.colors.qualitative.Bold[:len(clusters)]
    color_map = {cluster: i for i, cluster in enumerate(sorted(clusters))}

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[cluster_col].map(color_map),
                colorscale=[[i/(len(clusters)-1), c] for i, c in enumerate(colors)],
                showscale=True,
                cmin=0,
                cmax=len(clusters)-1,
                colorbar=dict(
                    title="Cluster",
                    tickvals=list(range(len(clusters))),
                    ticktext=[str(c) for c in sorted(clusters)]
                )
            ),
            dimensions=dimensions
        )
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height
    )

    return fig


def create_metrics_dashboard(
    k_range: range,
    inertias: List[float],
    silhouettes: List[float],
    calinskis: List[float],
    davies_bouldins: List[float],
    optimal_k: int = None,
    title: str = "K-Means Optimal K Selection",
    width: int = 1400,
    height: int = 800
) -> go.Figure:
    """
    Create interactive dashboard for K-Means optimal K selection

    Args:
        k_range: Range of K values tested
        inertias: Inertia values (Elbow method)
        silhouettes: Silhouette scores
        calinskis: Calinski-Harabasz scores
        davies_bouldins: Davies-Bouldin scores (lower is better)
        optimal_k: Optimal K to highlight (optional)
        title: Dashboard title
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    k_values = list(k_range)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Elbow Method (Inertia)",
            "Silhouette Score",
            "Calinski-Harabasz Score",
            "Davies-Bouldin Score"
        )
    )

    # Elbow plot
    fig.add_trace(
        go.Scatter(
            x=k_values, y=inertias,
            mode='lines+markers',
            name='Inertia',
            marker=dict(size=8, color='steelblue'),
            line=dict(width=2),
            hovertemplate='K=%{x}<br>Inertia=%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=k_values, y=silhouettes,
            mode='lines+markers',
            name='Silhouette',
            marker=dict(size=8, color='green'),
            line=dict(width=2),
            hovertemplate='K=%{x}<br>Silhouette=%{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Calinski-Harabasz plot
    fig.add_trace(
        go.Scatter(
            x=k_values, y=calinskis,
            mode='lines+markers',
            name='Calinski-Harabasz',
            marker=dict(size=8, color='orange'),
            line=dict(width=2),
            hovertemplate='K=%{x}<br>CH Score=%{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Davies-Bouldin plot
    fig.add_trace(
        go.Scatter(
            x=k_values, y=davies_bouldins,
            mode='lines+markers',
            name='Davies-Bouldin',
            marker=dict(size=8, color='red'),
            line=dict(width=2),
            hovertemplate='K=%{x}<br>DB Score=%{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )

    # Highlight optimal K if provided
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        # Add vertical lines at optimal K
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(
                    x=optimal_k, line_dash="dash", line_color="red",
                    annotation_text=f"Optimal K={optimal_k}",
                    row=row, col=col
                )

    # Update axes
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=2, col=2)

    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_yaxes(title_text="CH Score", row=2, col=1)
    fig.update_yaxes(title_text="DB Score (lower is better)", row=2, col=2)

    fig.update_layout(
        title_text=title,
        showlegend=False,
        width=width,
        height=height
    )

    return fig


def create_k_distance_plot(
    distances: np.ndarray,
    optimal_eps: Optional[float] = None,
    title: str = "K-Distance Graph (DBSCAN eps Selection)",
    width: int = 1000,
    height: int = 600
) -> go.Figure:
    """
    Create interactive K-distance plot for DBSCAN eps parameter selection

    Args:
        distances: Sorted K-nearest neighbor distances
        optimal_eps: Optimal eps value to highlight (optional)
        title: Plot title
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Main distance curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(distances))),
            y=distances,
            mode='lines',
            name='K-Distance',
            line=dict(color='steelblue', width=2),
            hovertemplate='Point: %{x}<br>Distance: %{y:.3f}<extra></extra>'
        )
    )

    # Highlight optimal eps if provided
    if optimal_eps is not None:
        fig.add_hline(
            y=optimal_eps,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal eps = {optimal_eps:.3f}",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Points (sorted by distance)",
        yaxis_title="K-th Nearest Neighbor Distance",
        width=width,
        height=height,
        hovermode='x'
    )

    return fig


def compare_clusters_interactive(
    df: pd.DataFrame,
    algorithms: List[str],
    x: str,
    y: str,
    title: str = "Cluster Comparison Across Algorithms",
    width: int = 1400,
    height: int = 500
) -> go.Figure:
    """
    Create side-by-side comparison of clustering results from different algorithms

    Args:
        df: DataFrame with cluster assignments from different algorithms
        algorithms: List of algorithm names (column names with cluster assignments)
        x: X-axis feature (e.g., 'PC1')
        y: Y-axis feature (e.g., 'PC2')
        title: Plot title
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure object
    """
    n_algorithms = len(algorithms)

    fig = make_subplots(
        rows=1, cols=n_algorithms,
        subplot_titles=[alg.replace('_', ' ').title() for alg in algorithms],
        shared_xaxes=True,
        shared_yaxes=True
    )

    for i, algorithm in enumerate(algorithms, 1):
        # Get cluster column
        cluster_col = f'{algorithm}_cluster' if not algorithm.endswith('_cluster') else algorithm

        if cluster_col not in df.columns:
            print(f"Warning: Column '{cluster_col}' not found in DataFrame")
            continue

        # Create scatter for each cluster
        for cluster in sorted(df[cluster_col].unique()):
            cluster_df = df[df[cluster_col] == cluster]

            fig.add_trace(
                go.Scatter(
                    x=cluster_df[x],
                    y=cluster_df[y],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(size=8, opacity=0.7),
                    text=cluster_df.get('company_id', ''),
                    hovertemplate='<b>%{text}</b><br>%s: %{x:.2f}<br>%s: %{y:.2f}<extra></extra>' % (x, y),
                    showlegend=(i == 1)  # Only show legend for first subplot
                ),
                row=1, col=i
            )

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y, col=1)

    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        hovermode='closest'
    )

    return fig
