"""
Output Reorganization Utilities
Reorganizes output structure into clean hybrid architecture:
OVERVIEW → ALGORITHMS → COMPARISONS → DETAILS
"""

import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputReorganizer:
    """Reorganizes clustering output into clean hybrid architecture"""

    def __init__(self, market: str, base_output_dir: str = "output", keep_details: bool = False):
        """
        Initialize reorganizer

        Args:
            market: Market name (e.g., 'germany')
            base_output_dir: Base output directory
            keep_details: If True, keeps DETAILS folder with all original outputs
        """
        self.market = market
        self.base_output_dir = Path(base_output_dir)
        self.market_dir = self.base_output_dir / market
        self.keep_details = keep_details

        # Track statistics
        self.stats = {
            'files_moved': 0,
            'duplicates_removed': 0,
            'space_saved_mb': 0,
            'errors': []
        }

        # Deletion log
        self.deletion_log: List[Dict] = []

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def find_duplicate_files(self, directory: Path) -> Dict[str, List[Path]]:
        """
        Find duplicate files by MD5 hash

        Returns:
            Dictionary mapping hash to list of file paths
        """
        hash_map: Dict[str, List[Path]] = {}

        for file_path in directory.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    file_hash = self.calculate_md5(file_path)
                    if file_hash not in hash_map:
                        hash_map[file_hash] = []
                    hash_map[file_hash].append(file_path)
                except Exception as e:
                    logger.warning(f"Could not hash {file_path}: {e}")

        # Only return hashes with duplicates
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    def select_best_file(self, duplicates: List[Path]) -> Path:
        """
        Select best version from duplicates
        Priority: combined > static > dynamic, newer > older
        """
        # Prefer 'combined' folder
        for file_path in duplicates:
            if 'combined' in str(file_path):
                return file_path

        # Prefer 'static' over 'dynamic'
        for file_path in duplicates:
            if 'static' in str(file_path):
                return file_path

        # Return newest by modification time
        return max(duplicates, key=lambda p: p.stat().st_mtime)

    def create_new_structure(self) -> Dict[str, Path]:
        """Create new folder structure with _new suffix to avoid conflicts"""
        structure = {
            'overview': self.market_dir / '_new_overview',
            'overview_clusters': self.market_dir / '_new_overview' / 'cluster_lists',
            'algorithms': self.market_dir / '_new_algorithms',
            'comparisons': self.market_dir / '_new_comparisons',
            'comp_gics': self.market_dir / '_new_comparisons' / '01_vs_gics',
            'comp_gics_tables': self.market_dir / '_new_comparisons' / '01_vs_gics' / 'contingency_tables',
            'comp_algo': self.market_dir / '_new_comparisons' / '02_algorithms',
            'comp_features': self.market_dir / '_new_comparisons' / '03_feature_importance',
            'comp_temporal': self.market_dir / '_new_comparisons' / '04_temporal',
            'details': self.market_dir / '_new_details' if self.keep_details else None,
        }

        for key, path in structure.items():
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")

        return structure

    def backup_original(self):
        """Create backup of original structure"""
        backup_dir = self.market_dir.parent / f"{self.market}_backup_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.market_dir.exists():
            logger.info(f"Creating backup: {backup_dir}")
            shutil.copytree(self.market_dir, backup_dir)
            logger.info(f"✓ Backup created successfully")
        else:
            logger.warning(f"Market directory does not exist: {self.market_dir}")

    def reorganize(self, create_backup: bool = True) -> Dict:
        """
        Main reorganization method

        Args:
            create_backup: If True, creates backup before reorganizing

        Returns:
            Statistics dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔄 STARTING OUTPUT REORGANIZATION")
        logger.info("=" * 80)
        logger.info(f"Market: {self.market}")
        logger.info(f"Base directory: {self.market_dir}")
        logger.info(f"Keep details: {self.keep_details}")

        # Create backup
        if create_backup:
            self.backup_original()

        # Find duplicates BEFORE reorganization
        logger.info("\n📊 Scanning for duplicate files...")
        duplicates = self.find_duplicate_files(self.market_dir)
        logger.info(f"Found {len(duplicates)} sets of duplicate files")

        # Remove duplicates
        self._remove_duplicates(duplicates)

        # Create new structure
        logger.info("\n📁 Creating new folder structure...")
        new_structure = self.create_new_structure()

        # Migrate files
        logger.info("\n🚚 Migrating files to new structure...")
        self._migrate_comparisons(new_structure)
        self._migrate_algorithms(new_structure)
        self._create_overview(new_structure)

        # Generate reports
        logger.info("\n📝 Generating navigation and summaries...")
        self._generate_readme(new_structure)

        # Save deletion log
        self._save_deletion_log()

        # Finalize: Remove old folders and rename new ones
        logger.info("\n🔧 Finalizing structure...")
        self._finalize_structure()

        # Print summary
        self._print_summary()

        return self.stats

    def _remove_duplicates(self, duplicates: Dict[str, List[Path]]):
        """Remove duplicate files, keeping only the best version"""
        for file_hash, file_list in duplicates.items():
            best_file = self.select_best_file(file_list)

            for file_path in file_list:
                if file_path != best_file:
                    try:
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        file_path.unlink()

                        self.stats['duplicates_removed'] += 1
                        self.stats['space_saved_mb'] += file_size

                        self.deletion_log.append({
                            'file': str(file_path),
                            'reason': 'duplicate',
                            'kept_version': str(best_file),
                            'size_mb': round(file_size, 2)
                        })

                        logger.debug(f"Removed duplicate: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")
                        self.stats['errors'].append(str(e))

    def _migrate_comparisons(self, structure: Dict[str, Path]):
        """Migrate comparison files to COMPARISONS folder"""
        old_comp_dir = self.market_dir / 'comparisons'

        if not old_comp_dir.exists():
            logger.warning(f"Comparisons directory not found: {old_comp_dir}")
            return

        # Map old to new structure
        mappings = {
            '01_gics_comparison': structure['comp_gics'],
            '02_algorithm_comparison': structure['comp_algo'],
            '03_feature_importance': structure['comp_features'],
            '04_temporal_stability': structure['comp_temporal'],
        }

        for old_name, new_path in mappings.items():
            old_path = old_comp_dir / old_name
            if old_path.exists():
                self._copy_directory_contents(old_path, new_path)

    def _migrate_algorithms(self, structure: Dict[str, Path]):
        """Migrate algorithm outputs to ALGORITHMS folder"""
        algorithms = ['kmeans', 'hierarchical', 'dbscan']

        for algo in algorithms:
            old_algo_dir = self.market_dir / algo
            if not old_algo_dir.exists():
                continue

            # Create algorithm directory
            new_algo_dir = structure['algorithms'] / algo
            new_algo_dir.mkdir(exist_ok=True)

            # Only keep 'combined' folder
            old_combined = old_algo_dir / 'combined'
            if old_combined.exists():
                new_combined = new_algo_dir / 'combined'
                self._copy_directory_contents(old_combined, new_combined)

    def _create_overview(self, structure: Dict[str, Path]):
        """Create OVERVIEW folder with key files"""
        overview_dir = structure['overview']

        # Copy metrics comparison if exists
        comp_algo_dir = structure['comp_algo']
        metrics_file = comp_algo_dir / 'algorithm_metrics_comparison.csv'
        if metrics_file.exists():
            shutil.copy2(metrics_file, overview_dir / 'metrics_comparison.csv')
            self.stats['files_moved'] += 1

        # Generate executive summary
        self._generate_executive_summary(structure)

        # Generate combined cluster lists
        self._generate_combined_cluster_lists(structure)

    def _copy_directory_contents(self, src: Path, dst: Path):
        """Copy directory contents, tracking moved files"""
        dst.mkdir(parents=True, exist_ok=True)

        for item in src.rglob('*'):
            if item.is_file() and not item.name.startswith('.'):
                rel_path = item.relative_to(src)
                dst_file = dst / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(item, dst_file)
                    self.stats['files_moved'] += 1
                except Exception as e:
                    logger.error(f"Error copying {item}: {e}")
                    self.stats['errors'].append(str(e))

    def _generate_executive_summary(self, structure: Dict[str, Path]):
        """Generate executive summary with key findings"""
        summary_path = structure['overview'] / 'executive_summary.txt'

        # Try to load metrics comparison
        metrics_file = structure['overview'] / 'metrics_comparison.csv'

        summary_lines = [
            "=" * 80,
            "EXECUTIVE SUMMARY",
            "=" * 80,
            f"Market: {self.market.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERVIEW:",
        ]

        if metrics_file.exists():
            try:
                df = pd.read_csv(metrics_file)

                # Find best algorithm (example: highest silhouette score)
                if 'silhouette_score' in df.columns:
                    best_idx = df['silhouette_score'].idxmax()
                    best_algo = df.loc[best_idx, 'algorithm']
                    best_silhouette = df.loc[best_idx, 'silhouette_score']

                    summary_lines.extend([
                        f"- Best Algorithm: {best_algo.upper()}",
                        f"  Silhouette Score: {best_silhouette:.3f}",
                        ""
                    ])

                summary_lines.append("ALGORITHM COMPARISON:")
                summary_lines.append(df.to_string(index=False))

            except Exception as e:
                summary_lines.append(f"- Could not load metrics: {e}")
        else:
            summary_lines.append("- Metrics comparison not available")

        summary_lines.extend([
            "",
            "=" * 80,
            "FOLDER STRUCTURE:",
            "=" * 80,
            "OVERVIEW/          - Key results and navigation",
            "ALGORITHMS/        - Detailed clustering results per algorithm",
            "COMPARISONS/       - All comparison analyses",
            "  01_vs_gics/      - Clustering vs GICS sectors",
            "  02_algorithms/   - Algorithm performance comparison",
            "  03_feature_importance/ - Feature importance analysis",
            "  04_temporal/     - Temporal stability analysis",
            "",
            "See README.md for detailed navigation guide.",
            "=" * 80
        ])

        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))

        logger.info(f"✓ Executive summary created: {summary_path}")

    def _generate_combined_cluster_lists(self, structure: Dict[str, Path]):
        """Generate combined Excel files with all clusters for each algorithm"""
        algorithms = ['kmeans', 'hierarchical', 'dbscan']
        cluster_lists_dir = structure['overview_clusters']

        for algo in algorithms:
            algo_dir = structure['algorithms'] / algo / 'combined'

            if not algo_dir.exists():
                continue

            # Look for cluster assignments CSV
            assignments_file = algo_dir / 'cluster_assignments.csv'

            if assignments_file.exists():
                try:
                    df = pd.read_csv(assignments_file)

                    # Create Excel with sheets per cluster
                    excel_path = cluster_lists_dir / f"{algo}_combined.xlsx"

                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        # Overall summary sheet
                        df.to_excel(writer, sheet_name='All_Companies', index=False)

                        # Separate sheet per cluster
                        if 'cluster' in df.columns:
                            for cluster_id in sorted(df['cluster'].unique()):
                                cluster_df = df[df['cluster'] == cluster_id]
                                sheet_name = f"Cluster_{cluster_id}"
                                cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    logger.info(f"✓ Created combined cluster list: {excel_path.name}")
                    self.stats['files_moved'] += 1

                except Exception as e:
                    logger.error(f"Error creating cluster list for {algo}: {e}")
                    self.stats['errors'].append(str(e))

    def _generate_readme(self, structure: Dict[str, Path]):
        """Generate README.md for navigation"""
        readme_path = structure['overview'] / 'README.md'

        readme_content = f"""# Output Navigation - {self.market.upper()} Market

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start

1. **Executive Summary**: [`executive_summary.txt`](executive_summary.txt)
   - Key findings and best algorithm recommendation

2. **Cluster Lists**: [`cluster_lists/`](cluster_lists/)
   - Company assignments per algorithm
   - Excel files with separate sheets per cluster

3. **Metrics Comparison**: [`metrics_comparison.csv`](metrics_comparison.csv)
   - Algorithm performance comparison

## Folder Structure

### OVERVIEW/ (You are here)
Quick access to key results and navigation

### ALGORITHMS/
Detailed clustering results for each algorithm:
- `kmeans/combined/` - K-Means results
- `hierarchical/combined/` - Hierarchical clustering results
- `dbscan/combined/` - DBSCAN results

Each folder contains:
- `cluster_assignments.csv` - Company cluster assignments
- `cluster_profiles.csv` - Cluster characteristics
- `plots/` - Visualizations

### COMPARISONS/
Comparative analyses:

#### 01_vs_gics/
Clustering results vs GICS sector classification
- Cramér's V correlation analysis
- Contingency tables
- Comparison plots per algorithm

#### 02_algorithms/
Algorithm performance comparison
- Metrics comparison (Silhouette, Davies-Bouldin, etc.)
- Best algorithm recommendation

#### 03_feature_importance/
Feature importance analysis
- Combined plot across all algorithms
- Detailed CSV files per algorithm

#### 04_temporal/
Temporal stability analysis
- Cluster migration patterns
- Stability metrics
- Migration plots per algorithm

{"### DETAILS/\nOriginal analysis outputs (static/dynamic/combined)\n" if self.keep_details else ""}

## File Naming Conventions

- `*_combined.*` - Analysis using combined static + dynamic features
- `*_static.*` - Analysis using only static features
- `*_dynamic.*` - Analysis using only dynamic features

## Tools Used

- K-Means Clustering
- Hierarchical Clustering (Ward's method)
- DBSCAN (Density-based clustering)

---
*Generated automatically by OutputReorganizer*
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.info(f"✓ README created: {readme_path}")

    def _save_deletion_log(self):
        """Save deletion log to file"""
        if not self.deletion_log:
            return

        log_path = self.market_dir / '_new_overview' / 'deletion_log.txt'

        log_lines = [
            "=" * 80,
            "DELETION LOG",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total files removed: {len(self.deletion_log)}",
            f"Total space saved: {self.stats['space_saved_mb']:.2f} MB",
            "",
            "DELETED FILES:",
            "-" * 80,
        ]

        for entry in self.deletion_log:
            log_lines.extend([
                f"File: {entry['file']}",
                f"  Reason: {entry['reason']}",
                f"  Kept: {entry['kept_version']}",
                f"  Size: {entry['size_mb']} MB",
                ""
            ])

        log_lines.append("=" * 80)

        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines))

        logger.info(f"✓ Deletion log saved: {log_path}")

    def _finalize_structure(self):
        """
        Remove old folders and rename new ones
        Old folders to remove: kmeans, hierarchical, dbscan, visualizations, clusters, reports
        New folders to rename: _new_overview -> OVERVIEW, etc.
        """
        # Old folders to remove
        old_folders = [
            'kmeans', 'hierarchical', 'dbscan',
            'visualizations', 'clusters', 'reports',
            'kmeans, hierarchical, dbscan'  # Handle weird folder name
        ]

        # Only remove if not using comparisons (case-insensitive check)
        if not (self.market_dir / 'comparisons').exists():
            old_folders.append('comparisons')

        for folder_name in old_folders:
            folder_path = self.market_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                try:
                    shutil.rmtree(folder_path)
                    logger.debug(f"Removed old folder: {folder_name}")
                except Exception as e:
                    logger.warning(f"Could not remove {folder_path}: {e}")

        # Rename new folders (remove _new_ prefix)
        rename_map = {
            '_new_overview': 'OVERVIEW',
            '_new_algorithms': 'ALGORITHMS',
            '_new_comparisons': 'COMPARISONS',
            '_new_details': 'DETAILS'
        }

        for old_name, new_name in rename_map.items():
            old_path = self.market_dir / old_name
            new_path = self.market_dir / new_name

            if old_path.exists():
                try:
                    # If target already exists, remove it first
                    if new_path.exists():
                        shutil.rmtree(new_path)

                    old_path.rename(new_path)
                    logger.debug(f"Renamed: {old_name} -> {new_name}")
                except Exception as e:
                    logger.error(f"Could not rename {old_path}: {e}")
                    self.stats['errors'].append(str(e))

        logger.info("✓ Structure finalized")

    def _print_summary(self):
        """Print reorganization summary"""
        logger.info("\n" + "=" * 80)
        logger.info("✓ REORGANIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Files moved: {self.stats['files_moved']}")
        logger.info(f"  Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"  Space saved: {self.stats['space_saved_mb']:.2f} MB")
        logger.info(f"  Errors: {len(self.stats['errors'])}")
        logger.info(f"\n📂 New structure: {self.market_dir}/")
        logger.info("   ├── OVERVIEW/")
        logger.info("   ├── ALGORITHMS/")
        logger.info("   ├── COMPARISONS/")
        if self.keep_details:
            logger.info("   └── DETAILS/")
        logger.info("\n👉 See OVERVIEW/README.md for navigation")
        logger.info("=" * 80)


def reorganize_output(market: str, base_output_dir: str = "output",
                     keep_details: bool = False, create_backup: bool = True) -> Dict:
    """
    Convenience function to reorganize output

    Args:
        market: Market name
        base_output_dir: Base output directory
        keep_details: Keep DETAILS folder with original outputs
        create_backup: Create backup before reorganizing

    Returns:
        Statistics dictionary
    """
    reorganizer = OutputReorganizer(market, base_output_dir, keep_details)
    return reorganizer.reorganize(create_backup=create_backup)
