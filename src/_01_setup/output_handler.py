"""
Output Handler
Vereinfachte, klare Output-Verwaltung
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import joblib

logger = logging.getLogger(__name__)


class OutputHandler:
    """Verwaltet alle Outputs in klarer Struktur"""

    def __init__(self, market: str = 'germany', base_dir: str = 'output', algorithm: str = 'kmeans'):
        """
        Initialisiert Output Handler

        Args:
            market: Market-Bezeichnung (germany, usa, etc.)
            base_dir: Basis-Verzeichnis
            algorithm: Clustering-Algorithmus (z.B. 'kmeans', 'hierarchical', 'dbscan')
        """
        self.market = market
        self.algorithm = algorithm
        # NEW: output/{market}/{algorithm}/ structure
        self.base_dir = Path(base_dir) / market / algorithm

        # Verzeichnisse erstellen - Analysis-type spezifische Struktur
        self.dirs = {
            # Base directories for each analysis type
            'static': self.base_dir / 'static',
            'dynamic': self.base_dir / 'dynamic',
            'combined': self.base_dir / 'combined',

            # Legacy aliases for backward compatibility
            'clusters': self.base_dir / 'static' / 'clusters',  # Default to static
            'reports': self.base_dir / 'static' / 'reports',
            'visualizations': self.base_dir / 'static' / 'visualizations',
            'data': self.base_dir / 'static' / 'reports' / 'data',
            'analysis': self.base_dir / 'static' / 'reports' / 'analysis',
            'models': self.base_dir / 'static' / 'reports' / 'models',
            'plots': self.base_dir / 'static' / 'visualizations'
        }

        # Create analysis-type specific directories
        for analysis_type in ['static', 'dynamic', 'combined']:
            base = self.base_dir / analysis_type
            (base / 'reports' / 'data').mkdir(parents=True, exist_ok=True)
            (base / 'reports' / 'models').mkdir(parents=True, exist_ok=True)
            (base / 'reports' / 'clusters').mkdir(parents=True, exist_ok=True)
            (base / 'reports' / 'analysis').mkdir(parents=True, exist_ok=True)
            (base / 'visualizations').mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Output-Struktur: {self.base_dir}")

    def save_cluster_data(
        self,
        df: pd.DataFrame,
        cluster_profiles: pd.DataFrame,
        analysis_type: str = 'static',
        metrics: Dict = None
    ):
        """
        Speichert Cluster-Zuordnungen und Profile

        Args:
            df: DataFrame mit Cluster-Spalte
            cluster_profiles: Cluster-Profile
            analysis_type: 'static', 'dynamic', oder 'combined'
            metrics: Optional: Clustering-Metriken
        """
        logger.info(f"\n  Speichere {analysis_type} Cluster-Daten...")

        # Get analysis-specific directory
        data_dir = self.base_dir / analysis_type / 'reports' / 'data'

        # 1. Cluster Assignments
        assignments_path = data_dir / 'assignments.csv'
        df.to_csv(assignments_path, index=False)
        logger.info(f"    ✓ Assignments: {assignments_path}")

        # 2. Cluster Profiles
        profiles_path = data_dir / 'profiles.csv'
        cluster_profiles.to_csv(profiles_path)
        logger.info(f"    ✓ Profiles: {profiles_path}")

        # 3. Metrics (optional)
        if metrics:
            import json
            metrics_clean = {k: v for k, v in metrics.items()
                           if not k in ['scaler', 'model']}  # Exclude non-serializable objects
            metrics_path = data_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_clean, f, indent=2)
            logger.info(f"    ✓ Metrics: {metrics_path}")

    def save_cluster_lists(
        self,
        df: pd.DataFrame,
        analysis_type: str = 'static',
        sort_by: str = 'roa'
    ):
        """
        Erstellt separate CSV pro Cluster

        Args:
            df: DataFrame mit Cluster-Spalte
            analysis_type: Analyse-Typ
            sort_by: Sortier-Spalte
        """
        logger.info(f"\n  Erstelle Listen pro Cluster...")

        # Get analysis-specific directory
        clusters_dir = self.base_dir / analysis_type / 'reports' / 'clusters'
        clusters_dir.mkdir(parents=True, exist_ok=True)

        valid_df = df[df['cluster'] >= 0]

        for cluster_id in sorted(valid_df['cluster'].unique()):
            cluster_df = valid_df[valid_df['cluster'] == cluster_id].copy()

            # Cluster-Name
            cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in cluster_df.columns else f'cluster_{cluster_id}'
            safe_name = cluster_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

            # Sortieren
            if sort_by in cluster_df.columns:
                cluster_df = cluster_df.sort_values(sort_by, ascending=False)

            # Speichern
            filename = f'{cluster_id}_{safe_name}.csv'
            path = clusters_dir / filename
            cluster_df.to_csv(path, index=False)

            logger.info(f"    ✓ Cluster {cluster_id}: {len(cluster_df):4} → {filename}")

    def save_analysis_results(
        self,
        df_static: Optional[pd.DataFrame] = None,
        df_dynamic: Optional[pd.DataFrame] = None,
        df_combined: Optional[pd.DataFrame] = None,
        df_migration: Optional[pd.DataFrame] = None
    ):
        """
        Speichert Analyse-Ergebnisse

        Args:
            df_static: Static Analysis Ergebnisse
            df_dynamic: Dynamic Analysis Ergebnisse
            df_combined: Combined Analysis Ergebnisse
            df_migration: Migration Analysis
        """
        logger.info(f"\n  Speichere Analyse-Ergebnisse...")

        if df_static is not None:
            path = self.dirs['analysis'] / 'static_results.csv'
            df_static.to_csv(path, index=False)
            logger.info(f"    ✓ Static Results: {path.name}")

        if df_dynamic is not None:
            path = self.dirs['analysis'] / 'dynamic_results.csv'
            df_dynamic.to_csv(path, index=False)
            logger.info(f"    ✓ Dynamic Results: {path.name}")

        if df_combined is not None:
            path = self.dirs['analysis'] / 'combined_results.csv'
            df_combined.to_csv(path, index=False)
            logger.info(f"    ✓ Combined Results: {path.name}")

        if df_migration is not None:
            path = self.dirs['analysis'] / 'migration_matrix.csv'
            df_migration.to_csv(path, index=False)
            logger.info(f"    ✓ Migration Matrix: {path.name}")

    def save_models(
        self,
        scaler,
        model,
        analysis_type: str = 'static'
    ):
        """
        Speichert Modelle

        Args:
            scaler: StandardScaler
            model: Clustering Model (any algorithm)
            analysis_type: Analyse-Typ
        """
        # Get analysis-specific directory
        models_dir = self.base_dir / analysis_type / 'reports' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = models_dir / 'scaler.pkl'
        model_path = models_dir / 'model.pkl'

        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)

        logger.info(f"    ✓ Models: {analysis_type} → scaler.pkl, model.pkl")

    def create_summary_report(
        self,
        results: Dict,
        output_file: str = 'summary.txt'
    ):
        """
        Erstellt zusammenfassenden Report

        Args:
            results: Dict mit Ergebnissen
            output_file: Dateiname
        """
        logger.info(f"\n  Erstelle Summary Report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLUSTERING ANALYSIS - SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Market: {self.market.upper()}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Static Analysis
        if 'static' in results:
            report_lines.append("STATIC ANALYSIS (Querschnitt - aktuelles Jahr)")
            report_lines.append("-" * 80)
            self._add_analysis_section(report_lines, results['static'])
            report_lines.append("")

            # Detaillierte Cluster-Informationen
            if 'profiles' in results['static'] and 'df' in results['static']:
                self._add_cluster_details(report_lines, results['static']['profiles'],
                                         results['static']['df'], 'static')

        # Dynamic Analysis
        if 'dynamic' in results:
            report_lines.append("DYNAMIC ANALYSIS (Zeitreihen - Entwicklung)")
            report_lines.append("-" * 80)
            self._add_analysis_section(report_lines, results['dynamic'])
            report_lines.append("")

            # Detaillierte Cluster-Informationen
            if 'profiles' in results['dynamic'] and 'df' in results['dynamic']:
                self._add_cluster_details(report_lines, results['dynamic']['profiles'],
                                         results['dynamic']['df'], 'dynamic')

        # Combined Analysis
        if 'combined' in results:
            report_lines.append("COMBINED ANALYSIS (Statisch + Dynamisch)")
            report_lines.append("-" * 80)
            self._add_analysis_section(report_lines, results['combined'])
            if 'weights' in results['combined']:
                w = results['combined']['weights']
                report_lines.append(f"  Gewichtung: {w['static']*100:.0f}% statisch, {w['dynamic']*100:.0f}% dynamisch")
            report_lines.append("")

            # Detaillierte Cluster-Informationen
            if 'profiles' in results['combined'] and 'df' in results['combined']:
                self._add_cluster_details(report_lines, results['combined']['profiles'],
                                         results['combined']['df'], 'combined')

        # Migration
        if 'migration' in results:
            report_lines.append("CROSS ANALYSIS (Cluster-Migration)")
            report_lines.append("-" * 80)
            m = results['migration']
            report_lines.append(f"  Analyzed Companies: {m.get('total', 'N/A')}")
            if 'patterns' in m:
                report_lines.append(f"  Migration Patterns:")
                for pattern, count in m['patterns'].items():
                    report_lines.append(f"    - {pattern}: {count}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("OUTPUT STRUCTURE")
        report_lines.append("=" * 80)
        report_lines.append(f"")
        report_lines.append(f"  {self.base_dir}/")
        report_lines.append(f"  ├── clusters/          # Pro Cluster eine CSV-Datei")
        report_lines.append(f"  ├── reports/           # Reports, Daten & Modelle")
        report_lines.append(f"  │   ├── data/          # Assignments & Profiles")
        report_lines.append(f"  │   ├── analysis/      # Detaillierte Ergebnisse")
        report_lines.append(f"  │   ├── models/        # Gespeicherte Modelle (Scaler, KMeans)")
        report_lines.append(f"  │   └── summary.txt    # Dieser Report")
        report_lines.append(f"  └── visualizations/    # Plots & Grafiken")
        report_lines.append(f"      ├── static/        # Static Analysis Plots")
        report_lines.append(f"      ├── dynamic/       # Dynamic Analysis Plots")
        report_lines.append(f"      └── combined/      # Combined Analysis Plots")
        report_lines.append("")
        report_lines.append("=" * 80)

        # Speichern
        path = self.dirs['reports'] / output_file
        path.write_text('\n'.join(report_lines))
        logger.info(f"    ✓ Summary: {output_file}")

        return '\n'.join(report_lines)

    def _add_analysis_section(self, lines: List[str], analysis_data: Dict):
        """Fügt Analyse-Section hinzu"""
        lines.append(f"  Klassifizierte Unternehmen: {analysis_data.get('n_companies', 'N/A')}")
        lines.append(f"  Anzahl Cluster: {analysis_data.get('n_clusters', 'N/A')}")

        if 'metrics' in analysis_data:
            m = analysis_data['metrics']
            lines.append(f"  Silhouette Score: {m.get('silhouette', 0):.3f}")
            lines.append(f"  Davies-Bouldin Index: {m.get('davies_bouldin', 0):.3f}")

        if 'duration' in analysis_data:
            lines.append(f"  Dauer: {analysis_data['duration']:.1f}s")

    def _add_cluster_details(
        self,
        lines: List[str],
        profiles: pd.DataFrame,
        df: pd.DataFrame,
        analysis_type: str
    ):
        """
        Fügt detaillierte Cluster-Informationen hinzu

        Args:
            lines: Liste für Report-Zeilen
            profiles: Cluster-Profile DataFrame
            df: Vollständiger DataFrame mit Cluster-Zuordnungen
            analysis_type: 'static', 'dynamic', oder 'combined'
        """
        lines.append("")
        lines.append("  CLUSTER DETAILS:")
        lines.append("  " + "-" * 76)

        # Nur Unternehmen mit gültigen Clustern
        df_valid = df[df['cluster'] >= 0].copy()
        total_companies = len(df_valid)

        # Sortiere Cluster nach ID
        for cluster_id in sorted(profiles.index):
            # Cluster-Daten
            cluster_name = profiles.loc[cluster_id, 'cluster_name']
            cluster_df = df_valid[df_valid['cluster'] == cluster_id]
            count = len(cluster_df)
            pct = (count / total_companies * 100) if total_companies > 0 else 0

            lines.append(f"")
            lines.append(f"  Cluster {cluster_id}: {cluster_name} ({count} Unternehmen, {pct:.1f}%)")

            # Key Metrics basierend auf Analyse-Typ
            if analysis_type == 'static':
                metrics_to_show = ['roa', 'roe', 'ebit_margin', 'equity_ratio', 'debt_to_equity']
                metric_labels = {
                    'roa': 'ROA',
                    'roe': 'ROE',
                    'ebit_margin': 'EBIT Margin',
                    'equity_ratio': 'Equity Ratio',
                    'debt_to_equity': 'Debt/Equity'
                }
            elif analysis_type == 'dynamic':
                metrics_to_show = ['roa', 'roa_trend', 'roa_volatility', 'roe_trend', 'revt_cagr']
                metric_labels = {
                    'roa': 'ROA',
                    'roa_trend': 'ROA Trend',
                    'roa_volatility': 'ROA Volatility',
                    'roe_trend': 'ROE Trend',
                    'revt_cagr': 'Revenue CAGR'
                }
            else:  # combined
                metrics_to_show = ['roa', 'roe', 'ebit_margin', 'roa_trend', 'revt_cagr']
                metric_labels = {
                    'roa': 'ROA',
                    'roe': 'ROE',
                    'ebit_margin': 'EBIT Margin',
                    'roa_trend': 'ROA Trend',
                    'revt_cagr': 'Revenue CAGR'
                }

            # Zeige verfügbare Metriken
            metric_line_parts = []
            for metric in metrics_to_show:
                if metric in profiles.columns:
                    value = profiles.loc[cluster_id, metric]
                    label = metric_labels.get(metric, metric)
                    metric_line_parts.append(f"{label}: {value:.2f}%")

            if metric_line_parts:
                lines.append(f"    Kennzahlen: {' | '.join(metric_line_parts)}")

            # Top 3 Unternehmen im Cluster
            if 'company_name' in cluster_df.columns or 'conm' in cluster_df.columns:
                company_col = 'company_name' if 'company_name' in cluster_df.columns else 'conm'

                # Sortiere nach ROA (falls vorhanden)
                if 'roa' in cluster_df.columns:
                    top_companies = cluster_df.nlargest(3, 'roa')
                else:
                    top_companies = cluster_df.head(3)

                lines.append(f"    Top Unternehmen:")
                for idx, row in top_companies.iterrows():
                    company_name = row[company_col]
                    if pd.notna(company_name):
                        roa_str = f" (ROA: {row['roa']:.2f}%)" if 'roa' in row and pd.notna(row['roa']) else ""
                        lines.append(f"      - {company_name}{roa_str}")

        lines.append("")

    def get_paths(self) -> Dict[str, Path]:
        """
        Gibt alle Pfade zurück

        Returns:
            Dict mit allen Verzeichnis-Pfaden
        """
        return self.dirs.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== OUTPUT HANDLER TEST ===")
    handler = OutputHandler(market='germany')

    print(f"\nBase Dir: {handler.base_dir}")
    print("\nVerzeichnisse:")
    for name, path in handler.dirs.items():
        print(f"  {name:12} → {path}")

    print("\n✓ Output Handler funktioniert!")
