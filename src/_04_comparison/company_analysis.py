"""
Company-Level Cluster Analysis
Generates Excel overview showing cluster assignments per company across all algorithms
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def create_company_cluster_excel(
    algorithm_results: Dict,
    market: str = 'germany',
    output_dir: str = None
) -> str:
    """
    Erstellt Excel-Datei mit Cluster-Zuordnungen pro Unternehmen

    Args:
        algorithm_results: Dict mit Ergebnissen aller Algorithmen
            Format: {
                'kmeans': {'static': {'df': df}, 'dynamic': {...}, 'combined': {...}},
                'hierarchical': {'static': {'df': df}, ...},
                'dbscan': {'static': {'df': df}, ...}
            }
        market: Market-Bezeichnung
        output_dir: Output-Verzeichnis (optional)

    Returns:
        Path to created Excel file
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING COMPANY-LEVEL CLUSTER ANALYSIS")
    logger.info("=" * 80)

    # Output path
    if output_dir is None:
        output_dir = f'output/{market}'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_file = output_path / 'company_cluster_analysis.xlsx'

    # Start with base company data from first available dataset
    base_df = None
    for algo_name, results in algorithm_results.items():
        if results.get('static') and 'df' in results['static']:
            base_df = results['static']['df'][['gvkey', 'conm', 'gsector', 'ggroup', 'gind', 'gsubind']].copy()
            break

    if base_df is None:
        logger.error("❌ Keine Daten gefunden für Company Analysis")
        return None

    # Rename conm to company_name
    base_df = base_df.rename(columns={'conm': 'company_name'})
    base_df = base_df.drop_duplicates(subset=['gvkey']).set_index('gvkey')

    logger.info(f"→ Basis: {len(base_df)} Unternehmen")

    # Add cluster assignments from each algorithm
    cluster_columns = []

    # K-Means (comparative mode - 3 separate clusterings)
    if 'kmeans' in algorithm_results:
        kmeans_results = algorithm_results['kmeans']

        for stage in ['static', 'dynamic', 'combined']:
            if stage in kmeans_results and 'df' in kmeans_results[stage]:
                df = kmeans_results[stage]['df']
                col_name = f'kmeans_{stage}'

                # Create cluster assignment dict
                cluster_map = df.set_index('gvkey')['cluster'].to_dict()
                base_df[col_name] = base_df.index.map(cluster_map)
                cluster_columns.append(col_name)

                logger.info(f"  ✓ K-Means {stage}: {base_df[col_name].notna().sum()} Zuordnungen")

    # Hierarchical (uses same labels across stages)
    if 'hierarchical' in algorithm_results:
        hier_results = algorithm_results['hierarchical']

        # Use static (master) labels
        if 'static' in hier_results and 'df' in hier_results['static']:
            df = hier_results['static']['df']
            col_name = 'hierarchical'

            cluster_map = df.set_index('gvkey')['cluster'].to_dict()
            base_df[col_name] = base_df.index.map(cluster_map)
            cluster_columns.append(col_name)

            logger.info(f"  ✓ Hierarchical: {base_df[col_name].notna().sum()} Zuordnungen")

    # DBSCAN (uses same labels across stages)
    if 'dbscan' in algorithm_results:
        dbscan_results = algorithm_results['dbscan']

        # Use static (master) labels
        if 'static' in dbscan_results and 'df' in dbscan_results['static']:
            df = dbscan_results['static']['df']
            col_name = 'dbscan'

            cluster_map = df.set_index('gvkey')['cluster'].to_dict()
            base_df[col_name] = base_df.index.map(cluster_map)
            cluster_columns.append(col_name)

            # Count noise points (-1)
            noise_count = (base_df[col_name] == -1).sum()
            valid_count = (base_df[col_name] >= 0).sum()
            logger.info(f"  ✓ DBSCAN: {valid_count} Zuordnungen, {noise_count} Noise")

    # Calculate agreement metrics
    logger.info("\n→ Berechne Übereinstimmungs-Metriken...")

    # Match count: Wie viele Algorithmen haben Daten für dieses Unternehmen?
    base_df['data_coverage'] = base_df[cluster_columns].notna().sum(axis=1)

    # Agreement analysis (nur für Unternehmen mit Daten von allen Algorithmen)
    companies_with_all = base_df[base_df['data_coverage'] == len(cluster_columns)]

    if len(companies_with_all) > 0:
        logger.info(f"  → {len(companies_with_all)} Unternehmen haben Daten von allen Algorithmen")

        # Simple agreement: All algorithms assign to same cluster ID?
        # (This is simplistic since cluster IDs may not be comparable across algorithms)
        # Better: Use this to identify companies where algorithms strongly disagree

        def analyze_agreement(row):
            """Analysiert Übereinstimmung zwischen Algorithmen"""
            clusters = [row[col] for col in cluster_columns if pd.notna(row[col])]

            if len(clusters) == 0:
                return 0, "no_data"

            # Count unique cluster assignments
            unique_clusters = len(set(clusters))

            if unique_clusters == 1:
                return len(clusters), "all_agree"
            elif unique_clusters == len(clusters):
                return 0, "all_disagree"
            else:
                return len(clusters) - unique_clusters, "partial_agreement"

        base_df[['agreement_count', 'agreement_type']] = base_df.apply(
            analyze_agreement, axis=1, result_type='expand'
        )
    else:
        base_df['agreement_count'] = 0
        base_df['agreement_type'] = 'insufficient_data'

    # Add notes column with interpretations
    def create_notes(row):
        """Erstellt Notizen basierend auf Cluster-Zuordnungen"""
        notes = []

        # DBSCAN noise
        if 'dbscan' in row and row['dbscan'] == -1:
            notes.append("DBSCAN: Noise/Outlier")

        # Agreement type
        if row['agreement_type'] == 'all_agree':
            notes.append("Alle Algorithmen einig")
        elif row['agreement_type'] == 'all_disagree':
            notes.append("Starke Uneinigkeit zwischen Algorithmen")

        # Coverage
        if row['data_coverage'] < len(cluster_columns):
            notes.append(f"Nur {row['data_coverage']}/{len(cluster_columns)} Algorithmen")

        return "; ".join(notes) if notes else ""

    base_df['notes'] = base_df.apply(create_notes, axis=1)

    # Reset index for Excel export
    base_df = base_df.reset_index()

    # Reorder columns for better readability
    ordered_columns = ['gvkey', 'company_name', 'gsector', 'ggroup', 'gind', 'gsubind']
    ordered_columns.extend(cluster_columns)
    ordered_columns.extend(['data_coverage', 'agreement_count', 'agreement_type', 'notes'])

    # Ensure all columns exist
    final_columns = [col for col in ordered_columns if col in base_df.columns]
    base_df = base_df[final_columns]

    # Save to Excel with formatting
    logger.info(f"\n→ Speichere Excel...")

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Main sheet
        base_df.to_excel(writer, sheet_name='Company Clusters', index=False)

        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Companies',
                'Companies with all algorithms',
                'Companies with partial data',
                'K-Means static clusters',
                'Hierarchical clusters',
                'DBSCAN clusters (excl. noise)',
                'DBSCAN noise points'
            ],
            'Value': [
                len(base_df),
                len(base_df[base_df['data_coverage'] == len(cluster_columns)]),
                len(base_df[base_df['data_coverage'] < len(cluster_columns)]),
                base_df['kmeans_static'].nunique() if 'kmeans_static' in base_df else 0,
                base_df['hierarchical'].nunique() if 'hierarchical' in base_df else 0,
                len(base_df[base_df['dbscan'] >= 0]['dbscan'].unique()) if 'dbscan' in base_df else 0,
                (base_df['dbscan'] == -1).sum() if 'dbscan' in base_df else 0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Agreement analysis sheet
        if len(companies_with_all) > 0:
            agreement_summary = base_df['agreement_type'].value_counts().reset_index()
            agreement_summary.columns = ['Agreement Type', 'Count']
            agreement_summary.to_excel(writer, sheet_name='Agreement Analysis', index=False)

    logger.info(f"✓ Excel erstellt: {excel_file}")
    logger.info(f"  → {len(base_df)} Unternehmen")
    logger.info(f"  → {len(cluster_columns)} Algorithmus-Spalten")
    logger.info("=" * 80)

    return str(excel_file)


if __name__ == "__main__":
    # Test function
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "=" * 80)
    print("COMPANY ANALYSIS MODULE TEST")
    print("=" * 80)
    print("\nThis module needs to be called from ComparisonPipeline with actual results.")
    print("See ComparisonPipeline.run_full_comparison_pipeline() for integration.")
    print("=" * 80)
