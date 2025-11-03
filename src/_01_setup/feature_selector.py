"""
Feature Selector - Zentrale Feature-Verwaltung mit Metadaten
Verwaltet Base Features (12) und Extended Features (25+) für Clustering-Analysen
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Zentrale Klasse für Feature-Verwaltung und -Selektion

    Funktionen:
    - Liefert Base Features (12) für Standard-Analysen
    - Liefert Extended Features (25+) für PCA-Analysen
    - Bietet Metadaten für jedes Feature (Formel, Kategorie, Range, Interpretation)
    - Gruppiert Features nach Kategorien (Profitability, Leverage, etc.)
    - Validiert Feature-Verfügbarkeit in DataFrames
    """

    def __init__(self, config_path: str = 'features_config.yaml'):
        """
        Initialisiert FeatureSelector

        Args:
            config_path: Pfad zur features_config.yaml
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Base Features (12) - manuell kuratiert für beste Clustering-Ergebnisse
        self._base_features = [
            'roa',
            'roe',
            'ebit_margin',
            'debt_to_equity',
            'current_ratio',
            'asset_turnover',
            'revenue_growth',
            'net_profit_margin',
            'quick_ratio',
            'interest_coverage',
            'fcf_margin',
            'asset_growth'
        ]

        logger.info(f"✓ FeatureSelector initialisiert mit {len(self._base_features)} Base Features")

    def _load_config(self) -> dict:
        """Lädt features_config.yaml"""
        if not self.config_path.exists():
            logger.error(f"❌ Features Config nicht gefunden: {self.config_path}")
            return {}

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.debug(f"✓ Features Config geladen: {self.config_path}")
        return config

    # =========================================================================
    # FEATURE SELECTION METHODS
    # =========================================================================

    def get_base_features(self) -> List[str]:
        """
        Liefert 12 Base Features für Standard-Clustering

        Returns:
            Liste mit 12 Feature-Namen
        """
        return self._base_features.copy()

    def get_extended_features(self) -> List[str]:
        """
        Liefert 25+ Extended Features für PCA-Analysen

        Nutzt das 'pca_optimized' Preset aus features_config.yaml

        Returns:
            Liste mit 25+ Feature-Namen
        """
        if 'presets' in self.config and 'pca_optimized' in self.config['presets']:
            preset = self.config['presets']['pca_optimized']

            # Extrahiere alle Features aus dem Preset
            extended_features = []
            if 'categories' in preset:
                for category_dict in preset['categories']:
                    for category_name, features in category_dict.items():
                        extended_features.extend(features)

            logger.info(f"✓ Extended Features: {len(extended_features)} (aus pca_optimized preset)")
            return extended_features
        else:
            # Fallback: alle enabled Features sammeln
            return self._collect_all_enabled_features()

    def get_features_by_preset(self, preset_name: str) -> List[str]:
        """
        Liefert Features basierend auf Preset-Namen

        Args:
            preset_name: 'minimal', 'standard', 'comprehensive', 'pca_optimized'

        Returns:
            Liste mit Feature-Namen
        """
        if 'presets' not in self.config or preset_name not in self.config['presets']:
            logger.warning(f"⚠️  Preset '{preset_name}' nicht gefunden, nutze base_features")
            return self.get_base_features()

        preset = self.config['presets'][preset_name]

        # Spezialfall: comprehensive = alle Features
        if preset.get('use_all', False):
            return self._collect_all_enabled_features()

        # Standard: Features aus Kategorien sammeln
        features = []
        if 'categories' in preset:
            for category_dict in preset['categories']:
                for category_name, feature_list in category_dict.items():
                    features.extend(feature_list)

        logger.info(f"✓ Preset '{preset_name}': {len(features)} Features")
        return features

    def _collect_all_enabled_features(self) -> List[str]:
        """Sammelt alle enabled Features aus allen Kategorien"""
        all_features = []

        # Static Features
        for category_name in ['profitability', 'liquidity', 'leverage', 'efficiency', 'cashflow', 'structure']:
            if category_name in self.config and self.config[category_name].get('enabled', False):
                metrics = self.config[category_name].get('metrics', {})
                for feature_name, feature_config in metrics.items():
                    if feature_config.get('enabled', False):
                        all_features.append(feature_name)

        # Dynamic Features (falls gewünscht)
        if 'dynamic' in self.config and self.config['dynamic'].get('enabled', False):
            for subcategory in ['growth', 'trends', 'volatility', 'quality']:
                if subcategory in self.config['dynamic']:
                    subcategory_config = self.config['dynamic'][subcategory]
                    if subcategory_config.get('enabled', False):
                        metrics = subcategory_config.get('metrics', {})
                        for feature_name, feature_config in metrics.items():
                            if feature_config.get('enabled', False):
                                all_features.append(feature_name)

        logger.info(f"✓ Alle enabled Features: {len(all_features)}")
        return all_features

    # =========================================================================
    # FEATURE METADATA
    # =========================================================================

    def get_feature_metadata(self, feature_name: str) -> Dict:
        """
        Liefert Metadaten für ein Feature

        Args:
            feature_name: Name des Features (z.B. 'roa')

        Returns:
            Dict mit: name, formula, category, unit, interpretation, typical_range
        """
        # Suche Feature in allen Kategorien
        for category_name in ['profitability', 'liquidity', 'leverage', 'efficiency', 'cashflow', 'structure']:
            if category_name in self.config:
                metrics = self.config[category_name].get('metrics', {})
                if feature_name in metrics:
                    feature_config = metrics[feature_name]
                    return {
                        'name': self._get_feature_display_name(feature_name),
                        'formula': feature_config.get('formula', 'N/A'),
                        'category': category_name.capitalize(),
                        'unit': feature_config.get('unit', 'N/A'),
                        'interpretation': feature_config.get('interpretation', 'N/A'),
                        'typical_range': self._infer_typical_range(feature_name, category_name),
                        'enabled': feature_config.get('enabled', False)
                    }

        # Suche in Dynamic Features
        if 'dynamic' in self.config:
            for subcategory in ['growth', 'trends', 'volatility', 'quality']:
                if subcategory in self.config['dynamic']:
                    metrics = self.config['dynamic'][subcategory].get('metrics', {})
                    if feature_name in metrics:
                        feature_config = metrics[feature_name]
                        return {
                            'name': self._get_feature_display_name(feature_name),
                            'formula': feature_config.get('formula', 'N/A'),
                            'category': f"Dynamic - {subcategory.capitalize()}",
                            'unit': feature_config.get('unit', 'N/A'),
                            'interpretation': feature_config.get('interpretation', 'N/A'),
                            'typical_range': [None, None],  # Dynamische Features haben keine fixen Ranges
                            'enabled': feature_config.get('enabled', False)
                        }

        # Feature nicht gefunden
        logger.warning(f"⚠️  Metadaten für Feature '{feature_name}' nicht gefunden")
        return {
            'name': feature_name,
            'formula': 'Unknown',
            'category': 'Unknown',
            'unit': 'N/A',
            'interpretation': 'N/A',
            'typical_range': [None, None],
            'enabled': False
        }

    def _get_feature_display_name(self, feature_name: str) -> str:
        """Konvertiert Feature-Name zu Display-Name"""
        name_mapping = {
            'roa': 'Return on Assets',
            'roe': 'Return on Equity',
            'ebit_margin': 'EBIT Margin',
            'ebitda_margin': 'EBITDA Margin',
            'net_profit_margin': 'Net Profit Margin',
            'operating_margin': 'Operating Margin',
            'gross_margin': 'Gross Margin',
            'roc': 'Return on Capital',
            'current_ratio': 'Current Ratio',
            'quick_ratio': 'Quick Ratio',
            'cash_ratio': 'Cash Ratio',
            'debt_to_equity': 'Debt-to-Equity',
            'total_debt_to_equity': 'Total Debt-to-Equity',
            'equity_ratio': 'Equity Ratio',
            'debt_ratio': 'Debt Ratio',
            'interest_coverage': 'Interest Coverage',
            'asset_turnover': 'Asset Turnover',
            'revenue_per_employee': 'Revenue per Employee',
            'fcf_margin': 'Free Cash Flow Margin',
            'revenue_growth': 'Revenue Growth',
            'asset_growth': 'Asset Growth'
        }
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

    def _infer_typical_range(self, feature_name: str, category: str) -> List[float]:
        """Inferiert typische Ranges basierend auf Kategorie"""
        # Hardcoded typical ranges für wichtigste Features
        typical_ranges = {
            'roa': [0.0, 0.20],
            'roe': [0.0, 0.25],
            'ebit_margin': [0.0, 0.25],
            'net_profit_margin': [0.0, 0.20],
            'current_ratio': [0.5, 3.0],
            'quick_ratio': [0.3, 2.0],
            'debt_to_equity': [0.0, 2.0],
            'asset_turnover': [0.5, 3.0],
            'interest_coverage': [1.0, 10.0],
            'fcf_margin': [-0.10, 0.20]
        }

        return typical_ranges.get(feature_name, [None, None])

    def get_features_by_category(self) -> Dict[str, List[str]]:
        """
        Gruppiert alle enabled Features nach Kategorien

        Returns:
            Dict mit Kategorien als Keys und Feature-Listen als Values
        """
        categorized_features = {
            'Profitability': [],
            'Liquidity': [],
            'Leverage': [],
            'Efficiency': [],
            'Cashflow': [],
            'Structure': [],
            'Growth': [],
            'Trends': [],
            'Volatility': [],
            'Quality': []
        }

        # Static Features
        for category_name in ['profitability', 'liquidity', 'leverage', 'efficiency', 'cashflow', 'structure']:
            if category_name in self.config and self.config[category_name].get('enabled', False):
                metrics = self.config[category_name].get('metrics', {})
                for feature_name, feature_config in metrics.items():
                    if feature_config.get('enabled', False):
                        categorized_features[category_name.capitalize()].append(feature_name)

        # Dynamic Features
        if 'dynamic' in self.config and self.config['dynamic'].get('enabled', False):
            for subcategory in ['growth', 'trends', 'volatility', 'quality']:
                if subcategory in self.config['dynamic']:
                    subcategory_config = self.config['dynamic'][subcategory]
                    if subcategory_config.get('enabled', False):
                        metrics = subcategory_config.get('metrics', {})
                        for feature_name, feature_config in metrics.items():
                            if feature_config.get('enabled', False):
                                categorized_features[subcategory.capitalize()].append(feature_name)

        # Entferne leere Kategorien
        categorized_features = {k: v for k, v in categorized_features.items() if v}

        return categorized_features

    # =========================================================================
    # FEATURE VALIDATION
    # =========================================================================

    def validate_features(
        self,
        features: List[str],
        df: pd.DataFrame,
        missing_threshold: float = 0.5,
        outlier_threshold: float = 0.1
    ) -> Tuple[List[str], Dict]:
        """
        Validiert Feature-Verfügbarkeit in DataFrame

        Args:
            features: Liste zu prüfender Features
            df: DataFrame mit Daten
            missing_threshold: Max. erlaubter Missing-Anteil (0.5 = 50%)
            outlier_threshold: Max. erlaubter Outlier-Anteil (0.1 = 10%)

        Returns:
            (valid_features, validation_report)
        """
        validation_report = {
            'total_features': len(features),
            'valid_features': [],
            'invalid_features': [],
            'missing_features': [],
            'high_missing_rate': [],
            'high_outlier_rate': [],
            'warnings': []
        }

        for feature in features:
            # Check 1: Existiert die Spalte?
            if feature not in df.columns:
                validation_report['missing_features'].append(feature)
                validation_report['invalid_features'].append(feature)
                validation_report['warnings'].append(f"{feature}: Column not found in DataFrame")
                continue

            # Check 2: Missing Values Rate
            missing_rate = df[feature].isna().sum() / len(df)
            if missing_rate > missing_threshold:
                validation_report['high_missing_rate'].append({
                    'feature': feature,
                    'missing_rate': missing_rate
                })
                validation_report['invalid_features'].append(feature)
                validation_report['warnings'].append(
                    f"{feature}: High missing rate ({missing_rate:.1%} > {missing_threshold:.1%})"
                )
                continue

            # Check 3: Outlier Rate (IQR-Methode)
            outlier_rate = self._calculate_outlier_rate(df[feature])
            if outlier_rate > outlier_threshold:
                validation_report['high_outlier_rate'].append({
                    'feature': feature,
                    'outlier_rate': outlier_rate
                })
                # Outliers sind kein Hard-Blocker, nur Warning
                validation_report['warnings'].append(
                    f"{feature}: High outlier rate ({outlier_rate:.1%})"
                )

            # Feature ist valide
            validation_report['valid_features'].append(feature)

        # Logging
        logger.info(f"\n{'='*80}")
        logger.info("FEATURE VALIDATION REPORT")
        logger.info(f"{'='*80}")
        logger.info(f"Total Features: {validation_report['total_features']}")
        logger.info(f"✓ Valid: {len(validation_report['valid_features'])}")
        logger.info(f"✗ Invalid: {len(validation_report['invalid_features'])}")

        if validation_report['missing_features']:
            logger.warning(f"  Missing: {validation_report['missing_features']}")
        if validation_report['high_missing_rate']:
            logger.warning(f"  High Missing Rate: {[x['feature'] for x in validation_report['high_missing_rate']]}")
        if validation_report['high_outlier_rate']:
            logger.info(f"  High Outlier Rate: {[x['feature'] for x in validation_report['high_outlier_rate']]}")

        logger.info(f"{'='*80}\n")

        return validation_report['valid_features'], validation_report

    def _calculate_outlier_rate(self, series: pd.Series) -> float:
        """Berechnet Outlier-Rate mit IQR-Methode"""
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return 1.0  # 100% outliers wenn keine Daten

        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = (series_clean < lower_bound) | (series_clean > upper_bound)
        outlier_rate = outliers.sum() / len(series_clean)

        return outlier_rate

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def print_feature_summary(self, features: List[str] = None):
        """Druckt Feature-Übersicht"""
        if features is None:
            features = self.get_base_features()

        print(f"\n{'='*80}")
        print(f"FEATURE SUMMARY ({len(features)} features)")
        print(f"{'='*80}\n")

        for feature in features:
            metadata = self.get_feature_metadata(feature)
            print(f"• {feature}")
            print(f"  Name: {metadata['name']}")
            print(f"  Category: {metadata['category']}")
            print(f"  Formula: {metadata['formula']}")
            print(f"  Interpretation: {metadata['interpretation']}")
            print()

    def __repr__(self):
        base_count = len(self._base_features)
        extended_count = len(self.get_extended_features())
        return f"FeatureSelector(base={base_count}, extended={extended_count})"


if __name__ == "__main__":
    # Test FeatureSelector
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("FEATURE SELECTOR TEST")
    print("="*80)

    selector = FeatureSelector()

    # Test 1: Base Features
    print("\n1. Base Features:")
    base_features = selector.get_base_features()
    print(f"   Count: {len(base_features)}")
    print(f"   Features: {base_features}")

    # Test 2: Extended Features
    print("\n2. Extended Features:")
    extended_features = selector.get_extended_features()
    print(f"   Count: {len(extended_features)}")
    print(f"   Features: {extended_features[:10]}...")  # Erste 10

    # Test 3: Metadata
    print("\n3. Feature Metadata (ROA):")
    roa_metadata = selector.get_feature_metadata('roa')
    for key, value in roa_metadata.items():
        print(f"   {key}: {value}")

    # Test 4: By Category
    print("\n4. Features by Category:")
    by_category = selector.get_features_by_category()
    for category, features in list(by_category.items())[:3]:  # Erste 3 Kategorien
        print(f"   {category}: {len(features)} features")

    # Test 5: Validation (Mock DataFrame)
    print("\n5. Feature Validation:")
    mock_df = pd.DataFrame({
        'roa': np.random.randn(100),
        'roe': np.random.randn(100),
        'ebit_margin': np.random.randn(100),
        # debt_to_equity absichtlich fehlen lassen
    })
    valid_features, report = selector.validate_features(
        ['roa', 'roe', 'ebit_margin', 'debt_to_equity'],
        mock_df
    )
    print(f"   Valid Features: {valid_features}")
    print(f"   Invalid Features: {report['invalid_features']}")

    print("\n✓ FeatureSelector Test erfolgreich!")
    print("="*80 + "\n")
