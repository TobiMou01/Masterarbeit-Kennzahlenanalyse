"""
Interactive Menu System
Provides user-friendly terminal interface for configuring analysis runs
"""

import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any

from src._01_setup import config_loader as config


class InteractiveMenu:
    """Interactive menu for clustering analysis configuration"""

    def __init__(self, config_dict: dict):
        self.config = config_dict
        self.session_config = {}
        self.last_session_file = Path('.last_session.yaml')

    def run(self) -> Dict[str, Any]:
        """
        Run interactive menu and return session configuration

        Returns:
            Dict with user selections
        """
        self._print_header()

        # Step 1: Preset or custom
        preset = self._select_preset()

        if preset:
            # Load preset configuration
            self.session_config = self._load_preset(preset)
            self._print_summary()

            if not self._confirm_proceed():
                print("\n❌ Abgebrochen.")
                sys.exit(0)
        else:
            # Custom configuration
            self._custom_configuration()

        # Save session for repeat
        self._save_session()

        return self.session_config

    def _print_header(self):
        """Print welcome header"""
        print("\n" + "=" * 80)
        print("🚀 CLUSTERING ANALYSIS PIPELINE - INTERACTIVE MODE")
        print("=" * 80)
        print(f"Market: {self.config.get('data', {}).get('market', 'germany')}")
        print(f"Config: config.yaml\n")

    def _select_preset(self) -> Optional[str]:
        """
        Let user select a preset or custom configuration

        Returns:
            Preset key or None for custom
        """
        print("📋 SETUP-MODUS WÄHLEN")
        print("-" * 80)

        presets = config.get_value(self.config, 'interactive', 'presets', default={})

        # Show preset options
        print("\n[0] Preset laden (vordefinierte Konfiguration)")
        print("[1] Custom Setup (manuelle Konfiguration)")

        if self.last_session_file.exists():
            print("[2] Letzte Session wiederholen")

        print()
        choice = self._get_input("→ Deine Auswahl", valid_options=['0', '1', '2'] if self.last_session_file.exists() else ['0', '1'])

        if choice == '0':
            # Show available presets
            print("\n" + "=" * 80)
            print("VERFÜGBARE PRESETS")
            print("=" * 80)

            preset_keys = list(presets.keys())
            for i, key in enumerate(preset_keys, 1):
                preset = presets[key]
                print(f"\n[{i}] {preset.get('name', key)}")
                print(f"    {preset.get('description', 'Keine Beschreibung')}")

            print(f"\n[{len(preset_keys) + 1}] Abbrechen (Custom Setup)")
            print()

            valid = [str(i) for i in range(1, len(preset_keys) + 2)]
            choice = self._get_input("→ Preset auswählen", valid_options=valid)

            idx = int(choice) - 1
            if idx < len(preset_keys):
                return preset_keys[idx]
            else:
                return None  # Custom

        elif choice == '1':
            return None  # Custom

        elif choice == '2':
            # Load last session
            try:
                import yaml
                with open(self.last_session_file, 'r') as f:
                    self.session_config = yaml.safe_load(f)
                print("\n✓ Letzte Session geladen!")
                self._print_summary()

                if self._confirm_proceed():
                    return 'last_session'
                else:
                    return None  # Go to custom
            except Exception as e:
                print(f"\n⚠️  Fehler beim Laden der letzten Session: {e}")
                return None

    def _load_preset(self, preset_key: str) -> Dict[str, Any]:
        """Load preset configuration"""
        if preset_key == 'last_session':
            return self.session_config

        presets = config.get_value(self.config, 'interactive', 'presets', default={})
        preset = presets.get(preset_key, {})

        print(f"\n✓ Preset geladen: {preset.get('name', preset_key)}")

        return {
            'algorithms': preset.get('algorithms', ['kmeans']),
            'analyses': preset.get('analyses', ['static']),
            'kmeans_mode': preset.get('kmeans_mode', 'comparative'),
            'hierarchical_mode': preset.get('hierarchical_mode', 'hierarchical'),
            'dbscan_mode': preset.get('dbscan_mode', 'hierarchical'),
            'skip_preprocessing': preset.get('skip_preprocessing', False),
            'create_plots': preset.get('create_plots', False),
            'comparison_mode': preset.get('comparison_mode', False),
            'auto_tune_dbscan': preset.get('auto_tune_dbscan', False),
            # New features
            'enable_scoring': preset.get('enable_scoring', True),
            'enable_naming': preset.get('enable_naming', True),
            'naming_method': preset.get('naming_method', 'z_score'),
            'enable_validation': preset.get('enable_validation', True),
            'enable_pca': preset.get('enable_pca', False),
        }

    def _custom_configuration(self):
        """Run custom configuration wizard"""
        print("\n" + "=" * 80)
        print("CUSTOM SETUP")
        print("=" * 80)

        # Question 1: Algorithms
        self.session_config['algorithms'] = self._select_algorithms()

        # Question 2: Analysis modes
        self.session_config['analyses'] = self._select_analyses()

        # Question 3: K-Means mode (if selected)
        if 'kmeans' in self.session_config['algorithms']:
            self.session_config['kmeans_mode'] = self._select_kmeans_mode()

        # Question 4: Hierarchical mode (if selected)
        if 'hierarchical' in self.session_config['algorithms']:
            self.session_config['hierarchical_mode'] = self._select_hierarchical_mode()

        # Question 5: DBSCAN mode (if selected)
        if 'dbscan' in self.session_config['algorithms']:
            self.session_config['dbscan_mode'] = self._select_dbscan_mode()
            self.session_config['auto_tune_dbscan'] = self._select_dbscan_tuning()

        # Question 6: Preprocessing
        self.session_config['skip_preprocessing'] = self._select_preprocessing()

        # Question 7: Visualizations
        self.session_config['create_plots'] = self._select_visualizations()

        # Question 8: Comparison mode
        if len(self.session_config['algorithms']) > 1:
            self.session_config['comparison_mode'] = self._select_comparison()
        else:
            self.session_config['comparison_mode'] = False

        # NEW: Question 9: Scoring
        self.session_config['enable_scoring'] = self._select_scoring()

        # NEW: Question 10: Cluster Naming
        if self.session_config['enable_scoring']:
            self.session_config['enable_naming'] = self._select_naming()
            if self.session_config['enable_naming']:
                self.session_config['naming_method'] = self._select_naming_method()
        else:
            self.session_config['enable_naming'] = False

        # NEW: Question 11: External Validation
        self.session_config['enable_validation'] = self._select_validation()

        # NEW: Question 12: PCA Validation
        self.session_config['enable_pca'] = self._select_pca()

        # Summary
        self._print_summary()

        if not self._confirm_proceed():
            print("\n❌ Abgebrochen.")
            sys.exit(0)

    def _select_algorithms(self) -> List[str]:
        """Select algorithms to run"""
        print("\n" + "-" * 80)
        print("[1] Welche Algorithmen möchtest du ausführen?")
        print("-" * 80)
        print("  [a] K-Means")
        print("  [b] Hierarchical")
        print("  [c] DBSCAN")
        print("  [d] Alle (Comparison Mode)")
        print()

        choice = self._get_input("→ Deine Auswahl (z.B. 'a,c' oder 'd')", allow_multiple=True)

        if 'd' in choice:
            return ['kmeans', 'hierarchical', 'dbscan']

        mapping = {'a': 'kmeans', 'b': 'hierarchical', 'c': 'dbscan'}
        return [mapping[c] for c in choice if c in mapping]

    def _select_analyses(self) -> List[str]:
        """Select analysis modes"""
        print("\n" + "-" * 80)
        print("[2] Welche Analyse-Modi?")
        print("-" * 80)
        print("  [a] Static only")
        print("  [b] Dynamic only")
        print("  [c] Combined only")
        print("  [d] Static + Dynamic")
        print("  [e] Alle drei (Static → Dynamic → Combined)")
        print()

        choice = self._get_input("→ Deine Auswahl")

        mapping = {
            'a': ['static'],
            'b': ['dynamic'],
            'c': ['combined'],
            'd': ['static', 'dynamic'],
            'e': ['static', 'dynamic', 'combined']
        }

        return mapping.get(choice, ['static'])

    def _select_kmeans_mode(self) -> str:
        """Select K-Means mode"""
        print("\n" + "-" * 80)
        print("[3] K-Means Modus wählen:")
        print("-" * 80)
        print("  [a] Comparative (3 separate Clusterings) - zum Vergleichen")
        print("  [b] Hierarchical (Static-Labels behalten) - konsistente Labels")
        print("  [c] Beide Modi parallel ausführen")
        print()

        choice = self._get_input("→ Deine Auswahl")

        mapping = {
            'a': 'comparative',
            'b': 'hierarchical',
            'c': 'both'
        }

        return mapping.get(choice, 'comparative')

    def _select_hierarchical_mode(self) -> str:
        """Select Hierarchical mode"""
        print("\n" + "-" * 80)
        print("[4] Hierarchical Modus wählen:")
        print("-" * 80)
        print("  [a] Hierarchical (Static-Labels behalten) - empfohlen")
        print("  [b] Comparative (3 separate Clusterings) - experimentell")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        mapping = {
            'a': 'hierarchical',
            'b': 'comparative'
        }

        return mapping.get(choice, 'hierarchical')

    def _select_dbscan_mode(self) -> str:
        """Select DBSCAN mode"""
        print("\n" + "-" * 80)
        print("[5] DBSCAN Vorgehensweise:")
        print("-" * 80)
        print("  [a] Hierarchical (Static als Basis, Dynamic anreichern) - empfohlen")
        print("  [b] Comparative (3 separate Clusterings) - experimentell")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        mapping = {
            'a': 'hierarchical',
            'b': 'comparative'
        }

        return mapping.get(choice, 'hierarchical')

    def _select_dbscan_tuning(self) -> bool:
        """Select DBSCAN auto-tuning"""
        print("\n  → Parameter-Tuning bei Problemen aktivieren? (>50% Noise)")
        choice = self._get_input("     [y/N]", default='n')
        return choice.lower() == 'y'

    def _select_preprocessing(self) -> bool:
        """Select preprocessing mode"""
        print("\n" + "-" * 80)
        print("[6] Preprocessing:")
        print("-" * 80)
        print("  [a] Neu ausführen (empfohlen bei Datenänderungen)")
        print("  [b] Vorhandene Daten nutzen (schneller)")
        print()

        choice = self._get_input("→ Deine Auswahl", default='b')

        return choice == 'b'  # True = skip

    def _select_visualizations(self) -> bool:
        """Select visualization mode"""
        print("\n" + "-" * 80)
        print("[7] Visualisierungen erstellen?")
        print("-" * 80)
        print("  [a] Ja, alle Plots erstellen")
        print("  [b] Nein, nur Daten speichern (schneller)")
        print()

        choice = self._get_input("→ Deine Auswahl", default='b')

        return choice == 'a'  # True = create plots

    def _select_comparison(self) -> bool:
        """Select comparison mode"""
        print("\n" + "-" * 80)
        print("[8] Comparison Analysis (bei mehreren Algorithmen):")
        print("-" * 80)
        print("  [a] Ja, Algorithmen vergleichen")
        print("  [b] Nein, nur einzelne Analysen")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        return choice == 'a'

    def _select_scoring(self) -> bool:
        """Select scoring mode"""
        print("\n" + "-" * 80)
        print("[9] Scoring System:")
        print("-" * 80)
        print("  Berechnet Qualitäts-Scores für jedes Unternehmen:")
        print("  • Proximity Score (Nähe zum Cluster-Zentrum)")
        print("  • Dimensional Scores (Profitability, Leverage, etc.)")
        print("  • Relative Score (im Vergleich zu Cluster-Peers)")
        print("  • Overall Score (Gesamtbewertung)")
        print()
        print("  [a] Ja, Scores berechnen (empfohlen)")
        print("  [b] Nein, nur Clustering")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        return choice == 'a'

    def _select_naming(self) -> bool:
        """Select cluster naming mode"""
        print("\n" + "-" * 80)
        print("[10] Cluster Naming:")
        print("-" * 80)
        print("  Generiert beschreibende Namen für Cluster")
        print("  (z.B. 'High ROE, Low Leverage' statt nur 'Cluster 0')")
        print()
        print("  [a] Ja, Namen generieren (empfohlen)")
        print("  [b] Nein, nur Nummern")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        return choice == 'a'

    def _select_naming_method(self) -> str:
        """Select naming method"""
        print("\n  → Naming-Stil:")
        print("     [a] Hybrid (empfohlen: Business-Archetype + technische Details)")
        print("     [b] Technical (detailliert: dominante Features mit Werten)")
        print("     [c] Business (einfach: nur Archetype wie 'Growth Champion')")
        print()

        choice = self._get_input("     Deine Auswahl", default='a')

        mapping = {
            'a': 'hybrid',
            'b': 'technical',
            'c': 'business'
        }

        return mapping.get(choice, 'hybrid')

    def _select_validation(self) -> bool:
        """Select external validation"""
        print("\n" + "-" * 80)
        print("[11] External Validation:")
        print("-" * 80)
        print("  Validiert Cluster gegen externe Labels:")
        print("  • GICS Sectors (Industrie-Klassifikation)")
        print("  • Company Size (Größenkategorien)")
        print("  • Cramér's V, Chi²-Tests, Contingency Tables")
        print()
        print("  [a] Ja, externe Validierung durchführen")
        print("  [b] Nein, überspringen")
        print()

        choice = self._get_input("→ Deine Auswahl", default='a')

        return choice == 'a'

    def _select_pca(self) -> bool:
        """Select PCA validation"""
        print("\n" + "-" * 80)
        print("[12] PCA Validation (Optional):")
        print("-" * 80)
        print("  Führt paralleles Clustering in PCA-Space durch:")
        print("  • Original-Features vs. PCA-transformierte Features")
        print("  • Vergleich der Cluster-Strukturen")
        print("  • Dimensionsreduktion & Interpretation")
        print()
        print("  ⚠️  Benötigt erweiterte Features (pca_optimized preset)")
        print()
        print("  [a] Ja, PCA Validation (dauert länger)")
        print("  [b] Nein, nur Standard-Clustering")
        print()

        choice = self._get_input("→ Deine Auswahl", default='b')

        return choice == 'a'

    def _print_summary(self):
        """Print configuration summary"""
        print("\n" + "=" * 80)
        print("📊 ZUSAMMENFASSUNG")
        print("=" * 80)

        algos = ', '.join(self.session_config.get('algorithms', []))
        analyses = ' → '.join([a.capitalize() for a in self.session_config.get('analyses', [])])

        print(f"  Algorithmen:    {algos}")
        print(f"  Analysen:       {analyses}")

        if 'kmeans' in self.session_config.get('algorithms', []):
            mode = self.session_config.get('kmeans_mode', 'comparative')
            mode_name = {'comparative': 'Comparative', 'hierarchical': 'Hierarchical', 'both': 'Beide Modi'}.get(mode, mode)
            print(f"  K-Means:        {mode_name}")

        if 'hierarchical' in self.session_config.get('algorithms', []):
            mode = self.session_config.get('hierarchical_mode', 'hierarchical')
            print(f"  Hierarchical:   {mode.capitalize()}")

        if 'dbscan' in self.session_config.get('algorithms', []):
            mode = self.session_config.get('dbscan_mode', 'hierarchical')
            print(f"  DBSCAN:         {mode.capitalize()}")

        print(f"  Preprocessing:  {'Übersprungen' if self.session_config.get('skip_preprocessing') else 'Neu ausführen'}")
        print(f"  Plots:          {'Aktiviert' if self.session_config.get('create_plots') else 'Deaktiviert'}")
        print(f"  Comparison:     {'Aktiviert' if self.session_config.get('comparison_mode') else 'Deaktiviert'}")

        print()
        print("  🆕 Erweiterte Features:")
        print(f"  Scoring:        {'Aktiviert' if self.session_config.get('enable_scoring', True) else 'Deaktiviert'}")

        if self.session_config.get('enable_scoring', True):
            naming_status = 'Aktiviert' if self.session_config.get('enable_naming', True) else 'Deaktiviert'
            naming_method = self.session_config.get('naming_method', 'z_score')
            print(f"  Naming:         {naming_status} ({naming_method})")

        print(f"  Validation:     {'Aktiviert' if self.session_config.get('enable_validation', True) else 'Deaktiviert'}")
        print(f"  PCA:            {'Aktiviert' if self.session_config.get('enable_pca', False) else 'Deaktiviert'}")

    def _confirm_proceed(self) -> bool:
        """Ask user to confirm and proceed"""
        print()
        choice = self._get_input("Möchtest du fortfahren? [Y/n]", default='y')
        return choice.lower() != 'n'

    def _get_input(self, prompt: str, valid_options: List[str] = None,
                    default: str = None, allow_multiple: bool = False) -> str:
        """
        Get user input with validation

        Args:
            prompt: Prompt text
            valid_options: List of valid options (None = any input)
            default: Default value if user presses Enter
            allow_multiple: Allow comma-separated multiple selections

        Returns:
            User input (validated)
        """
        while True:
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip() or default
            else:
                user_input = input(f"{prompt}: ").strip()

            if not user_input:
                if default:
                    return default
                print("⚠️  Bitte eine Auswahl treffen.")
                continue

            if allow_multiple:
                # Parse comma-separated input
                choices = [c.strip().lower() for c in user_input.replace(' ', '').split(',')]
                if valid_options:
                    if all(c in valid_options for c in choices):
                        return choices
                    print(f"⚠️  Ungültige Auswahl. Erlaubt: {', '.join(valid_options)}")
                else:
                    return choices
            else:
                user_input = user_input.lower()
                if valid_options is None or user_input in valid_options:
                    return user_input
                print(f"⚠️  Ungültige Auswahl. Erlaubt: {', '.join(valid_options)}")

    def _save_session(self):
        """Save session configuration for repeat"""
        try:
            import yaml
            with open(self.last_session_file, 'w') as f:
                yaml.dump(self.session_config, f)
        except Exception:
            pass  # Silent fail

    @staticmethod
    def pause_for_review(stage: str, output_dir: Path):
        """
        Pause after analysis stage and offer to review results

        Args:
            stage: Stage name (Static, Dynamic, Combined)
            output_dir: Path to output directory
        """
        print("\n" + "=" * 80)
        print(f"✓ {stage} Analysis abgeschlossen")
        print("=" * 80)

        # Ask if user wants to review
        print(f"\n📊 Ergebnisse unter: {output_dir}")
        choice = input("\nMöchtest du die Ergebnisse jetzt sichten? [y/N]: ").strip().lower()

        if choice == 'y':
            # Try to open folder
            try:
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', str(output_dir)])
                elif platform.system() == 'Windows':
                    subprocess.run(['explorer', str(output_dir)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(output_dir)])
                print("✓ Ordner geöffnet")
            except Exception as e:
                print(f"⚠️  Konnte Ordner nicht öffnen: {e}")

        # Ask if continue
        print()
        choice = input(f"Fortfahren mit nächster Stufe? [Y/n/q]: ").strip().lower()

        if choice == 'q' or choice == 'quit':
            print("\n👋 Pipeline abgebrochen.")
            sys.exit(0)
        elif choice == 'n':
            print("\n👋 Pipeline gestoppt.")
            sys.exit(0)

        print()  # Empty line before next stage
