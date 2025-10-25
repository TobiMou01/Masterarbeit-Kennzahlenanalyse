"""
Environment Setup and venv Checker
Extraced from main.py to keep main clean
"""

import sys
from pathlib import Path


def is_venv_active():
    """Check if a virtual environment is active"""
    return sys.prefix != sys.base_prefix


def check_environment():
    """
    Interactive venv setup checker.
    If no venv is active, prompts user with options.
    Returns True to continue, False to exit.

    MUST be called BEFORE any other imports that require packages!
    """
    if is_venv_active():
        # venv is active, continue silently
        return True

    # No venv active - show interactive dialog
    print("\n" + "=" * 80)
    print("⚠️  KEINE VIRTUELLE UMGEBUNG (venv) AKTIV!")
    print("=" * 80)
    print(f"\n⚙️  Aktuell verwendeter Python-Interpreter:")
    print(f"   {sys.executable}")
    print()
    print("💡 WICHTIG:")
    print("   Wenn du VS Code verwendest und die venv bereits aktiviert ist,")
    print("   aber diese Meldung trotzdem erscheint:")
    print()
    print("   → VS Code verwendet den FALSCHEN Python-Interpreter!")
    print()
    print("   Fix:")
    print("   1. Cmd + Shift + P")
    print("   2. Tippe: 'Python: Select Interpreter'")
    print("   3. Wähle: venv_masterarbeit/bin/python")
    print()
    print("   ODER führe im Terminal aus:")
    print("   → python src/main.py --market germany --compare")
    print("   (NICHT /usr/bin/python3 verwenden!)")
    print()
    print("=" * 80)
    print("\nOptionen:")
    print("[1] venv im Projekt-Ordner nutzen (./venv)")
    print("[2] Externe venv nutzen (/Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit)")
    print("[3] Dependencies prüfen/installieren")
    print("[4] Abbrechen")
    print()

    choice = input("Wahl [1/2/3/4]: ").strip()

    if choice == "1":
        # Check project venv
        venv_path = Path("venv")
        print()
        if venv_path.exists():
            print("✓ venv gefunden!")
            print("\nFühre folgenden Befehl aus:\n")
            print("    source venv/bin/activate")
            print("\nDanach starte das Skript erneut.")
        else:
            print("✗ venv nicht gefunden!")
            print("\nErstelle sie mit:\n")
            print("    python3 -m venv venv")
            print("    source venv/bin/activate")
            print("    pip install -r requirements.txt")
            print("\nDanach starte das Skript erneut.")
        print()
        sys.exit(0)

    elif choice == "2":
        # Check external venv
        external_venv = Path("/Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit")
        print()
        if external_venv.exists():
            print("✓ Externe venv gefunden!")
            print("\nFühre folgenden Befehl aus:\n")
            print("    source /Users/tobi/masterarbeit-kennzahlenanalyse/venv_masterarbeit/bin/activate")
            print("\nDanach starte das Skript erneut.")
        else:
            print("✗ Externe venv nicht gefunden unter:")
            print(f"   {external_venv}")
            print("\nBitte prüfe den Pfad oder wähle Option [1].")
        print()
        sys.exit(0)

    elif choice == "3":
        # Show dependency info
        print()
        print("📦 Dependencies aus requirements.txt:")
        print()
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            print("Installiere alle mit:\n")
            print("    pip install -r requirements.txt")
            print()
            print("Oder einzeln prüfen:\n")
            print("    pip list | grep -E \"pandas|numpy|scikit-learn|matplotlib|seaborn|openpyxl|pyyaml|scipy|joblib\"")
            print()
            print("\nBenötigte Pakete:")
            with open(requirements_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        print(f"    - {line.strip()}")
        else:
            print("✗ requirements.txt nicht gefunden!")
        print()
        sys.exit(0)

    elif choice == "4":
        # Cancel
        print("\n✓ Abgebrochen.\n")
        sys.exit(0)

    else:
        print(f"\n✗ Ungültige Wahl: '{choice}'")
        print("Bitte wähle 1, 2, 3 oder 4.\n")
        sys.exit(1)
