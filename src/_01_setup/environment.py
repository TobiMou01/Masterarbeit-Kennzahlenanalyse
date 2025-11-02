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

    # No venv active - check if venv exists
    venv_path = Path("venv")
    venv_exists = venv_path.exists() and (venv_path / "bin" / "python").exists()

    # Show interactive dialog
    print("\n" + "=" * 80)
    print("⚠️  KEINE VIRTUELLE UMGEBUNG (venv) AKTIV!")
    print("=" * 80)
    print(f"\n⚙️  Aktuell verwendeter Python-Interpreter:")
    print(f"   {sys.executable}")
    print()

    if venv_exists:
        print("✓ venv/ gefunden im Projekt-Ordner!")
        print()
        print("💡 Du hast vergessen die venv zu aktivieren.")
        print()
    else:
        print("💡 WICHTIG:")
        print("   Wenn du VS Code verwendest und die venv bereits aktiviert ist,")
        print("   aber diese Meldung trotzdem erscheint:")
        print()
        print("   → VS Code verwendet den FALSCHEN Python-Interpreter!")
        print()
        print("   Fix:")
        print("   1. Cmd + Shift + P")
        print("   2. Tippe: 'Python: Select Interpreter'")
        print("   3. Wähle: venv/bin/python")
        print()

    print("=" * 80)
    print("\nOptionen:")

    if venv_exists:
        # venv exists but not activated
        print("[1] venv aktivieren (EMPFOHLEN)")
        print("    → Zeigt Aktivierungs-Befehl")
        print("[2] Neu installieren")
        print("    → Löscht venv/ und installiert neu (nur bei Problemen)")
        print("[3] Dependencies prüfen/installieren")
        print("[4] Abbrechen")
    else:
        # venv does not exist
        print("[1] Automatisch installieren (EMPFOHLEN)")
        print("    → Erstellt venv/ und installiert alle Dependencies")
        print("[2] Manuelle Anweisungen anzeigen")
        print("[3] Dependencies prüfen/installieren")
        print("[4] Abbrechen")
    print()

    choice = input("Wahl [1/2/3/4]: ").strip()

    if choice == "1":
        if venv_exists:
            # Option [1] when venv exists: Show activation instructions
            print()
            print("=" * 80)
            print("📌 AKTIVIERE DIE VENV")
            print("=" * 80)
            print()
            print("Führe folgenden Befehl in deinem Terminal aus:\n")
            print("    source venv/bin/activate\n")
            print("Dein Prompt sollte dann (venv) anzeigen:\n")
            print("    (venv) user@computer Masterarbeit-Kennzahlenanalyse %\n")
            print("Danach starte das Script erneut:")
            print("    python src/main.py --market germany --compare\n")
            print("=" * 80)
            print()
            sys.exit(0)
        else:
            # Option [1] when venv doesn't exist: Automatic installation
            import subprocess

            print()
            print("=" * 80)
            print("🔧 AUTOMATISCHE INSTALLATION")
            print("=" * 80)

            # 1. Create venv
            print("\n1️⃣  Erstelle virtuelle Umgebung...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", "venv"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("    ✅ venv/ erstellt")
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Fehler beim Erstellen der venv:")
                print(f"    {e.stderr}")
                sys.exit(1)

            # 2. Upgrade pip
            print("\n2️⃣  Upgrade pip...")
            try:
                subprocess.run(
                    ["venv/bin/pip", "install", "--upgrade", "pip"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("    ✅ pip aktualisiert")
            except subprocess.CalledProcessError:
                print("    ⚠️  pip upgrade fehlgeschlagen (nicht kritisch)")

            # 3. Install requirements
            print("\n3️⃣  Installiere Dependencies aus requirements.txt...")
            requirements_path = Path("requirements.txt")
            if not requirements_path.exists():
                print("    ❌ requirements.txt nicht gefunden!")
                sys.exit(1)

            try:
                subprocess.run(
                    ["venv/bin/pip", "install", "-r", "requirements.txt"],
                    check=True
                )
                print("    ✅ Alle Dependencies installiert")
            except subprocess.CalledProcessError:
                print("    ❌ Fehler bei der Installation!")
                print("    Prüfe requirements.txt und versuche es manuell.")
                sys.exit(1)

            # Success message
            print("\n" + "=" * 80)
            print("✅ SETUP KOMPLETT!")
            print("=" * 80)
            print("\n📌 NÄCHSTER SCHRITT:")
            print("   Aktiviere die venv mit:\n")
            print("       source venv/bin/activate\n")
            print("   Dann führe das Script erneut aus:")
            print("       python src/main.py --market germany --compare\n")
            print("=" * 80)
            print()
            sys.exit(0)

    elif choice == "2":
        if venv_exists:
            # Option [2] when venv exists: Reinstall (delete and recreate)
            import subprocess
            import shutil

            print()
            print("=" * 80)
            print("🔄 NEU-INSTALLATION")
            print("=" * 80)
            print()
            print("⚠️  WARNUNG: Dies löscht die existierende venv/ komplett!")
            print()
            confirm = input("Fortfahren? [j/N]: ").strip().lower()

            if confirm not in ['j', 'ja', 'y', 'yes']:
                print("\n✓ Abgebrochen.\n")
                sys.exit(0)

            # 1. Delete old venv
            print("\n1️⃣  Lösche alte venv...")
            try:
                shutil.rmtree("venv")
                print("    ✅ venv/ gelöscht")
            except Exception as e:
                print(f"    ❌ Fehler beim Löschen: {e}")
                sys.exit(1)

            # 2. Create new venv
            print("\n2️⃣  Erstelle neue virtuelle Umgebung...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", "venv"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("    ✅ venv/ erstellt")
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Fehler beim Erstellen der venv:")
                print(f"    {e.stderr}")
                sys.exit(1)

            # 3. Upgrade pip
            print("\n3️⃣  Upgrade pip...")
            try:
                subprocess.run(
                    ["venv/bin/pip", "install", "--upgrade", "pip"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("    ✅ pip aktualisiert")
            except subprocess.CalledProcessError:
                print("    ⚠️  pip upgrade fehlgeschlagen (nicht kritisch)")

            # 4. Install requirements
            print("\n4️⃣  Installiere Dependencies aus requirements.txt...")
            requirements_path = Path("requirements.txt")
            if not requirements_path.exists():
                print("    ❌ requirements.txt nicht gefunden!")
                sys.exit(1)

            try:
                subprocess.run(
                    ["venv/bin/pip", "install", "-r", "requirements.txt"],
                    check=True
                )
                print("    ✅ Alle Dependencies installiert")
            except subprocess.CalledProcessError:
                print("    ❌ Fehler bei der Installation!")
                sys.exit(1)

            # Success message
            print("\n" + "=" * 80)
            print("✅ NEU-INSTALLATION KOMPLETT!")
            print("=" * 80)
            print("\n📌 NÄCHSTER SCHRITT:")
            print("   Aktiviere die venv mit:\n")
            print("       source venv/bin/activate\n")
            print("   Dann führe das Script erneut aus:")
            print("       python src/main.py --market germany --compare\n")
            print("=" * 80)
            print()
            sys.exit(0)
        else:
            # Option [2] when venv doesn't exist: Manual instructions
            print()
            print("=" * 80)
            print("📋 MANUELLE INSTALLATION")
            print("=" * 80)
            print()
            print("Führe folgende Befehle aus:\n")
            print("    python3 -m venv venv")
            print("    source venv/bin/activate")
            print("    pip install -r requirements.txt")
            print()
            print("Danach starte das Script erneut:")
            print("    python src/main.py --market germany --compare\n")
            print("=" * 80)
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
