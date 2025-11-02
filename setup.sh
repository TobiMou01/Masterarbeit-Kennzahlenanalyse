#!/bin/bash
# Automatisches Setup-Script für Financial Clustering Analysis Pipeline
# Erstellt venv und installiert alle Dependencies

set -e  # Exit on error

echo ""
echo "=================================================================================="
echo "🔧 FINANCIAL CLUSTERING ANALYSIS - AUTOMATISCHES SETUP"
echo "=================================================================================="
echo ""

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 nicht gefunden!"
    echo "   Bitte installiere Python 3.8 oder höher."
    exit 1
fi

echo "✓ Python gefunden: $(python3 --version)"
echo ""

# 1. Create venv
echo "1️⃣  Erstelle virtuelle Umgebung..."
if [ -d "venv" ]; then
    echo "   ⚠️  venv/ existiert bereits - überspringe Erstellung"
else
    python3 -m venv venv
    echo "   ✅ venv/ erstellt"
fi
echo ""

# 2. Activate venv
echo "2️⃣  Aktiviere venv..."
source venv/bin/activate
echo "   ✅ venv aktiviert"
echo ""

# 3. Upgrade pip
echo "3️⃣  Upgrade pip..."
pip install --upgrade pip --quiet
echo "   ✅ pip aktualisiert"
echo ""

# 4. Install requirements
echo "4️⃣  Installiere Dependencies aus requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "   ❌ requirements.txt nicht gefunden!"
    exit 1
fi

pip install -r requirements.txt
echo "   ✅ Alle Dependencies installiert"
echo ""

# Success message
echo "=================================================================================="
echo "✅ SETUP KOMPLETT!"
echo "=================================================================================="
echo ""
echo "📌 Die virtuelle Umgebung ist JETZT AKTIV in diesem Terminal."
echo ""
echo "Du kannst jetzt die Pipeline starten:"
echo ""
echo "    python src/main.py --market germany --compare"
echo ""
echo "💡 WICHTIG für NEUE TERMINALS:"
echo "   Wenn du ein neues Terminal öffnest, aktiviere die venv mit:"
echo ""
echo "    source venv/bin/activate"
echo ""
echo "=================================================================================="
echo ""
