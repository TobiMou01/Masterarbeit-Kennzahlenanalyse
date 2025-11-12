# Jupyter Notebook Setup mit Virtual Environment

## 🐍 Wie Jupyter mit der venv funktioniert

### Das Wichtigste zuerst:

**Jupyter Notebook nutzt automatisch die Python-Umgebung, aus der es gestartet wurde!**

Das heißt:
- Wenn du Jupyter **aus der aktivierten venv** startest, nutzt es die venv
- Alle Packages aus der venv sind verfügbar
- Kein extra Setup nötig! ✨

---

## 🚀 Setup-Schritte

### 1. Virtual Environment erstellen & aktivieren

```bash
# Ins Projekt-Verzeichnis wechseln
cd Masterarbeit-Kennzahlenanalyse

# Venv erstellen (falls noch nicht geschehen)
python -m venv venv

# Venv aktivieren
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

Nach Aktivierung siehst du `(venv)` vor deinem Prompt:
```bash
(venv) user@computer:~/Masterarbeit-Kennzahlenanalyse$
```

### 2. Requirements installieren (inkl. Jupyter)

```bash
# Alle Packages installieren (jetzt inkl. Jupyter!)
pip install -r requirements.txt
```

Das installiert automatisch:
- ✅ jupyter
- ✅ notebook
- ✅ ipywidgets
- ✅ alle anderen Dependencies (pandas, sklearn, etc.)

### 3. Jupyter Notebook starten

```bash
# WICHTIG: Stelle sicher, dass die venv noch aktiv ist (siehst du an (venv))
cd notebooks/
jupyter notebook
```

**Das war's!** Jupyter öffnet sich im Browser und nutzt automatisch deine venv.

---

## ✅ Überprüfen, ob es funktioniert

### Im Notebook:

Führe in der ersten Zelle aus:

```python
import sys
print(sys.executable)
```

**Erwartete Ausgabe:**
```
/home/user/Masterarbeit-Kennzahlenanalyse/venv/bin/python
```

Wenn der Pfad deine venv enthält → ✅ Alles richtig!

### Package-Test:

```python
import pandas as pd
import sklearn
import seaborn as sns

print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print("✅ Alle Packages verfügbar!")
```

---

## 🔧 Alternativer Weg: IPython Kernel registrieren

Falls Jupyter mal die falsche Python-Version nutzt (sehr selten), kannst du einen expliziten Kernel registrieren:

```bash
# In aktivierter venv:
pip install ipykernel
python -m ipykernel install --user --name=masterarbeit --display-name="Masterarbeit (venv)"
```

Dann in Jupyter Notebook:
- `Kernel` → `Change Kernel` → `Masterarbeit (venv)` auswählen

**Aber das ist normalerweise NICHT nötig!** Wenn du Jupyter aus der aktivierten venv startest, funktioniert es automatisch.

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pandas'"

**Ursache:** Jupyter nutzt nicht die venv

**Lösung:**
1. Jupyter beenden (Ctrl+C im Terminal)
2. Venv deaktivieren: `deactivate`
3. Venv neu aktivieren: `source venv/bin/activate` (Linux/Mac)
4. Jupyter neu starten: `jupyter notebook`

### Problem: Jupyter startet nicht

**Lösung:**
```bash
# Jupyter nochmal installieren
pip install --upgrade jupyter notebook

# Oder komplett neu:
pip uninstall jupyter notebook
pip install jupyter notebook
```

### Problem: Venv vergessen zu aktivieren

**Symptom:** Packages fehlen, oder Jupyter nutzt System-Python

**Lösung:**
```bash
# Schaue, ob (venv) vor dem Prompt steht
# Wenn nicht:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

---

## 📝 Best Practices

### Beim Arbeiten:

1. **Terminal 1** (Jupyter-Server):
   ```bash
   source venv/bin/activate
   cd notebooks/
   jupyter notebook
   # Läuft weiter...
   ```

2. **Browser**: Notebooks bearbeiten

3. **Terminal 2** (optional für Git, Tests, etc.):
   ```bash
   source venv/bin/activate
   git status
   python src/main.py
   ```

### Nach der Arbeit:

```bash
# Im Jupyter-Terminal: Ctrl+C (beendet Jupyter)
# Dann:
deactivate  # Deaktiviert venv
```

---

## 🎯 Zusammenfassung

```bash
# 1. Venv aktivieren
source venv/bin/activate

# 2. (Einmalig) Packages installieren
pip install -r requirements.txt

# 3. Jupyter starten
cd notebooks/
jupyter notebook

# 4. Im Browser: Notebooks nutzen!
```

**Das war's!** Jupyter nutzt automatisch die venv, keine Magie nötig. ✨

---

## 💡 Pro-Tipp

Wenn du immer vergisst, die venv zu aktivieren, erstelle ein Startscript:

**start_jupyter.sh** (Linux/Mac):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
cd notebooks/
jupyter notebook
```

**start_jupyter.bat** (Windows):
```batch
@echo off
call venv\Scripts\activate
cd notebooks
jupyter notebook
```

Dann einfach das Script ausführen → Jupyter startet mit venv!

---

**Happy Jupyter! 📓**
