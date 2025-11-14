# 📊 Interactive Jupyter Notebooks - Quick Start

## ✨ Was sind diese Notebooks?

Die Jupyter Notebooks ermöglichen **step-by-step interaktive Clustering-Analyse** mit **on-the-fly Visualisierungen**.

Im Gegensatz zur `main.py` (vollautomatischer Durchlauf) kannst du hier:
- ✅ Parameter live anpassen (k, Features, Algorithmus)
- ✅ Sofortige Visualisierungen sehen
- ✅ Cluster-Qualität interaktiv testen
- ✅ Zwischenergebnisse speichern

---

## 🚀 Setup

### 1. Jupyter installieren (falls noch nicht geschehen)
```bash
pip install jupyter ipywidgets notebook
```

### 2. Jupyter starten
```bash
cd /path/to/Masterarbeit-Kennzahlenanalyse
jupyter notebook
```

Der Browser öffnet sich automatisch auf `localhost:8888`

---

## 📚 Workflow

Die Notebooks sind nummeriert für schrittweise Analyse:

### **00_Setup_and_Config.ipynb** ⚙️
- Environment validieren
- Config laden
- Rohdaten laden
- **Dauer**: 2 min

### **01_Data_Preprocessing.ipynb** 🧹
- Fehlende Werte behandeln
- Feature-Engineering
- Korrelations-Analyse
- **Dauer**: 5 min

### **02_KMeans_Clustering.ipynb** 🎯
- **Interaktiv k wählen** (Slider!)
- Elbow-Methode, Silhouette
- PCA Visualisierung (2D/3D)
- Cluster-Profile
- **Dauer**: 10 min

### **03_Hierarchical_Clustering.ipynb** 🌳
- Dendrogram anschauen
- Hierarchie erkunden
- Mit K-Means vergleichen
- **Dauer**: 5 min

### **04_DBSCAN_Clustering.ipynb** 🔍
- eps/min_samples tunen
- Outlier finden
- Density-based Clustering
- **Dauer**: 5 min

### **05_Algorithm_Comparison.ipynb** ⚖️
- Alle Algorithmen vergleichen
- ARI-Matrix
- Best Algorithm wählen
- **Dauer**: 5 min

---

## 🔧 Beispiel: Interaktives K-Tuning

```python
# Cell 1: Setup
from notebook_utils import setup_notebook
cfg, state = setup_notebook("K-Means Test", "germany")

# Cell 2: Load data
df = state.load('df_preprocessed')

# Cell 3: Interactive K-Selection
from ipywidgets import interact, IntSlider

@interact(k=IntSlider(min=2, max=10, value=4, description='K:'))
def test_kmeans(k):
    """User kann k mit Slider ändern - Ergebnisse sofort sichtbar!"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    
    # Silhouette Score
    sil = silhouette_score(X_pca, labels)
    
    # Cluster sizes
    sizes = pd.Series(labels).value_counts()
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'K={k} | Silhouette={sil:.3f}')
    
    plt.subplot(122)
    sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.show()
```

**→ User bewegt Slider** → **Visualisierung updated sofort!** ✨

---

## 💡 Best Practices

### 1. **State Management nutzen**
```python
# Zwischenergebnisse speichern
state.save('df_clustered', df_result)
state.save('best_k', 4)

# In nächstem Notebook laden
df_clustered = state.load('df_clustered')
```

### 2. **Experimente dokumentieren**
```markdown
## Experiment 1: k=4 vs k=5
- k=4: Silhouette 0.35, bessere Separation
- k=5: Silhouette 0.32, Mini-Cluster (2 companies)
→ **Entscheidung: k=4**
```

### 3. **Plots exportieren**
```python
plt.savefig('output/custom_analysis/my_cluster_plot.png', dpi=300)
```

---

## 🆚 Notebooks vs. main.py

| Aspekt | main.py | Jupyter Notebooks |
|--------|---------|-------------------|
| **Use Case** | Vollautomatischer Durchlauf | Interaktive Exploration |
| **Output** | 7 Excel-Dateien | Live Visualizations |
| **Flexibilität** | Config-gesteuert | Parameter on-the-fly ändern |
| **Geschwindigkeit** | 30s | 30 min (hands-on) |
| **Ideal für** | Produktionsdaten | Methodenentwicklung, Debugging |

**Empfehlung**: 
- 🧪 **Entwicklung/Forschung**: Notebooks verwenden
- 🚀 **Finale Analysen**: main.py verwenden

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: notebook_utils"
```bash
# Stelle sicher dass du im Projekt-Root bist
cd /path/to/Masterarbeit-Kennzahlenanalyse
jupyter notebook
```

### Problem: "State file not found"
→ Starte mit `00_Setup_and_Config.ipynb` - es erstellt den State

### Problem: Kernel crashed
→ Starte Kernel neu: `Kernel > Restart & Clear Output`

---

## 📞 Support

Bei Fragen zu Notebooks siehe:
- `notebooks/README.md` - Detaillierte Dokumentation
- `notebooks/notebook_utils.py` - Source Code der Utilities

---

**Happy Clustering! 🎉**
