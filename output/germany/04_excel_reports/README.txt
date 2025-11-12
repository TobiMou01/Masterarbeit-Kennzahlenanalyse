EXCEL REPORTS - MASTERARBEIT KENNZAHLENANALYSE
==============================================

Generiert: 2024-11-12
Markt: Germany


📁 DATEI-ÜBERSICHT
==================

1. research_analysis_master.xlsx (1.2 MB, 14 Sheets)
   ─────────────────────────────────────────────────
   HAUPTFILE FÜR MASTERARBEIT - Strukturiert nach 4 Kernfragen

   Zielgruppe: Prüfer, Thesis-Leser
   Inhalt:
   - Section 0: Config & Overview (1 Sheet)
   - Section 1: Homogenität - Interne Cluster-Qualität (4 Sheets)
   - Section 2: Kongruenz - Vergleich mit Klassifikationen (3 Sheets)
   - Section 3: Treiber - Feature Importance & PCA (3 Sheets)
   - Section 4: Stabilität - Zeitliche & kontextuelle Analysen (3 Sheets)

   Empfehlung: Starte hier für Gesamtüberblick!


2. algorithm_comparison_combined.xlsx (65 KB, 5 Sheets)
   ──────────────────────────────────────────────────
   ALGORITHMEN-VERGLEICH (K-Means vs. Hierarchical vs. DBSCAN)

   Zielgruppe: Methodischer Vergleich
   Inhalt:
   - Overview: Metriken-Vergleich (Silhouette, Calinski-Harabasz, etc.)
   - Summary: Konvergenz & Robustheit
   - Cluster Analysis: Cluster-Größen & -Verteilungen
   - Score Distributions: Score-Statistiken pro Algorithmus
   - Score Comparisons: Score-Korrelationen zwischen Algorithmen


3. company_cluster_analysis/ (5 Files)
   ────────────────────────────────────
   DETAIL-ANALYSEN PRO ALGORITHMUS & FEATURE-VARIANTE

   a) kmeans_combined.xlsx (150 KB, 9 Sheets) ⭐ EMPFOHLEN
      - Beste Balance: Static + Dynamic Features
      - 6 Cluster (0-5)
      - Sheets: Overview, 6x Cluster-Details, Score Evolution, Summary

   b) kmeans_static.xlsx (221 KB, 7 Sheets)
      - Nur Momentaufnahme-Kennzahlen
      - 5 Cluster (0-4)
      - Sheets: Overview, 5x Cluster-Details, Summary

   c) kmeans_dynamic.xlsx (32 KB, 7 Sheets)
      - Nur Wachstumsraten (CAGR)
      - 5 Cluster (0-4)
      - Sheets: Overview, 5x Cluster-Details, Summary

   d) hierarchical.xlsx (192 KB, 7 Sheets)
      - Hierarchisches Clustering (Validierung)
      - 5 Cluster (0-4)
      - Sheets: Overview, 5x Cluster-Details, Summary

   e) dbscan.xlsx (184 KB, 4 Sheets)
      - DBSCAN (Validierung)
      - 2 Cluster (0-1) + Outliers
      - Sheets: Overview, 2x Cluster-Details, Summary


📊 VERWENDUNG
=============

Für Masterarbeit-Kapitel:
-------------------------
- Kapitel 4.3 (Clusterbildung): research_analysis_master.xlsx, Section 1
- Kapitel 4.4 (Vergleich & Bewertung): research_analysis_master.xlsx, Section 2
- Kapitel 4.5 (Treiberanalyse): research_analysis_master.xlsx, Section 3
- Kapitel 4.6 (Stabilität): research_analysis_master.xlsx, Section 4

Für Detail-Analysen:
--------------------
- Cluster-Profile Deep-Dive: company_cluster_analysis/kmeans_combined.xlsx
- Methodischer Vergleich: algorithm_comparison_combined.xlsx
- Feature-Varianten testen: Vergleiche kmeans_static vs. kmeans_dynamic vs. kmeans_combined


🔍 SHEET-STRUKTUREN
===================

research_analysis_master.xlsx:
-------------------------------
1. 0_Config - Konfiguration & Übersicht
2. 1.1_Cluster_Quality - Silhouette, Homogenität
3. 1.2_Score_Distributions - Score-Verteilungen
4. 1.3_Cluster_Profiles - Cluster-Profile & Naming
5. 1.4_Score_Correlations - Score-Korrelationen
6. 2.1_GICS_Congruence - Vergleich mit Branchen (Cramér's V)
7. 2.2_Size_Congruence - Vergleich mit Größenklassen
8. 2.3_Algorithm_Agreement - ARI zwischen Algorithmen
9. 3.1_Feature_Importance - Feature Importance (Random Forest)
10. 3.2_PCA_Analysis - Hauptkomponentenanalyse
11. 3.3_Driver_Interpretation - Treiber-Interpretation
12. 4.1_Temporal_Stability - Migration Matrices
13. 4.2_Cluster_Migrations - Cluster-Wechsel
14. 4.3_Context_Analysis - Größen- & Länder-Effekte

company_cluster_analysis/*.xlsx:
--------------------------------
1. Overview - Alle Unternehmen mit Cluster-Zuweisungen & Scores
2. Cluster_0..N - Pro Cluster: Top/Bottom Performers, Feature-Statistiken
3. Score_Evolution - Score-Entwicklung über Zeit (nur in kmeans_combined)
4. Summary - Zusammenfassung: Cluster-Größen, Metriken


⚠️ BEKANNTE ISSUES
==================

1. Excel-Visualisierungen unvollständig
   - Aktuell: Nur 4 Bilder in research_analysis_master.xlsx
   - Soll: ~30-40 Bilder
   - Status: Wird in Phase 2 behoben

2. Feature-Problem beim Clustering
   - Symptom: Scores = 0, generische Cluster-Namen
   - Root Cause: Features fehlen im DataFrame beim Clustering
   - Status: Separate Analyse läuft

3. Veraltete Daten?
   - Generierungsdatum: 12.11.2024
   - Bei Änderungen an config.yaml: Pipeline neu laufen lassen!


📝 CHANGELOG
============

2024-11-12:
- Output-Cleanup: Excel-Files organisiert in 04_excel_reports/
- Veraltetes company_cluster_analysis.xlsx (18KB) gelöscht
- Neue Struktur: Master-Files + Detail-Analysen in Unterordner


💡 TIPPS
========

1. Öffne research_analysis_master.xlsx zuerst → Gesamtüberblick
2. Für Cluster-Details: company_cluster_analysis/kmeans_combined.xlsx
3. Für Algorithmen-Robustheit: algorithm_comparison_combined.xlsx
4. Alle Sheets haben farbcodierte Headers (Blau = Wichtig, Orange = Config)
5. Zahlen-Formatierung: Score-Spalten (0.00), Prozente (%), Währungen (€)
