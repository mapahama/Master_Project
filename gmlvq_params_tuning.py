# .\venv311\Scripts\activate

#####################################
#
# GMLVQ Parameter Tuning via Kreuzvalidierung (5 Folds)
# Parameters:
# 1. Prototypen pro Klasse
# 2. Regularization-Parameter
#
# Dataset splits (Training / Testing):
# 1. 80/20  (voreingestellt)
#
#####################################

# === Bibliotheken importieren ===
# importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo
# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
# um Merkmale zu standardisieren (Mittelwert 0, Standardabweichung 1)
from sklearn.preprocessing import StandardScaler
# GmlvqModel: Die Klasse für das Generalized Matrix Learning Vector Quantization Modell
from sklearn_lvq import GmlvqModel

# Verschiedene Metriken aus sklearn.metrics zur Evaluation des Klassifikationsmodells:
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix 
)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# === === === === === ===  ===
# ===  1. Dataset laden    ===
# === === === === === ===  ===
print("Lade Heart Disease Datensatz...")
# Ruft den "Heart Disease" Datensatz (mit der ID 45) vom UCI Machine Learning Repository ab.
heart_disease = fetch_ucirepo(id=45)

# Extrahiert die Merkmalsdaten (Features) aus dem geladenen Objekt
X = heart_disease.data.features.copy()

# Extrahiert die Zielvariablen (Targets) aus dem geladenen Objekt.
y = heart_disease.data.targets.copy()
print("Datensatz geladen.")

# === === === === === === === === ===
# ===  2. Targets binär machen   ===
# === === === === === === === === ===
# (0 = gesund, 1-4 = verschiedene Stufen von krank).
if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
    y = y.iloc[:, 0]
elif isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] == 1:
    # Wandle es in einen 1D-Array um und dann in eine Pandas Series
    y = pd.Series(y.ravel())
elif not isinstance(y, pd.Series):
    # Wandle es sicherheitshalber in eine Pandas Series um.
    y = pd.Series(y.ravel())

# Erstelle die binäre Zielvariable `y_binary`:
# Alle Werte in `y`, die größer als 0 sind (also 1, 2, 3, 4), werden als `True` ausgewertet (krank).
# Werte gleich 0 werden als `False` ausgewertet (gesund)
y_binary = (y > 0).astype(int)
print("Verteilung der binären Zielvariable:")
# zählt die Häufigkeit jedes einzigartigen Wertes in der Series.
print(y_binary.value_counts())

# === === === === === === ===
# ===  3. Vorverarbeitung ===
# === === === === === === ===
# Merkmalsdaten (X) bereinigen und für das Modell vorbereiten
print("\nStarte Vorverarbeitung...")
# Ersetze fehlende Werte, die im Datensatz als '?' (Fragezeichen) kodiert sind, durch `np.nan`.
X.replace('?', np.nan, inplace=True)

# Versuche alle Spalten des DataFrames `X` in numerische Werte umzuwandeln.
# falls noch andere nicht-numerische Zeichenketten (außer '?') in den Daten vorhanden waren.
X = X.apply(pd.to_numeric, errors='coerce')

# Fülle alle verbleibenden `np.nan`-Werte (fehlende Werte) mit dem Median der jeweiligen Spalte
X.fillna(X.median(), inplace=True)

### HIER FEATURE-NAMEN SPEICHERN ###
# Speichere die Spaltennamen, bevor die Daten in ein Numpy-Array umgewandelt werden
feature_names = X.columns.tolist()


# Dieser Scaler standardisiert Merkmale, indem er den Mittelwert jeder Spalte abzieht
# und dann durch die Standardabweichung teilt. Ergebnis: Mittelwert 0, Standardabweichung 1.
# Ziel: alle Merkmale  auf eine ähnliche Skala (Bereich) zu bringen!
scaler = StandardScaler()

# berechne Mittelwert und Standardabweichung für jede Spalte
X_scaled = scaler.fit_transform(X)
# Wichtig: Konvertiere y_binary zu einem numpy array für die Kreuzvalidierung
y_binary_np = y_binary.to_numpy() 
print("Vorverarbeitung abgeschlossen.")


# === === === === === === === === === === === === === === === === === ===
# === Aufteilen in Trainings- und Test-Set  // 80/20          
# === === === === === === === === === === === === === === === === === ===
# Die Kreuzvalidierung wird nur auf dem Trainingsset durchgeführt!
# Das Testset wird für eine spätere, finale Evaluation zurückgehalten.
X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
    X_scaled,
    y_binary_np,
    test_size=0.20,      # Trainings-Set ist 80%, Test-Set ist 20%
    random_state=42,     # Für reproduzierbare Ergebnisse 
    stratify=y_binary_np # Stellt sicher, dass die Klassenverteilung in beiden Sets gleich ist
)

print(f"\nDaten aufgeteilt: {len(X_train_cv)} Proben für Kreuzvalidierung und {len(X_test_holdout)} für das finale Testen.")


# === === === === === === === === === === === === === === === === === === === === ===
# ===  4. Hyperparameter-Tuning mit GridSearchCV für GMLVQ                       ===
# === === === === === === === === === === === === === === === === === === === === ===
print("\n--- STARTE GridSearchCV: Tuning von 'prototypes_per_class' und 'regularization' ---")

# 1. Definiere das Gitter der zu testenden Parameter
param_grid = {
    'prototypes_per_class': [1, 2, 3, 4, 5],
    'regularization': np.arange(0.0, 0.5, 0.05).tolist() # [0.0, 0.05, 0.1, ..., 0.45]
}

# 2. Definiere die Kreuzvalidierungs-Strategie
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 3. Initialisiere GridSearchCV
# Es testet automatisch alle Parameter-Kombinationen mit 5-facher Kreuzvalidierung
# n_jobs=-1 nutzt alle verfügbaren CPU-Kerne, um die Suche zu beschleunigen
grid_search = GridSearchCV(
    estimator=GmlvqModel(random_state=42),
    param_grid=param_grid,
    scoring='accuracy', # Optimiere für die beste Genauigkeit
    cv=skf,
    verbose=1, # Zeigt den Fortschritt an
    n_jobs=-1
)

# 4. Starte die Suche auf dem 80%-Trainingsdatensatz
print("GridSearchCV trainiert nun...")
grid_search.fit(X_train_cv, y_train_cv)
print("GridSearchCV abgeschlossen.")


# === === === === === === === === ===
# ===  5. FINALES TUNING-ERGEBNIS ===
# === === === === === === === === ===
print("\n\n==================================================")
print("===           FINALES TUNING-ERGEBNIS           ===")
print("===================================================")

# Gib die besten gefundenen Parameter und die dazugehörige CV-Genauigkeit aus
print(f"\nDie beste gefundene Hyperparameter-Kombination ist:")
print(f"  -> {grid_search.best_params_}")
print(f"\nBeste Kreuzvalidierungs-Genauigkeit (Accuracy) auf den Trainingsdaten: {grid_search.best_score_:.4f}")


# === === === === === === === === === === === === === === === === ===
# ===  6. FINALE EVALUATION AUF UNGESEHENEN TESTDATEN (20 %)    ===
# === === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===  FINALE EVALUATION AUF DEM UNBERÜHRTEN TEST-SET  ===")
print("================================================================")

# Wichtig: GridSearchCV trainiert automatisch ein finales Modell mit den besten Parametern
# auf dem gesamten zur Verfügung gestellten Datensatz (X_train_cv).
# Wir können dieses beste Modell direkt für die Vorhersage verwenden.
print("\nDas beste Modell aus GridSearchCV wird für die Vorhersage auf dem Test-Set verwendet.")

# 1. Mache Vorhersagen auf dem zurückgehaltenen Test-Set (Holdout-Set)
y_pred_final = grid_search.predict(X_test_holdout)

# 2. Berechne die finalen Leistungsmetriken
final_accuracy = accuracy_score(y_test_holdout, y_pred_final)
final_precision = precision_score(y_test_holdout, y_pred_final, zero_division=0)
final_recall = recall_score(y_test_holdout, y_pred_final, zero_division=0)

# 3. Gib die finalen Ergebnisse aus
print("\nLeistung des finalen Modells auf den ungesehenen Testdaten:")
print(f"  -> Finale Accuracy:   {final_accuracy:.4f}")
print(f"  -> Finale Precision:  {final_precision:.4f}")
print(f"  -> Finale Recall:     {final_recall:.4f}")


# === === === === === === === === === === === === === === === ===
# ===  7. KONFUSIONSMATRIX FÜR DAS FINALE TEST-SET            ===
# === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===           KONFUSIONSMATRIX FÜR DAS TEST-SET             ===")
print("================================================================")

# 1. Berechne die Konfusionsmatrix
cm = confusion_matrix(y_test_holdout, y_pred_final)

# 2. Erstelle eine Visualisierung mit Seaborn
plt.figure(figsize=(8, 6)) # Definiert die Größe des Diagramms
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)

# 3. Füge Beschriftungen hinzu für bessere Lesbarkeit
plt.title('Konfusionsmatrix für das Test-Set', fontsize=16)
plt.xlabel('Vorhergesagte Klasse', fontsize=12)
plt.ylabel('Tatsächliche Klasse', fontsize=12)
# Setze die Achsenbeschriftungen auf die Klassennamen
class_names = ['Gesund (0)', 'Krank (1)']
plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names)
plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)

# 4. Zeige das Diagramm an / in einem Fenster
print("\nZeige Konfusionsmatrix an...")
plt.show()

'''
# === === === === === === === === === === === === === === === ===
# ===  8. VISUALISIERUNG DER GMLVQ RELEVANZ-MATRIX (OMEGA)    ===
# === === === === === === === === === === === === === === === ===
print("\n\n================================================================")
print("===    GMLVQ RELEVANZ-MATRIX (OMEGA) VISUALISIERUNG     ===")
print("================================================================")

# 1. Extrahiere die OMEGA Relevanz-Matrix aus dem besten Modell
# Das korrekte Attribut heißt .omega_    // laut sklearn_lvq Dokumentation
omega_matrix = grid_search.best_estimator_.omega_

# 2. Erstelle die Heatmap
# Eine größere Figure-Size ist für die Lesbarkeit der Achsenbeschriftungen hilfreich
plt.figure(figsize=(12, 10))
sns.heatmap(omega_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=False,  # Auf False gesetzt, da die Matrix sonst zu unübersichtlich wird
            fmt=".2f",
            cmap='viridis') # 'viridis' ist  gute Farbpalette für Relevanz

# 3. Füge Titel und Beschriftungen hinzu und optimiere die Darstellung
plt.title('GMLVQ Relevanz-Matrix (Omega Ω)', fontsize=16)
plt.xticks(rotation=45, ha="right") # Rotiert die x-Achsen-Beschriftung für bessere Lesbarkeit
plt.yticks(rotation=0)
plt.tight_layout() # Passt das Layout an, um Überschneidungen zu vermeiden

# 4. Zeige das Diagramm an
print("\nZeige Relevanz-Matrix an...")
print("Helle Felder auf der Diagonalen zeigen die wichtigsten Merkmale an.")
plt.show()
'''

# === === === === === === === === === === === === === === === ===
# ===  9. VISUALISIERUNG DER GMLVQ RELEVANZ-MATRIX (LAMBDA)    ===
# === === === === === === === === === === === === === === === ===
print("\n\n================================================================")
print("===    GMLVQ RELEVANZ-MATRIX (LAMBDA) VISUALISIERUNG     ===")
print("================================================================")

# Extrahiere omega_ aus dem trainierten Modell
omega = grid_search.best_estimator_.omega_

# Berechne daraus die Lambda-Matrix (symmetrisch, positive semi-definit)
lambda_matrix = np.dot(omega.T, omega)

# Spaltennamen (Feature-Namen) verwenden
xticklabels = feature_names
yticklabels = feature_names

# Visualisierung
plt.figure(figsize=(12, 10))
sns.heatmap(lambda_matrix,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            annot=False,
            fmt=".2f",
            cmap='viridis')

plt.title("GMLVQ Relevanzmatrix (LAMBDA  Λ = Ωᵀ·Ω)", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()