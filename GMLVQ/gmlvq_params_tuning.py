
# .\venv311\Scripts\activate

#####################################
#
# GMLVQ Parameter Tuning via Kreuzvalidierung (5 Folds)
# Parameters:
# 1. Prototypen pro Klasse
# 2. Regularization-Parameter
# 3. Gradient Toleranz (gtol)
#
# Dataset splits (Training / Testing):
# 1. 80/20
# 2. 90/10
#
#####################################

# === Bibliotheken importieren ===
# importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo
# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# GmlvqModel: Die Klasse für das Generalized Matrix Learning Vector Quantization Modell
from sklearn_lvq import GmlvqModel
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

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
# ===  2. Targets binär machen    ===
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
print("Basis-Vorverarbeitung abgeschlossen.")

# Speichere die Spaltennamen, bevor die Daten in ein Numpy-Array umgewandelt werden
feature_names = X.columns.tolist()

#---------------------
# AUSREISSERERKENNUNG 
#---------------------

# --- AUSREISSERERKENNUNG Schritt 1: Visuelle Ausreißererkennung mit Boxplots (Univariat) ---
# Die Boxplots zeigen uns, OB es Ausreißer gibt
print("\n--- Erstelle Boxplots für jedes Merkmal ---")
plt.figure(figsize=(20, 15))
for i, col in enumerate(X.columns):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=X[col], color='skyblue')
    plt.title(col)
    plt.ylabel('')
plt.suptitle('Boxplots für jedes Merkmal zur univariaten Ausreißererkennung', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- AUSREISSERERKENNUNG Schritt 2: Ausreißererkennung mit Isolation Forest (Multivariat) ---
# Der Isolation Forest entscheidet WELCHE  Ausreißer entfernt werden sollten
print("\n--- Wende Isolation Forest an ---")
# HINWEIS: Die Daten werden hier nur für den Zweck der Ausreißererkennung skaliert !!
scaler_for_iso = MinMaxScaler(feature_range=(-1, 1))
X_scaled_for_iso = scaler_for_iso.fit_transform(X)

# Isolation Forest initialisieren und anwenden
# contamination legt den erwarteten Anteil der Ausreißer fest
iso_forest = IsolationForest(contamination=0.08, random_state=42) # 8%
predictions = iso_forest.fit_predict(X_scaled_for_iso) # -1 für Ausreißer, 1 für Inlier

# Indizes der als Ausreißer markierten Datenpunkte
outlier_indices = np.where(predictions == -1)[0]
print(f"Anzahl der erkannten Ausreißer: {len(outlier_indices)}")

# --- AUSREISSERERKENNUNG Schritt 3: Visualisierung der erkannten Ausreißer via PCA ---
print("\n---Schritt 3: Visualisiere Ausreißer in 2D ---")
# Reduziere die Dimensionen auf 2D für die Visualisierung
pca = PCA(n_components=2)
# HINWEIS: PCA wird auf denselben skalierten Daten wie der Isolation Forest ausgeführt
X_pca = pca.fit_transform(X_scaled_for_iso)

plt.figure(figsize=(12, 8))
# Zeichne die regulären Datenpunkte ("Inlier")
plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1], c='dodgerblue', alpha=0.7, label='Inlier')
# Zeichne die erkannten Ausreißer
plt.scatter(X_pca[outlier_indices, 0], X_pca[outlier_indices, 1], c='red', edgecolor='k', s=90, label='Ausreißer')

plt.title('Erkannte Ausreißer durch Isolation Forest (visualisiert mit PCA)')
plt.xlabel('Erste Hauptkomponente')
plt.ylabel('Zweite Hauptkomponente')
plt.legend()
plt.grid(True)
plt.show()


# --- AUSREISSERERKENNUNG Schritt 4: Entfernen der Ausreißer ---
print("\n--- Entferne Ausreißer aus dem Datensatz ---")
print(f"Originale Datenform (X): {X.shape}")
# y_binary wird hier verwendet, da es die korrekte binäre Zielvariable ist
print(f"Originale Datenform (y): {y_binary.shape}")

# Entferne die Ausreißer aus X und y_binary
X_clean = X.drop(X.index[outlier_indices])
y_clean = y_binary.drop(y_binary.index[outlier_indices])

print(f"Neue Datenform nach Außreiser-Bereinigung (X_clean): {X_clean.shape}")
print(f"Neue Datenform nach Außreiser-Bereinigung (y_clean): {y_clean.shape}")


# === === === === === === === === === === === === === === === === === ===
# === 4. Aufteilen und Skalieren nach der Ausreißererkennung
# === === === === === === === === === === === === === === === === === ===

# Die Kreuzvalidierung wird nur auf dem Trainingsset durchgeführt!
# Das Testset wird für eine spätere, finale Evaluation zurückgehalten.

# Schritt A: Aufteilung der BEREINIGTEN Daten (X_clean, y_clean)
X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
    X_clean,
    y_clean,
    test_size=0.10,      # Trainings-Set ist 80%, Test-Set ist 20%
    random_state=42,     # Für reproduzierbare Ergebnisse
    stratify=y_clean     # Stellt sicher, dass die Klassenverteilung in beiden Sets gleich ist
)

print(f"\nDaten aufgeteilt: {len(X_train_cv)} Proben für Kreuzvalidierung und {len(X_test_holdout)} für das finale Testen.")

# Schritt B: Skalierung der aufgeteilten Daten im Bereich [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_cv = scaler.fit_transform(X_train_cv)
X_test_holdout = scaler.transform(X_test_holdout) 
print("Skalierung der Trainings- und Testsets abgeschlossen.")


# === === === === === === === === === === === === === === === === === === === === ===
# ===  5. Hyperparameter-Tuning mit GridSearchCV für GMLVQ                       ===
# === === === === === === === === === === === === === === === === === === === === ===
print("\n--- STARTE GridSearchCV: Tuning von 'prototypes_per_class' , 'regularization' , 'gtol' ---")

# 1. Definiere das Gitter der zu testenden Parameter
param_grid = {
    'prototypes_per_class': [1, 2, 3, 4, 5],
    'regularization': np.arange(0.0, 0.5, 0.05).tolist(), # Hilft Overfitting der Relevanzmatrix zu verhindern
    'gtol': [1e-4, 1e-5, 1e-6]  # Das Training stoppt, wenn die Norm des Gradienten unter diesen Wert fällt.
                                 # Ein kleinerer Wert führt zu einer längeren, aber potenziell genaueren Konvergenz.
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



# FINALES TUNING-ERGEBNIS 
print("\n\n==================================================")
print("===           FINALES TUNING-ERGEBNIS            ===")
print("===================================================")

# Gib die besten gefundenen Parameter und die dazugehörige CV-Genauigkeit aus
print(f"\nDie beste gefundene Hyperparameter-Kombination ist:")
print(f"  -> {grid_search.best_params_}")
print(f"\nBeste Kreuzvalidierungs-Genauigkeit (Accuracy) auf den Trainingsdaten: {grid_search.best_score_:.4f}")


# === === === === === === === === === === === === === === === === ===
# ===  6. FINALE EVALUATION AUF UNGESEHENEN TESTDATEN (20 %)     ===
# === === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===   FINALE EVALUATION AUF DEM UNBERÜHRTEN TEST-SET   ===")
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
# ===  7. KONFUSIONSMATRIX FÜR DAS FINALE TEST-SET        ===
# === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===        KONFUSIONSMATRIX FÜR DAS TEST-SET         ===")
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


# === === === === === === === === === === === === === === === ===
# ===  8. VISUALISIERUNG DER GMLVQ RELEVANZ-MATRIX (OMEGA)    ===
# === === === === === === === === === === === === === === === ===
print("\n\n================================================================")
print("===    GMLVQ RELEVANZ-MATRIX (OMEGA) VISUALISIERUNG    ===")
print("================================================================")

# 1. Extrahiere die OMEGA Relevanz-Matrix aus dem besten Modell
# Das korrekte Attribut heißt .omega_      // laut sklearn_lvq Dokumentation
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


# === === === === === === === === === === === === === === === ===
# ===  9. VISUALISIERUNG DER GMLVQ RELEVANZ-MATRIX (LAMBDA)     ===
# === === === === === === === === === === === === === === === ===
print("\n\n================================================================")
print("===    GMLVQ RELEVANZ-MATRIX (LAMBDA) VISUALISIERUNG    ===")
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

##########################
# Ergebnisse:
###########################
#
# GridSearchCV Tuning von 'prototypes_per_class' , 'regularization' , 'gtol' 
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
#
#------------------------
# ( 80/20 Dataset-Split )
#------------------------
# Die beste gefundene Hyperparameter-Kombination  ( 80/20 Dataset-Split ) ist:
#  -> {'gtol': 1e-06, 'prototypes_per_class': 3, 'regularization': 0.05}
#
# Leistung des finalen Modells auf den ungesehenen Testdaten (80/20 Dataset-Split):
#  -> Finale Accuracy:   0.8571
#  -> Finale Precision:  0.8400
#  -> Finale Recall:     0.8400
#
#
#------------------------
# ( 90/10 Dataset-Split )
#------------------------
# Die beste gefundene Hyperparameter-Kombination ist:
#  -> {'gtol': 1e-05, 'prototypes_per_class': 2, 'regularization': 0.4}
#
# Leistung des finalen Modells auf den ungesehenen Testdaten:
#  -> Finale Accuracy:   0.8929
#  -> Finale Precision:  0.9091
#  -> Finale Recall:     0.8333