# .\venv311\Scripts\activate

# === Bibliotheken importieren ===
# importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo
# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import StratifiedKFold
# um Merkmale zu standardisieren (Mittelwert 0, Standardabweichung 1)
from sklearn.preprocessing import StandardScaler
# GlvqModel: Die Klasse für das Generalized Learning Vector Quantization Modell
from sklearn_lvq import GlvqModel

# Verschiedene Metriken aus sklearn.metrics zur Evaluation des Klassifikationsmodells:
from sklearn.metrics import (
    precision_score,           
    recall_score,              
    accuracy_score            
)


import pandas as pd
import numpy as np

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

# Dieser Scaler standardisiert Merkmale, indem er den Mittelwert jeder Spalte abzieht
# und dann durch die Standardabweichung teilt. Ergebnis: Mittelwert 0, Standardabweichung 1.
# Ziel: alle Merkmale  auf eine ähnliche Skala (Bereich) zu bringen!
scaler = StandardScaler()

# berechne Mittelwert und Standardabweichung für jede Spalte
X_scaled = scaler.fit_transform(X)
# Wichtig: Konvertiere y_binary zu einem numpy array für die Kreuzvalidierung
y_binary_np = y_binary.to_numpy()
print("Vorverarbeitung abgeschlossen.")


# === === === === === === === === === === === === === === === ===
# ===  4. Stufe 1: Tuning der Anzahl der Prototypen (1-5)     ===
# === === === === === === === === === === === === === === === ===
print("\n--- STUFE 1: Starte 5-fache Kreuzvalidierung zum Tunen der PROTOTYPEN-ANZAHL ---")

# Definiere die Anzahl der Folds (Teile) für die Kreuzvalidierung
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Definiere den Bereich der zu testenden Prototypen-Anzahlen
prototypes_to_test = [1, 2, 3, 4, 5]

# Speicher für die Ergebnisse der Kreuzvalidierung
proto_cv_results = {}

# Äußere Schleife: Iteriert über jede zu testende Prototypen-Anzahl
for n_prototypes in prototypes_to_test:
    print(f"\nTeste mit {n_prototypes} Prototypen pro Klasse...")
    
    # Listen um die Metriken für jeden Fold zu speichern
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    
    # Innere Schleife: Führt die 5-fache Kreuzvalidierung auf dem GESAMTEN Datensatz durch
    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_binary_np)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_binary_np[train_index], y_binary_np[test_index]
        
        model_cv = GlvqModel(prototypes_per_class=n_prototypes, random_state=42)
        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)
        
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))

    # Berechne den Durchschnitt der Metriken über alle Folds
    proto_cv_results[n_prototypes] = {
        'mean_accuracy': np.mean(fold_accuracies),
        'mean_precision': np.mean(fold_precisions),
        'mean_recall': np.mean(fold_recalls)
    }
    
    print(f"  -> Avg. Accuracy: {proto_cv_results[n_prototypes]['mean_accuracy']:.4f}")
    print(f"  -> Avg. Precision: {proto_cv_results[n_prototypes]['mean_precision']:.4f}")
    print(f"  -> Avg. Recall: {proto_cv_results[n_prototypes]['mean_recall']:.4f}")

# Finde die beste Anzahl an Prototypen basierend auf der höchsten mittleren Genauigkeit
best_n_prototypes = max(proto_cv_results, key=lambda k: proto_cv_results[k]['mean_accuracy'])

print("\n--- Stufe 1 abgeschlossen ---")
print(f"Beste Anzahl an Prototypen pro Klasse: {best_n_prototypes} (basierend auf der höchsten mittleren Genauigkeit)")


# === === === === === === === === === === === === === ===
# ===  5. Stufe 2: Tuning des Beta-Parameters         ===
# === === === === === === === === === === === === === ===
print(f"\n--- STUFE 2: Starte 5-fache Kreuzvalidierung zum Tunen des BETA-PARAMETERS (mit {best_n_prototypes} Prototypen) ---")

# Definiere die zu testenden Beta-Werte
betas_to_test = [1, 2, 3, 4, 5]

# Speicher für die Ergebnisse der Beta-Kreuzvalidierung
beta_cv_results = {}

# Äußere Schleife: Iteriert über jeden zu testenden Beta-Wert
for beta_value in betas_to_test:
    print(f"\nTeste mit beta = {beta_value}...")
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    
    # Innere Schleife: Führt die 5-fache Kreuzvalidierung durch
    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_binary_np)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_binary_np[train_index], y_binary_np[test_index]
        
        # Initialisiere Modell mit bestem n_prototypes und aktuellem beta
        model_cv = GlvqModel(
            prototypes_per_class=best_n_prototypes, 
            beta=beta_value, 
            random_state=42
        )
        
        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)
        
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))

    # Berechne den Durchschnitt der Metriken über alle Folds
    beta_cv_results[beta_value] = {
        'mean_accuracy': np.mean(fold_accuracies),
        'mean_precision': np.mean(fold_precisions),
        'mean_recall': np.mean(fold_recalls)
    }
    
    print(f"  -> Avg. Accuracy: {beta_cv_results[beta_value]['mean_accuracy']:.4f}")
    print(f"  -> Avg. Precision: {beta_cv_results[beta_value]['mean_precision']:.4f}")
    print(f"  -> Avg. Recall: {beta_cv_results[beta_value]['mean_recall']:.4f}")

# Finde den besten Beta-Wert
best_beta = max(beta_cv_results, key=lambda k: beta_cv_results[k]['mean_accuracy'])

print("\n--- Stufe 2 abgeschlossen ---")
print(f"Bester Beta-Wert: {best_beta} (basierend auf der höchsten mittleren Genauigkeit)")


# === === === === === === === === ===
# ===  6. FINALES ERGEBNIS        ===
# === === === === === === === === ===
print("\n\n=======================================================")
print("===           FINALES TUNING-ERGEBNIS           ===")
print("=======================================================")
print(f"\nDie beste gefundene Hyperparameter-Kombination ist:")
print(f"  -> Anzahl Prototypen pro Klasse: {best_n_prototypes}")
print(f"  -> Beta-Parameter: {best_beta}")

# TODO:  GridSearchGV  ausprobieren