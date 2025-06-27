# .\venv311\Scripts\activate

#####################################
#
# Parameter Tuning via Kreuzvalidierung ( 5 Folds // 4 Folds dienen zum Training und 1 Fold zum Testen)
# Parameters:
# 1. Prototypen pro Klasse
# 2. Beta-Parameter
#
# Dataset splits (Training / Testing):
# 1. 80/20  
# 2. 90/10      
# Zum Dataset-Split muss der folgenden Parameter angepasst werden (jeweils 0.1 oder 0.2)   'test_size' - Zeile 113
#
#####################################

# === Bibliotheken importieren ===
# importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo
# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import StratifiedKFold, train_test_split
# um Merkmale zu standardisieren (Mittelwert 0, Standardabweichung 1)
from sklearn.preprocessing import StandardScaler
# GlvqModel: Die Klasse für das Generalized Learning Vector Quantization Modell
from sklearn_lvq import GlvqModel

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
# === NEU: Aufteilen in Trainings- und Test-Set     // jeweils  80/20 oder 90/10         
# === === === === === === === === === === === === === === === === === ===
# Die Kreuzvalidierung wird nur auf dem Trainingsset durchgeführt!
# Das Testset wird für eine spätere, finale Evaluation zurückgehalten.
X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
    X_scaled,
    y_binary_np,
    test_size=0.20,      # Hier Trainings-Set anpassen (entweder 20% oder 10%)
    random_state=42,     # Für reproduzierbare Ergebnisse 
    stratify=y_binary_np # Stellt sicher, dass die Klassenverteilung in beiden Sets gleich ist
)

print(f"\nDaten aufgeteilt: {len(X_train_cv)} Proben für Kreuzvalidierung und {len(X_test_holdout)} für das finale Testen.")


# === === === === === === === === === === === === === === === ===
# ===  4. Stufe 1: Tuning der Anzahl der Prototypen (1-5)     ===
# === === === === === === === === === === === === === === === ===
print("\n--- STUFE 1: Starte 5-fache Kreuzvalidierung zum Tunen der PROTOTYPEN-ANZAHL ---")

# Definiere die Anzahl der Folds (Teile) für die Kreuzvalidierung
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2) # sorgt für die gleiche Klassenverteilung in den Folds

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

    # Innere Schleife: Führt die 5-fache Kreuzvalidierung auf dem TRAININGSSET durch
    for fold, (train_index, test_index) in enumerate(skf.split(X_train_cv, y_train_cv)):
        X_train, X_test = X_train_cv[train_index], X_train_cv[test_index]
        y_train, y_test = y_train_cv[train_index], y_train_cv[test_index]

        model_cv = GlvqModel(prototypes_per_class=n_prototypes, random_state=2)
        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))

    # Berechne den Durchschnitt und Standardabweichung der Metriken über alle Folds
    proto_cv_results[n_prototypes] = {
        'mean_accuracy': np.mean(fold_accuracies),# <-- Durchschnitt
        'std_accuracy': np.std(fold_accuracies),  # <-- Standardabweichung
        'mean_precision': np.mean(fold_precisions),
        'std_precision': np.std(fold_precisions),
        'mean_recall': np.mean(fold_recalls),
        'std_recall': np.std(fold_recalls)
    }

    # Gib die Ergebnisse mit Durchschnitt und Standardabweichung aus
    print(f"  -> Durchschnitt Accuracy:  {proto_cv_results[n_prototypes]['mean_accuracy']:.4f} --- Standardabweichung: (+/- {proto_cv_results[n_prototypes]['std_accuracy']:.4f})")
    print(f"  -> Durchschnitt Precision: {proto_cv_results[n_prototypes]['mean_precision']:.4f} --- Standardabweichung: (+/- {proto_cv_results[n_prototypes]['std_precision']:.4f})")
    print(f"  -> Durchschnitt Recall:    {proto_cv_results[n_prototypes]['mean_recall']:.4f} --- Standardabweichung: (+/- {proto_cv_results[n_prototypes]['std_recall']:.4f})")


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

    # Innere Schleife: Führt die 5-fache Kreuzvalidierung auf dem TRAININGSSET durch
    for fold, (train_index, test_index) in enumerate(skf.split(X_train_cv, y_train_cv)):
        X_train, X_test = X_train_cv[train_index], X_train_cv[test_index]
        y_train, y_test = y_train_cv[train_index], y_train_cv[test_index]

        # Initialisiere Modell mit bestem n_prototypes und aktuellem beta
        model_cv = GlvqModel(
            prototypes_per_class=best_n_prototypes,
            beta=beta_value,
            random_state=2
        )

        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))

    # Berechne den Durchschnitt der Metriken über alle Folds
    beta_cv_results[beta_value] = {
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'mean_precision': np.mean(fold_precisions),
        'std_precision': np.std(fold_precisions),
        'mean_recall': np.mean(fold_recalls),
        'std_recall': np.std(fold_recalls)
    }

    # Gib die Ergebnisse mit Durchschnitt und Standardabweichung aus
    print(f"  -> Durchschnitt Accuracy: {beta_cv_results[beta_value]['mean_accuracy']:.4f} --- Standardabweichung: (+/- {beta_cv_results[beta_value]['std_accuracy']:.4f})")
    print(f"  -> Durchschnitt Precision: {beta_cv_results[beta_value]['mean_precision']:.4f} --- Standardabweichung: (+/- {beta_cv_results[beta_value]['std_precision']:.4f})")
    print(f"  -> Durchschnitt Recall: {beta_cv_results[beta_value]['mean_recall']:.4f} --- Standardabweichung: (+/- {beta_cv_results[beta_value]['std_recall']:.4f})")


# Finde den besten Beta-Wert
best_beta = max(beta_cv_results, key=lambda k: beta_cv_results[k]['mean_accuracy'])

print("\n--- Stufe 2 abgeschlossen ---")
print(f"Bester Beta-Wert: {best_beta} (basierend auf der höchsten mittleren Genauigkeit)")


# === === === === === === === === ===
# ===  6. FINALES ERGEBNIS        ===
# === === === === === === === === ===
print("\n\n==================================================")
print("===           FINALES TUNING-ERGEBNIS           ===")
print("===================================================")
print(f"\nDie beste gefundene Hyperparameter-Kombination ist:")
print(f"  -> Anzahl Prototypen pro Klasse: {best_n_prototypes}")
print(f"  -> Beta-Parameter: {best_beta}")

# Hole die Leistungsmetriken für die beste Kombination
final_performance = beta_cv_results[best_beta] # enthält die besten Beta-Parameter und Anzahl Prototypen für max. Accuracy

# Leistungsmetriken fürs finale Modell !
print("\nLeistung des finalen Modells (geschätzt durch 5-fache Kreuzvalidierung auf den 80% Trainingsdaten):")
print(f"  -> Accuracy:  {final_performance['mean_accuracy']:.4f} (+/- {final_performance['std_accuracy']:.4f})")
print(f"  -> Precision: {final_performance['mean_precision']:.4f} (+/- {final_performance['std_precision']:.4f})")
print(f"  -> Recall:    {final_performance['mean_recall']:.4f} (+/- {final_performance['std_recall']:.4f})")


# === === === === === === === === === === === === === === === === ===
# ===  7. FINALE EVALUATION AUF UNGESEHENEN TESTDATEN (20 % oder 10 %)    ===
# === === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===  FINALE EVALUATION AUF DEM UNBERÜHRTEN TEST-SET  ===")
print("================================================================")

# 1. Trainiere das finale Modell mit den besten Parametern auf dem GESAMTEN Trainingsset 
print("\nTrainiere finales Modell mit den besten Hyperparametern...")
final_model = GlvqModel(
    prototypes_per_class=best_n_prototypes,
    beta=best_beta,
    random_state=2
)
final_model.fit(X_train_cv, y_train_cv)
print("Training des finalen Modells abgeschlossen.")

# 2. Mache Vorhersagen auf dem zurückgehaltenen Test-Set (Holdout-Set)
print("Mache Vorhersagen auf dem Test-Set...")
y_pred_final = final_model.predict(X_test_holdout)

# 3. Berechne die finalen Leistungsmetriken
final_accuracy = accuracy_score(y_test_holdout, y_pred_final)
final_precision = precision_score(y_test_holdout, y_pred_final, zero_division=0)
final_recall = recall_score(y_test_holdout, y_pred_final, zero_division=0)

# 4. Gib die finalen Ergebnisse aus
print("\nLeistung des finalen Modells auf den ungesehenen Testdaten:")
print(f"  -> Finale Accuracy:  {final_accuracy:.4f}")
print(f"  -> Finale Precision: {final_precision:.4f}")
print(f"  -> Finale Recall:    {final_recall:.4f}")



# === === === === === === === === === === === === === === === ===
# ===  8. KONFUSIONSMATRIX FÜR DAS FINALE TEST-SET           ===
# === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===         KONFUSIONSMATRIX FÜR DAS TEST-SET             ===")
print("================================================================")

# 1. Berechne die Konfusionsmatrix
cm = confusion_matrix(y_test_holdout, y_pred_final)

# 2. Erstelle eine  Visualisierung mit Seaborn
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