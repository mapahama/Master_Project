
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
# Zum Dataset-Split muss der folgenden Parameter angepasst werden (jeweils 0.1 oder 0.2)   'test_size'
#
#####################################

# === Bibliotheken importieren ===
# importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo

# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import StratifiedKFold, train_test_split
# um Merkmale zu standardisieren (Mittelwert 0, Standardabweichung 1)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# GlvqModel: Die Klasse für das Generalized Learning Vector Quantization Modell
from sklearn_lvq import GlvqModel
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
print("Vorverarbeitung abgeschlossen.")

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
# === 4. Aufteilen in Trainings- und Test-Set VOR der Skalierung
# === === === === === === === === === === === === === === === === === ===
# Die Kreuzvalidierung wird nur auf dem Trainingsset durchgeführt!
# Das Testset wird für eine spätere, finale Evaluation zurückgehalten.

# Die BEREINIGTEN Daten (X_clean, y_clean) werden hier verwendet.
X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
    X_clean,
    y_clean,
    test_size=0.20,      # Hier Trainings-Set anpassen (entweder 20% oder 10%)
    random_state=42,     # Für reproduzierbare Ergebnisse
    stratify=y_clean     # Stellt sicher, dass die Klassenverteilung in beiden Sets gleich ist
)

print(f"\nDaten aufgeteilt: {len(X_train_cv)} Proben für Kreuzvalidierung und {len(X_test_holdout)} für das finale Testen.")

# === Skalierung NACH der Aufteilung ===
# Ziel: alle Merkmale  auf eine ähnliche Skala (Bereich [-1,1]) zu bringen!
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_cv = scaler.fit_transform(X_train_cv)
X_test_holdout = scaler.transform(X_test_holdout) 
print("Skalierung der Trainings- und Testsets abgeschlossen.")


# === === === === === === === === === === === === === === === ===
# ===  5. Stufe 1: Tuning der Anzahl der Prototypen (1-5)    ===
# === === === === === === === === === === === === === === === ===
print("\n--- STUFE 1: Starte 5-fache Kreuzvalidierung zum Tunen der PROTOTYPEN-ANZAHL ---")

# Definiere die Anzahl der Folds (Teile) für die Kreuzvalidierung
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42) # sorgt für die gleiche Klassenverteilung in den Folds

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
    # y_train_cv muss zu einem NumPy-Array konvertiert werden, da es ein Pandas-Series ist
    for fold, (train_index, test_index) in enumerate(skf.split(X_train_cv, y_train_cv.to_numpy())):
        X_train, X_test = X_train_cv[train_index], X_train_cv[test_index]
        y_train, y_test = y_train_cv.to_numpy()[train_index], y_train_cv.to_numpy()[test_index]

        model_cv = GlvqModel(prototypes_per_class=n_prototypes, random_state=42)
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
# ===  5. Stufe 2: Tuning des Beta-Parameters        ===
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
    for fold, (train_index, test_index) in enumerate(skf.split(X_train_cv, y_train_cv.to_numpy())):
        X_train, X_test = X_train_cv[train_index], X_train_cv[test_index]
        y_train, y_test = y_train_cv.to_numpy()[train_index], y_train_cv.to_numpy()[test_index]

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
print("===           FINALES TUNING-ERGEBNIS            ===")
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
print("===   FINALE EVALUATION AUF DEM UNBERÜHRTEN TEST-SET   ===")
print("================================================================")

# 1. Trainiere das finale Modell mit den besten Parametern auf dem GESAMTEN Trainingsset
print("\nTrainiere finales Modell mit den besten Hyperparametern...")
final_model = GlvqModel(
    prototypes_per_class=best_n_prototypes,
    beta=best_beta,
    random_state=42
)
# Das finale Modell wird auf dem gesamten (bereits skalierten) Kreuzvalidierungs-Set trainiert
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
# ===  8. KONFUSIONSMATRIX FÜR DAS FINALE TEST-SET        ===
# === === === === === === === === === === === === === === === ===

print("\n\n================================================================")
print("===        KONFUSIONSMATRIX FÜR DAS TEST-SET         ===")
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

#######################
# Ergebnisse:
#######################

# Leistung des finalen Modells auf den ungesehenen Testdaten (80/20 Dataset-Split):
#  -> Finale Accuracy:  0.8750
#  -> Finale Precision: 0.8750
#  -> Finale Recall:    0.8400
#
# Die beste gefundene Hyperparameter-Kombination ist:
#  -> Anzahl Prototypen pro Klasse: 1
#  -> Beta-Parameter: 3



# Leistung des finalen Modells auf den ungesehenen Testdaten (90/10 Dataset-Split):
#  -> Finale Accuracy:  0.8571
#  -> Finale Precision: 0.8333
#  -> Finale Recall:    0.8333
#
# Die beste gefundene Hyperparameter-Kombination ist:
#  -> Anzahl Prototypen pro Klasse: 1
#  -> Beta-Parameter: 5