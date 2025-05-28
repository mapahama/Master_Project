# .\venv311\Scripts\activate

# === Bibliotheken importieren ===
# Dieser Block importiert alle notwendigen Python-Bibliotheken und spezifische Module/Funktionen.

# Heart Disease Dataset
from ucimlrepo import fetch_ucirepo
# um einen Datensatz in Trainings- und Test-Subsets aufzuteilen
from sklearn.model_selection import train_test_split
# um Merkmale zu standardisieren (Mittelwert 0, Standardabweichung 1)
from sklearn.preprocessing import StandardScaler
# GlvqModel: Die Klasse für das Generalized Learning Vector Quantization Modell
from sklearn_lvq import GlvqModel

# Verschiedene Metriken aus `sklearn.metrics` zur Evaluation des Klassifikationsmodells:
from sklearn.metrics import (
    confusion_matrix,           # Zur Berechnung der Konfusionsmatrix (zeigt richtig/falsch klassifizierte Instanzen pro Klasse).
    ConfusionMatrixDisplay,     # Zum einfachen grafischen Darstellen der Konfusionsmatrix.
    precision_score,            # Zur Berechnung der Präzision (Anteil der korrekt positiven Vorhersagen an allen positiven Vorhersagen).
    recall_score,               # Zur Berechnung des Recalls/Sensitivität (Anteil der korrekt positiven Vorhersagen an allen tatsächlichen Positiven)
    f1_score,                   # Zur Berechnung des F1-Scores (harmonisches Mittel aus Precision und Recall).
    classification_report       # Zur Erstellung eines Textberichts mit den wichtigsten Metriken (Precision, Recall, F1-Score, Support) pro Klasse
)

# Bibliothek für Datenmanipulation und -analyse
import pandas as pd
# Bibliothek für numerische Berechnungen in Python (Arrays und Matrizen)
import numpy as np
# Erstellen von Plots und Visualisierungen 
import matplotlib.pyplot as plt
# Farbskalen für Plots.
from matplotlib.colors import ListedColormap
# statistische Grafiken
import seaborn as sns
# benutzerdefinierte Legenden in Plots.
from matplotlib.lines import Line2D

# Klasse für die Hauptkomponentenanalyse (Principal Component Analysis) verwendet zur Dimensionsreduktion
from sklearn.decomposition import PCA

# === === === === === ===  === 
# ===  1. Dataset laden    ===
# === === === === === ===  === 
print("Lade Heart Disease Datensatz...")
# Ruft den "Heart Disease" Datensatz (mit der ID 45) vom UCI Machine Learning Repository ab.
heart_disease = fetch_ucirepo(id=45)

# Extrahiert die Merkmalsdaten (Features) aus dem geladenen Objekt
X = heart_disease.data.features.copy()

# Extrahiert die Zielvariable(n) (Targets) aus dem geladenen Objekt.
y = heart_disease.data.targets.copy()
print("Datensatz geladen.")

# === === === === === === === === === 
# ===  2. Targets binär machen    ===
# === === === === === === === === === 
# Die ursprüngliche Zielvariable dieses Datensatzes kann mehrere Werte für den Schweregrad
# der Herzerkrankung enthalten (0 = gesund, 1-4 = verschiedene Stufen von krank).
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
print(y_binary.value_counts())  # 0    164
                                # 1    139

# === === === === === === === 
# ===  3. Vorverarbeitung ===
# === === === === === === === 
# Merkmalsdaten (X) bereinigen und für das Modell vorbereiten
print("Starte Vorverarbeitung...")
# Ersetze fehlende Werte, die im Datensatz als '?' (Fragezeichen) kodiert sind, durch `np.nan`.
X.replace('?', np.nan, inplace=True)

# Versuche, alle Spalten des DataFrames `X` in numerische Werte umzuwandeln.
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
print("Vorverarbeitung abgeschlossen.")

# === === === === === === ===  === 
# ===   4. Train/Test-Split    ===
# === === === === === === ===  === 
# Aufteilung der Daten in Trainings- und Testsets, um das Modell auf einem Teil der Daten zu trainieren
# und auf einem separaten, ungesehenen Teil zu evaluieren.

# `stratify=y_binary`: Stellt sicher, dass das Verhältnis der Klassen (0 und 1) in den Trainings-
# und Testsets ungefähr dem Verhältnis im gesamten Datensatz `y_binary` entspricht.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"Daten aufgeteilt: {X_train.shape[0]} Trainingspunkte, {X_test.shape[0]} Testpunkte.")

# === === === === === === === === === === === 
# ===  5. GLVQ mit mehreren Prototypen    ===
# === === === === === === === === === === ===  
# Initialisierung und Training des GLVQ-Modells.

# Erstelle eine Instanz des 'GlvqModel'.
model = GlvqModel(prototypes_per_class=9, random_state=42) # <-- hier wird Anzahl von Prototypen bestimmt! TODO - dynamisch machen
print(f"Trainiere GLVQ-Modell mit {model.prototypes_per_class} Prototypen pro Klasse...")

# Trainiere das GLVQ-Modell mit den Trainingsdaten (`X_train`, `y_train`).
model.fit(X_train, y_train)
print("Modelltraining abgeschlossen.")

# === === === === === === === === === 
# ===  6. Vorhersage & Auswertung === 
# === === === === === === === === === 
# Evaluation des trainierten Modells auf dem ungesehenen Testset

print("\n--- Starte Evaluation auf dem Testset (2 Klassen) ---")
# Mache Vorhersagen für die Merkmale des Testsets (`X_test`).
# `model.predict()` gibt ein Array mit den vorhergesagten Klassenlabels (0 oder 1) zurück
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
# Gib die Genauigkeit aus, formatiert auf 4 Nachkommastellen.
print(f"Accuracy: {accuracy:.4f}")

# --- Konfusionsmatrix ---
# Berechne die Konfusionsmatrix. Sie zeigt die Anzahl der True Positives, True Negatives,
# False Positives und False Negatives.
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Gesund (0)", "Krank (1)"])
# Plotte die Konfusionsmatrix.
disp.plot(cmap=plt.cm.Blues, values_format='d')
# Setze einen Titel für den Plot, der auch die erreichte Genauigkeit anzeigt
plt.title(f"Confusion Matrix (2 Klassen)\nAccuracy: {accuracy:.2f}")

# --- Verteilung der vorhergesagten Klassen ---
# Zähle wie oft jede Klasse (0 und 1) in den Vorhersagen `y_pred` vorkommt
unique_preds, counts_preds = np.unique(y_pred, return_counts=True)
# Erstelle ein Dictionary, um die Zählungen den Klassen zuzuordnen
pred_counts_map = dict(zip(unique_preds, counts_preds))
# Erstelle eine Liste der Zählungen in der Reihenfolge [Klasse 0, Klasse 1]
ordered_counts_pred = [pred_counts_map.get(0,0), pred_counts_map.get(1,0)]

# Erstelle eine neue Figur für das Balkendiagramm. (krank, gesund - tätsächlich vs vorhergesagt)
plt.figure(figsize=(6, 4))
plt.bar(["Gesund (0)", "Krank (1)"], ordered_counts_pred, color=["skyblue", "salmon"])
plt.title("Vorhergesagte Klassenzuordnung")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Anzahl")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# --- Vergleich: Tatsächlich vs. Vorhergesagt ---
# Zähle die Verteilung der wahren Klassen im Testset `y_test`.
actual_unique, actual_counts_val = np.unique(y_test, return_counts=True)
classes_defined = [0, 1]
actual_dict = dict(zip(actual_unique, actual_counts_val))
actual_plot_values = [actual_dict.get(cls, 0) for cls in classes_defined]
x_plot_indices = np.arange(len(classes_defined))
bar_width_val = 0.35

# Erstelle eine neue Figur.
plt.figure(figsize=(7, 4))
plt.bar(x_plot_indices - bar_width_val/2, actual_plot_values, bar_width_val, label="Tatsächlich", color="lightgreen")
plt.bar(x_plot_indices + bar_width_val/2, ordered_counts_pred, bar_width_val, label="Vorhergesagt", color="coral")
plt.xlabel("Klasse")
plt.ylabel("Anzahl")
plt.title("Tatsächlich vs. Vorhergesagt")
plt.xticks(x_plot_indices, ["Gesund (0)", "Krank (1)"])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# --- Weitere Metriken ---
# Berechne Precision
precision = precision_score(y_test, y_pred, zero_division=0)
# Berechne Recall.
recall = recall_score(y_test, y_pred, zero_division=0)
# berechne F1-Score
f1 = f1_score(y_test, y_pred, zero_division=0)
# Gib die einzelnen Metriken aus
print("\n--- Bewertungsmetriken ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}") 
print(f"F1 Score:  {f1:.4f}")   

# Gib einen  Klassifikationsbericht aus
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Gesund (0)", "Krank (1)"], zero_division=0))
# Zeige alle bisher erstellten Matplotlib-Plots an.
plt.show()
print("--- Evaluation abgeschlossen ---")

# === === === === === === === ===  
# ===  7. Prototypen anzeigen ===
# === === === === === === === === 
# Analyse der vom GLVQ-Modell gelernten Prototypen.

print("\n--- Analyse der gelernten Prototypen ---")
# `model.w_` enthält ein NumPy-Array, wobei jede Zeile ein Prototypvektor ist (im skalierten Merkmalsraum)
prototypes = model.w_
# `model.c_w_` enthält ein Array mit den Klassenlabels (0 oder 1) für jeden Prototyp
proto_labels = model.c_w_
features = X.columns.tolist()

# Iteriere über jeden Prototyp, um seine Werte auszugeben.
for i, p_coords in enumerate(prototypes):
    # Hole das numerische Label des aktuellen Prototyps
    p_class_label_numeric = proto_labels[i]
    # Wandle das numerische Label in einen lesbaren Text um
    p_class_label_text = "Krank" if p_class_label_numeric == 1 else "Gesund"
    # Gib Index und Klasse des Prototyps aus
    print(f"\n Prototyp {i} (Klasse {p_class_label_numeric} - '{p_class_label_text}'):")
    # Iteriere über die Merkmalsnamen und die entsprechenden Werte im Prototypvektor.
    for name, val in zip(features, p_coords):
        # Gib den Merkmalsnamen und den (standardisierten) Wert des Prototyps für dieses Merkmal aus
        print(f"  {name}: {val:.3f}") # Formatiert auf 3 Nachkommastellen

# === === === === === === === === === === === ===
# ===  8. Prototyp vs. Durchschnitt (krank)   ===
# === === === === === === === === === === === ===
# Vergleicht die Merkmalswerte eines gelernten "Krank"-Prototyps

print("\n--- Vergleich 'Krank'-Prototyp vs. Durchschnitt 'Krank'-Patient ---")
# Finde die Indizes aller Prototypen, die zur Klasse 1 ('Krank') gehören.
indices_kranke_prototypen = np.where(proto_labels == 1)[0]
if len(indices_kranke_prototypen) > 0:
    # Wähle den Index des ersten gefundenen "Krank"-Prototyps für diesen Vergleich
    index_krank_proto_selected = indices_kranke_prototypen[0]
    # hole den Merkmalsvektor dieses ausgewählten Prototyps.
    proto_krank_for_comparison = prototypes[index_krank_proto_selected]

    # Berechne den durchschnittlichen Merkmalsvektor für alle Patienten im Trainingsset,
    mean_patient_krank_train = np.mean(X_train[y_train == 1], axis=0)
    # --- Plot: Vergleich Prototyp vs. Durchschnitt ---
    x_feature_positions = np.arange(len(features))
    bar_width_compare = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x_feature_positions - bar_width_compare/2, proto_krank_for_comparison, bar_width_compare, label=f"Prototyp {index_krank_proto_selected} (Krank)", color='salmon')
    plt.bar(x_feature_positions + bar_width_compare/2, mean_patient_krank_train, bar_width_compare, label="Durchschnitt 'Krank' Patient (Training)", color='skyblue')
    plt.xticks(x_feature_positions, features, rotation=45, ha="right")
    plt.ylabel("Merkmalswert (standardisiert)")
    plt.title("Vergleich: 'Krank'-Prototyp vs. Durchschnittlicher 'Krank'-Patient")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    # Fallback, falls keine "Krank"-Prototypen gelernt wurden (unwahrscheinlich)
    print("Kein Prototyp für Klasse 'Krank' gefunden für den Vergleich.")

# === === === === === === === === === === === === ===
# ===  9. Patientenanalyse mit Prototyp-Vergleich ===
# === === === === === === === === === === === === ===
# Detaillierte Analyse eines einzelnen Patienten aus dem Testset.
print("\n--- Analyse eines einzelnen Testpatienten ---")

# === Initialisierung der Variablen für den spezifischen Patienten  ===
patient_index_for_analysis = 5 # Index des Patienten im Testset, der analysiert werden soll (0-basiert)
specific_patient_features = None   # Wird die Merkmale des ausgewählten Patienten speichern
specific_patient_prediction = None # Wird die Vorhersage für den ausgewählten Patienten speichern

if patient_index_for_analysis < len(X_test):
    # Hole die (skalierten) Merkmale des ausgewählten Patienten
    specific_patient_features = X_test[patient_index_for_analysis]
    # Hole das wahre Klassenlabel des Patienten.
    patient_true_label = y_test.iloc[patient_index_for_analysis]

    # Mache eine Vorhersage für diesen einzelnen Patienten.
    specific_patient_prediction = model.predict([specific_patient_features])[0]
    # Gib wahre und vorhergesagte Klasse aus.
    print(f"Patient #{patient_index_for_analysis} - Wahre Klasse: {'Krank' if patient_true_label == 1 else 'Gesund'}")
    print(f"Patient #{patient_index_for_analysis} - Vorhergesagte Klasse: {'Krank' if specific_patient_prediction == 1 else 'Gesund'}")

    # Berechne die euklidischen Distanzen vom Patientenvektor zu *allen* gelernten Prototypen
    distances_to_all_prototypes = [np.linalg.norm(specific_patient_features - proto) for proto in prototypes]

    # Erstelle einen Pandas DataFrame, um die Informationen zu den Prototypen und ihren Distanzen zum Patienten übersichtlich darzustellen.
    prototype_info_df = pd.DataFrame({
        "Prototyp-Index": list(range(len(prototypes))), 
        "Prototyp-Klasse-Num": proto_labels,           
        "Prototyp-Klasse-Text": ["Krank" if label == 1 else "Gesund" for label in proto_labels], 
        "Distanz_zum_Patienten": distances_to_all_prototypes # Berechnete Distanz
    }).sort_values("Distanz_zum_Patienten") 

    print("\nDistanzen des Patienten zu den Prototypen (sortiert nach Distanz):")
    print(prototype_info_df)
    # === === === === === === === === === === === === === === === ===
    # ===  9.1 Visualisierung der Distanzen zu allen Prototypen   ===
    # === === === === === === === === === === === === === === === ===
    plt.figure(figsize=(10, max(4, len(prototypes) * 0.6))) 
    prototype_display_labels_dist_plot = [f"P{row['Prototyp-Index']} ({row['Prototyp-Klasse-Text']})" for index, row in prototype_info_df.iterrows()]
    bar_colors_dist_plot = ['salmon' if cls_text == 'Krank' else 'skyblue' for cls_text in prototype_info_df["Prototyp-Klasse-Text"]]
    bars_dist = plt.barh(prototype_display_labels_dist_plot, prototype_info_df["Distanz_zum_Patienten"], color=bar_colors_dist_plot)
    if len(bars_dist) > 0: 
        bars_dist[0].set_color('limegreen') # Der erste Balken (kleinste Distanz) wird grün.
        bars_dist[0].set_edgecolor('black') 

    plt.xlabel(f"Euklidische Distanz zum Patienten #{patient_index_for_analysis}")
    plt.ylabel("Prototypen")
    plt.title(f"Distanzen von Patient #{patient_index_for_analysis} zu allen GLVQ-Prototypen (sortiert)")
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7) 
    plt.show() 

    # --- Detaillierter Vergleich mit dem nächstgelegenen "Krank"-Prototyp (unabhängig von der Vorhersage) ---
    krank_prototypes_info = prototype_info_df[prototype_info_df["Prototyp-Klasse-Num"] == 1]
    if not krank_prototypes_info.empty:
        # Wähle den "Krank"-Prototyp, der dem Patienten am nächsten ist 
        closest_krank_prototype_from_list = krank_prototypes_info.iloc[0]
        # Hole Index und Koordinaten dieses spezifischen Prototyps
        proto_index_for_krank_specific_compare = int(closest_krank_prototype_from_list["Prototyp-Index"])
        proto_coords_for_krank_specific_compare = prototypes[proto_index_for_krank_specific_compare]

        # Erstelle einen DataFrame für den Merkmalsvergleich: Patient vs. dieser nächste "Krank"-Prototyp.
        df_compare_vs_specific_krank = pd.DataFrame({
            "Merkmal": features,
            "Patientenwert": specific_patient_features,
            f"Nächster 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare})": proto_coords_for_krank_specific_compare
        })
        # Berechne die absolute Differenz pro Merkmal.
        df_compare_vs_specific_krank["Differenz"] = abs(df_compare_vs_specific_krank["Patientenwert"] - df_compare_vs_specific_krank[f"Nächster 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare})"])

        # === === === === === === === === === === === === === === === === === ===
        # ===  10. Balkendiagramm: Abweichung vom (nächsten kranken) Prototyp ===
        # === === === === === === === === === === === === === === === === === ===
        # Sortiere nach der Differenz (absteigend), um die größten Abweichungen oben zu haben.
        df_sorted_diff_vs_specific_krank = df_compare_vs_specific_krank.sort_values("Differenz", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.barh(df_sorted_diff_vs_specific_krank["Merkmal"], df_sorted_diff_vs_specific_krank["Differenz"], color="salmon")
        plt.xlabel(f"Absolute Differenz zum nächsten 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare}, standardisiert)")
        plt.title(f"Unterschiede: Patient #{patient_index_for_analysis} vs. nächster 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare})")
        plt.gca().invert_yaxis() # größte Differenz oben
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show() 

        # === === === === === === === === === === === === === === === === === === ===
        # ===  11. Heatmap: Patientenwerte vs. (nächster kranker) Prototypwerte   ===
        # === === === === === === === === === === === === === === === === === === ===        
        # Bereite Daten für die Heatmap vor. Setze "Merkmal" als Index
        df_heat_specific_krank = df_compare_vs_specific_krank.set_index("Merkmal")[["Patientenwert", f"Nächster 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare})"]]
        plt.figure(figsize=(10, 4)) 
        sns.heatmap(df_heat_specific_krank.T, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Standardisierter Wert'})
        plt.title(f"Heatmap Vergleich: Patient #{patient_index_for_analysis} vs. nächster 'Krank'-Prototyp (P{proto_index_for_krank_specific_compare})")
        plt.ylabel("Datenquelle")
        plt.tight_layout()
        plt.show() 
    else:
        print(f"Kein 'Krank'-Prototyp für detaillierten Vergleich mit Patient #{patient_index_for_analysis} vorhanden.")

    # === === === === === === === === === === === === 
    # ===  12. Automatisch generierte Erklärung   ===
    # === === === === === === === === === === === ===
    # Die Erklärung bezieht sich auf den *Gewinner-Prototyp* 
    winning_proto_idx_patient = int(prototype_info_df.iloc[0]["Prototyp-Index"]) # Index des Gewinners
    winning_proto_class_text_patient = prototype_info_df.iloc[0]["Prototyp-Klasse-Text"] # Klasse des Gewinners
    winning_proto_coords_for_patient = prototypes[winning_proto_idx_patient] # Koordinaten des Gewinners

    # Erstelle einen DataFrame für den Vergleich mit dem Gewinner-Prototyp.
    df_compare_patient_vs_winner = pd.DataFrame({
        "Merkmal": features,
        "Patientenwert": specific_patient_features,
        f"Gewinner-Prototyp (P{winning_proto_idx_patient}, {winning_proto_class_text_patient})": winning_proto_coords_for_patient
    })
    # Berechne die Differenzen zum Gewinner-Prototyp
    df_compare_patient_vs_winner["Differenz_zum_Gewinner"] = abs(df_compare_patient_vs_winner["Patientenwert"] - df_compare_patient_vs_winner[f"Gewinner-Prototyp (P{winning_proto_idx_patient}, {winning_proto_class_text_patient})"])
    
    # Finde die Top 3 Merkmale mit der KLEINSTEN Differenz (d.h. größte Ähnlichkeit) zum Gewinner-Prototyp.
    top_features_closest_to_winner = df_compare_patient_vs_winner.sort_values("Differenz_zum_Gewinner", ascending=True).head(3)["Merkmal"].values

    # Gib die generierte Erklärung aus
    print(f"\n Beispiel-Erklärung für Klassifikation von Patient #{patient_index_for_analysis}:")
    print(f"Der Patient wurde als '{'Krank' if specific_patient_prediction == 1 else 'Gesund'}' klassifiziert.")
    print(f"Dies basiert auf der größten Ähnlichkeit (kleinste Distanz: {prototype_info_df.iloc[0]['Distanz_zum_Patienten']:.2f}) zum Prototyp #{winning_proto_idx_patient}, der repräsentativ für die Klasse '{winning_proto_class_text_patient}' ist.")
    print(f"Die Merkmale '{', '.join(top_features_closest_to_winner)}' des Patienten zeigen die größte Übereinstimmung mit diesem 'Referenz'-Prototyp.")
else:
    print(f"Fehler: patient_index_for_analysis {patient_index_for_analysis} ist außerhalb des Bereichs für das Testset (Größe {len(X_test)}). Detaillierte Patientenanalyse übersprungen.")


# === === === === === === === === === === === === === === === === === === ===
# ===   13. Visualisierung der Entscheidungsgrenzen (mittels PCA in 2D)   ===
# === === === === === === === === === === === === === === === === === === ===
# visualisiert die vom GLVQ-Modell gelernten Entscheidungsgrenzn

print("\n\n--- Visualisierung der Entscheidungsgrenzen (2D PCA, 2 Klassen) mit Patient Hervorhebung ---")

# 1. PCA zur Dimensionsreduktion
pca = PCA(n_components=2)
# Passe das PCA-Modell an die Trainingsdaten an (`fit`) und transformiere sie dann (`transform`).
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test)     # X_test ist bereits skaliert
prototypes_pca = pca.transform(prototypes)

# 2. Erstellen eines Meshgrids für den Plot
all_pca_points_x = np.concatenate((X_test_pca[:, 0], prototypes_pca[:, 0])) # Alle x-Koordinaten
all_pca_points_y = np.concatenate((X_test_pca[:, 1], prototypes_pca[:, 1])) # Alle y-Koordinaten

# Definiere Min/Max-Werte für die Achsen mit einem kleinen Rand.
x_min, x_max = all_pca_points_x.min() - 1, all_pca_points_x.max() + 1
y_min, y_max = all_pca_points_y.min() - 1, all_pca_points_y.max() + 1
# Definiere die Schrittweite (Auflösung) des Gitters.
mesh_step_size = 0.05
# Erstelle das Meshgrid 
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                     np.arange(y_min, y_max, mesh_step_size))

# 3. Vorhersage für jeden Punkt im Meshgrid
mesh_points = np.c_[xx.ravel(), yy.ravel()]
# Initialisiere ein Array für die vorhergesagten Labels der Gitterpunkte.
predicted_mesh_labels = np.zeros(mesh_points.shape[0])
# Iteriere über jeden Gitterpunkt.
for i, point in enumerate(mesh_points):
    # Berechne die Distanzen dieses Gitterpunkts zu allen *2D-transformierten Prototypen*
    distances_to_pca_protos = [np.linalg.norm(point - single_proto_pca) for single_proto_pca in prototypes_pca]
    # Finde den Index des Prototyps mit der kleinsten Distanz.
    closest_proto_idx = np.argmin(distances_to_pca_protos)
    # Weise dem Gitterpunkt das Klassenlabel dieses nächstgelegenen Prototyps zu.
    predicted_mesh_labels[i] = proto_labels[closest_proto_idx]
# `Z` enthält  für jeden Gitterpunkt das vorhergesagte Klassenlabel (0 oder 1)
Z = predicted_mesh_labels.reshape(xx.shape)

# 4. Plotten der Entscheidungsgrenzen und Datenpunkte
plt.figure(figsize=(12, 8))

# Definiere  Farbskalen 
cmap_regions_binary = ListedColormap(['lightblue', 'lightcoral']) 
cmap_points_binary = ListedColormap(['blue', 'red'])             

# Definiere die Levels für `contourf`, um klare Grenzen zwischen den Klassen 0 und 1 zu zeichnen.
levels_binary = [-0.5, 0.5, 1.5]
# Zeichne die Entscheidungsregionen als gefüllte Konturen
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_regions_binary, levels=levels_binary)

# Zeichne die Test-Datenpunkte (PCA-transformiert) als Scatterplot.
scatter_test_points_binary = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                                         s=40, edgecolor='k', alpha=0.7, cmap=cmap_points_binary, zorder=2)

# Zeichne die GLVQ-Prototypen (PCA-transformiert) als größere 'X'-Marker.
scatter_prototypes_binary = plt.scatter(prototypes_pca[:, 0], prototypes_pca[:, 1],
                                        c=proto_labels, marker='X', s=250,
                                        edgecolor='black', linewidth=1.5, cmap=cmap_points_binary, zorder=3)

# --- Plotten des spezifischen Patienten (z.B. Patient #5) --- (hardcoded Patient #5) TODO - löschen
if specific_patient_features is not None and specific_patient_prediction is not None:
    # Transformiere die Merkmale des spezifischen Patienten in den 2D-PCA-Raum.
    specific_patient_pca_coords = pca.transform([specific_patient_features]) # Ergibt [[pc1, pc2]]
    color_for_specific_patient = cmap_points_binary.colors[specific_patient_prediction]
    
    # Plotte den spezifischen Patienten als hervorgehobenen Punkt.
    plt.scatter(specific_patient_pca_coords[0, 0],  
                specific_patient_pca_coords[0, 1],  
                marker='P',                          
                s=350,                              
                color=color_for_specific_patient,    
                edgecolor='yellow',                  
                linewidth=2,                         
                label=f"Patient #{patient_index_for_analysis} (Vorh.: {'Krank' if specific_patient_prediction == 1 else 'Gesund'})", 
                zorder=4)                           
else:
    print(f"Merkmale/Vorhersage für Patient #{patient_index_for_analysis} nicht verfügbar für PCA-Plot-Hervorhebung.")


# Setze Titel und Achsenbeschriftungen des Plots.
title_pca_plot = f'GLVQ Entscheidungsgrenzen (2D PCA, 2 Klassen)'
if specific_patient_features is not None:
    title_pca_plot += f' mit Patient #{patient_index_for_analysis}'
title_pca_plot += '\nUCI Heart Disease Dataset'
plt.title(title_pca_plot)
plt.xlabel('Erste Hauptkomponente (PCA1)')
plt.ylabel('Zweite Hauptkomponente (PCA2)')

# Erstelle eine  Legende für den Plot.
legend_handles_pca_binary = [
    Line2D([0], [0], marker='o', color='w', label='Testdaten Gesund (Kl.0)',
           markerfacecolor=cmap_points_binary.colors[0], markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Testdaten Krank (Kl.1)',
           markerfacecolor=cmap_points_binary.colors[1], markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='X', color='w', label='Prototyp (entspr. Farbe)',
           markeredgecolor='black', markerfacecolor='grey', markersize=12, linestyle='None'),
]

if specific_patient_features is not None and specific_patient_prediction is not None:
    legend_handles_pca_binary.append(
        Line2D([0], [0], marker='P', color='w', label=f"Patient #{patient_index_for_analysis} (Vorh.: {'Krank' if specific_patient_prediction == 1 else 'Gesund'})",
               markerfacecolor=color_for_specific_patient, markeredgecolor='yellow', markersize=12, linestyle='None')
    )
plt.legend(handles=legend_handles_pca_binary, title="Legende")

# Füge ein Raster hinzu und optimiere das Layout.
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("--- Visualisierung der Entscheidungsgrenzen (2 Klassen) abgeschlossen ---")

# === === === === === === === === === === === === === === 
# ===  14 μ-Werte berechnen und Histogramm darstellen ===
# === === === === === === === === === === === === === === 
# Analyse der Klassifikationssicherheit 

print("\n--- Analyse der Klassifikationssicherheit mit μ-Werten ---") 
# Gibt eine Überschrift in der Konsole aus, um den Beginn dieses Code-Abschnitts zu markieren. 
# Das "\n" sorgt für eine Leerzeile davor zur besseren Lesbarkeit.

def glvq_mu(x, w_j, w_k):
    d_j = np.sum((x - w_j) ** 2)
    # Berechnet die QUADRIERTE euklidische Distanz zwischen dem Datenpunkt `x` und dem Prototyp `w_j`.
    d_k = np.sum((x - w_k) ** 2)
    return (d_j - d_k) / (d_j + d_k)
    # Berechnet den μ-Wert nach der GLVQ-Formel: (d_j - d_k) / (d_j + d_k).
    # Diese Formel normalisiert die Differenz der Distanzen durch ihre Summe.
    # Ein negativer Wert bedeutet, dass d_j (Distanz zum 'richtigen' Prototyp) kleiner 

mu_values = []

# Nutze die gelernten Prototypen und Klassenzugehörigkeit
# Greift auf das Attribut `w_` des trainierten GLVQ-Modells (`model`) zu. 
prototypes = model.w_
# Greift auf das Attribut `c_w_` des trainierten GLVQ-Modells zu. 
proto_labels = model.c_w_

# Startet eine Schleife, die durch jeden einzelnen Datenpunkt im Testset iteriert
for i in range(len(X_test)):
    x = X_test[i]
    y_true = y_test.iloc[i]

    w_j = prototypes[proto_labels == y_true][0]
    w_k = prototypes[proto_labels != y_true][0]

    mu = glvq_mu(x, w_j, w_k)
    # Fügt den berechneten μ-Wert für den aktuellen Datenpunkt zur Liste `mu_values` hinzu.
    mu_values.append(mu)

# Zeichne Histogramm
plt.figure(figsize=(8, 5))
plt.hist(mu_values, bins=20, color='mediumseagreen', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='μ = 0 (Entscheidungsgrenze)')
plt.title("Verteilung der μ-Werte im Testset")
plt.xlabel("μ-Wert (GLVQ)")
plt.ylabel("Anzahl der Testdatenpunkte")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

mu_array = np.array(mu_values)
# Berechnet den arithmetischen Mittelwert aller μ-Werte im NumPy-Array.
mean_mu = mu_array.mean()
safe_classified = (mu_array < 0).sum()
unsafe_classified = (mu_array >= 0).sum()

total = len(mu_array)

print(f"\n--- GLVQ Kostenanalyse (Testset) ---")
print(f" Durchschnittlicher μ-Wert: {mean_mu:.4f}")
print(f" Sicher klassifiziert (μ < 0): {safe_classified}/{total} = {safe_classified/total:.1%}")
# Gibt die Anzahl und den Prozentsatz der Testpunkte aus, deren μ-Wert >= 0 war.
print(f" Unsicher/falsch (μ ≥ 0): {unsafe_classified}/{total} = {unsafe_classified/total:.1%}")
# test git push