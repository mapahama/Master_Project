# .\venv311\Scripts\activate

# CKKS-Implementierung Lösung 1: client-server-simulation
# Validierung, ob die euklidischen Distanzen bei Verschlüsselung → Entschlüsselung erhalten bleiben
# Prüfung, ob durch den ckks-workflow (verschlüsseln → entschlüsseln) Datenverluste auftreten

# Ergebnis: verschlüsselte und unverschlüsselte Distanzen sind identisch

# === Bibliotheken importieren ===
from ucimlrepo import fetch_ucirepo # Heart Disease Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GlvqModel
from scipy.spatial.distance import euclidean

import pandas as pd
import numpy as np

# TenSEAL für homomorphe Verschlüsselung (CKKS)
import tenseal as ts

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
X = X.apply(pd.to_numeric, errors='coerce')

# Fülle alle verbleibenden `np.nan`-Werte (fehlende Werte) mit dem Median der jeweiligen Spalte
X.fillna(X.median(), inplace=True)

# Dieser Scaler standardisiert Merkmale, indem er den Mittelwert jeder Spalte abzieht
# und dann durch die Standardabweichung teilt. Ergebnis: Mittelwert 0, Standardabweichung 1.
scaler = StandardScaler()

# Wende die Skalierung auf den gesamten Datensatz an
X_scaled = scaler.fit_transform(X)
y_binary_np = y_binary.to_numpy()
print("Vorverarbeitung abgeschlossen.")


# === === === === === === === === === === === === ===
# ===  4. Aufteilung in Trainings- und Testdaten   ===
# === === === === === === === === === === === === ===
# Teilen die Daten in 95% Trainingsdaten und 5% Testdaten.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary_np, test_size=0.05, random_state=42, stratify=y_binary_np
)
print(f"\nDaten aufgeteilt: {len(X_train)} Trainingspunkte, {len(X_test)} Testpunkte.")


# === === === === === === === === === === ===
# ===  5. SERVER-SEITE: Modell trainieren  ===
# === === === === === === === === === === ===
print("\n\n==============================================")
print("###  SERVER-AKTION: Einmaliges Modell-Training ###")
print("==============================================")

# Der Server trainiert sein GLVQ-Modell EINMAL mit den Trainingsdaten.
print("-> Server trainiert das GLVQ-Modell...")
server_model = GlvqModel(
    prototypes_per_class=3,
    beta=2,
    random_state=42
)
server_model.fit(X_train, y_train)

# Der Server extrahiert die gelernten Prototypen. Dies ist die Wissensbasis
prototypes = server_model.w_
proto_labels = server_model.c_w_
print(f"-> Server-Modell ist trainiert und hat {len(prototypes)} Prototypen gelernt.")


# === === === === === === === === === === === === ===
# ===  6. CLIENT-SEITE: CKKS-Kontext erstellen     ===
# === === === === === === === === === === === === ===
print("\n\n==============================================")
print("###  CLIENT-AKTION: CKKS-Kontext und Schlüssel generieren  ###")
print("==============================================")
# Der Client erstellt den Kontext und die Schlüssel nur einmal.
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
secret_key = context.secret_key()
print("-> Client hat Kontext und Schlüsselpaar generiert.")


# ####################################################################################
# ###  BEGINN DER SCHLEIFE ZUM VERGLEICH VON VERSCHLÜSSELTER VS. UNVERSCHLÜSSELTER KLASSIFIKATION ###
# ####################################################################################

# Initialisieren einen Zähler, um die Übereinstimmungen zu verfolgen.
classification_matches = 0

# Schleife über jeden Patienten im 5%-Testset
for patient_index in range(len(X_test)):
    print(f"\n\n######################################################################")
    print(f"###   VERARBEITE PATIENT #{patient_index} AUS DEM TEST-SET")
    print(f"######################################################################")

    user_patient_vector = X_test[patient_index]
    user_patient_true_label = y_test[patient_index]
    true_label_text = 'KRANK (1)' if user_patient_true_label == 1 else 'GESUND (0)'
    print(f"Wahre Klasse des Patienten: {true_label_text}")

    # ------------------------------------------------------------------
    # A) VERSCHLÜSSELTE KLASSIFIKATION (CLIENT-SERVER-INTERAKTION)
    # ------------------------------------------------------------------
    print("\n--- A) Starte verschlüsselte Klassifikation ---")
    # Client verschlüsselt den Vektor
    encrypted_patient_vector = ts.ckks_vector(context, user_patient_vector)

    # Server berechnet homomorph die Distanzen
    encrypted_distances = []
    for p_vector in prototypes:
        enc_diff = encrypted_patient_vector - p_vector
        enc_squared_diff = enc_diff.pow_(2)
        enc_distance = enc_squared_diff.sum()
        encrypted_distances.append(enc_distance)

    # Client entschlüsselt die Distanzen und trifft die Entscheidung
    decrypted_distances = [d.decrypt(secret_key)[0] for d in encrypted_distances]
    winning_proto_encrypted_idx = np.argmin(decrypted_distances)
    pred_class_encrypted = proto_labels[winning_proto_encrypted_idx]


    # ------------------------------------------------------------------
    # B) UNVERSCHLÜSSELTE KLASSIFIKATION (KONTROLLEXPERIMENT)
    # ------------------------------------------------------------------
    print("\n--- B) Starte unverschlüsselte Klassifikation (Kontrolle) ---")
    # Direkte Berechnung der euklidischen Distanzen mit den Original-Daten
    unencrypted_distances = [euclidean(user_patient_vector, p) for p in prototypes]
    # Hier wird die *echte* Distanz berechnet, nicht die quadrierte, für einen faireren Vergleich.
    # Die quadrierte Distanz würde zum selben Index führen.
    winning_proto_unencrypted_idx = np.argmin(unencrypted_distances)
    pred_class_unencrypted = proto_labels[winning_proto_unencrypted_idx]


    # ------------------------------------------------------------------
    # C) VERGLEICH DER ERGEBNISSE
    # ------------------------------------------------------------------
    print("\n--- C) Ergebnisvergleich für Patient #{} ---".format(patient_index))
    print("      | {:^25} | {:^25}".format("Verschlüsselt (CKKS)", "Unverschlüsselt (Klartext)"))
    print("----------------------------------------------------------------")
    # Wir geben hier die Wurzel der entschlüsselten Distanz aus, um sie mit der echten Euklidischen Distanz vergleichbar zu machen.
    print("Min. Distanz  | {:^25.4f} | {:^25.4f}".format(np.sqrt(decrypted_distances[winning_proto_encrypted_idx]), unencrypted_distances[winning_proto_unencrypted_idx]))
    print("Vorhersage    | {:^25} | {:^25}".format('KRANK' if pred_class_encrypted == 1 else 'GESUND', 'KRANK' if pred_class_unencrypted == 1 else 'GESUND'))
    print("----------------------------------------------------------------")


    if pred_class_encrypted == pred_class_unencrypted:
        print("✅ Klassifikationen stimmen überein!")
        classification_matches += 1
    else:
        print("❌ ACHTUNG: Klassifikationen weichen ab!")


# === === === === === === === === === === === === ===
# ===  7. FINALE AUSWERTUNG                       ===
# === === === === === === === === === === === === ===
print("\n\n==============================================")
print("###  FINALE ZUSAMMENFASSUNG")
print("==============================================")
print(f"Insgesamt wurden {len(X_test)} Patienten aus dem Test-Set klassifiziert.")
print(f"In {classification_matches} von {len(X_test)} Fällen stimmte die verschlüsselte Klassifikation mit der unverschlüsselten überein.")
accuracy = (classification_matches / len(X_test)) * 100
print(f"-> Übereinstimmungs-Genauigkeit: {accuracy:.2f}%")