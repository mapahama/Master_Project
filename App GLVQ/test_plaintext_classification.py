# ==================================================================================================
# Dieses Skript dient als Referenzprüfung für die Client-Server-Anwendung mit CKKS-Verschlüsselung.
#
# Ziel:
# -----
# In der eigentlichen Client-Server-App werden Patientendaten auf dem Client mit CKKS verschlüsselt,
# an den Server gesendet, und dort erfolgt die Distanzberechnung zu den GLVQ-Prototypen.
# Die Klassifikation (gesund/krank) findet anschließend auf dem Client statt,
# indem die verschlüsselten Distanzen entschlüsselt und verglichen werden .
#
# Dieses Skript überprüft die Genauigkeit dieser Architektur, indem dieselben Patientendaten
# in der Konsole eingegeben und **auf dem Server im Klartext verarbeitet** werden.
#
# Ablauf:
# -------
# - Die eingegebenen Patientendaten (in die Konsole) werden skaliert.
# - Die quadrierten euklidischen Distanzen zu den GLVQ-Prototypen werden berechnet.
# - Die Vorhersage erfolgt durch Auswahl der geringsten Distanz.
#
# Ziel ist es zu zeigen, dass die Distanzen (und damit auch die Klassifikationsergebnisse),
# die in der verschlüsselten Version berechnet und entschlüsselt werden,
# nahezu identisch mit denen der unverschlüsselten Referenz sind.
#
# Ergebnis (vorläufige Beobachtung): Die Distanzen sind in über 99 % der Fälle identisch.
# ==================================================================================================


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GlvqModel
from scipy.spatial.distance import sqeuclidean

# ==============================================================================
# 1. SETUP: MODELL UND DATEN VORBEREITEN
# ==============================================================================
print("="*50)
print("### SERVER PROOF OF CONCEPT - SETUP ###")
print("="*50)

print("-> Lade Heart Disease Datensatz...")
try:
    df = pd.read_csv("heart_data_pretty.csv", sep='\s+')
except FileNotFoundError:
    print("\nFEHLER: Die Datensatz-Datei `heart_data_pretty.csv` wurde nicht gefunden.")
    exit()

X = df.drop(columns=["target"]).copy()
y = df["target"].copy()
feature_names = X.columns.tolist()
y_binary = (y > 0).astype(int)

print("-> Starte Vorverarbeitung...")
X.replace('?', np.nan, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_binary_np = y_binary.to_numpy()

print("-> Server trainiert das GLVQ-Modell (dies geschieht nur einmal)...")
server_model = GlvqModel(prototypes_per_class=3, beta=2, random_state=42)
server_model.fit(X_scaled, y_binary_np)

prototypes = server_model.w_
proto_labels = server_model.c_w_
print(f"-> Setup abgeschlossen. Modell ist trainiert und hat {len(prototypes)} Prototypen.")
print("="*50 + "\n")


# ==============================================================================
# 2. INTERAKTIVE KLASSIFIKATIONSSCHLEIFE
# ==============================================================================

while True:
    print("\n### GEBEN SIE NEUE PATIENTENDATEN EIN ###")
    print("Geben Sie die 13 Merkmalswerte durch Leerzeichen getrennt ein.")
    print("Beispiel: 54.4 0.68 3.16 131.6 246.7 0.14 0.99 149.6 0.32 1.0 1.6 0.67 4.73")
    print("Geben Sie 'exit' oder 'q' ein, um das Programm zu beenden.")
    
    user_input_str = input("> ")

    if user_input_str.lower() in ['exit', 'q']:
        print("Programm wird beendet.")
        break

    # --- 1. Eingabe verarbeiten ---
    try:
        patient_vector_str = user_input_str.split()
        if len(patient_vector_str) != 13:
            raise ValueError(f"Falsche Anzahl an Werten. Erwartet: 13, Bekommen: {len(patient_vector_str)}")
        
        patient_vector_list = [float(v) for v in patient_vector_str]

    except ValueError as e:
        print(f"\nFEHLER BEI DER EINGABE: {e}")
        print("Bitte versuchen Sie es erneut und achten Sie auf das korrekte Format.\n")
        continue

    # --- 2. Daten für das Modell vorbereiten (skalieren) ---
    patient_df = pd.DataFrame([patient_vector_list], columns=feature_names)
    scaled_patient_vector = scaler.transform(patient_df)

    # --- 3. Unverschlüsselte Distanzen berechnen ---
    #  Berechne die *quadrierte* euklidische Distanz
    unencrypted_distances = [sqeuclidean(scaled_patient_vector[0], p) for p in prototypes]

    # --- 4. Kürzeste Distanz und Klassifikation finden ---
    min_dist_idx = np.argmin(unencrypted_distances)
    predicted_class = proto_labels[min_dist_idx]

    # --- 5. Ergebnis anzeigen ---
    print("\n--- ERGEBNIS DER SERVERSEITIGEN UNVERSCHLÜSSELTEN KLASSIFIKATION ---")
    print("-" * 70)
    print(f"{'Prototyp #':<15} | {'Prototyp-Klasse':<20} | {'Quadrierte Eukl. Distanz':<28}")
    print("-" * 70)
    
    for i, (dist, label) in enumerate(zip(unencrypted_distances, proto_labels)):
        class_text = 'GESUND' if label == 0 else 'KRANK'
        highlight = "   <-- KÜRZESTE DISTANZ" if i == min_dist_idx else ""
        print(f"{i:<15} | {class_text:<20} | {dist:<28.4f}{highlight}")
    
    print("-" * 70)
    
    final_class_text = "KRANK" if predicted_class == 1 else "GESUND"
    print(f"\nFINALE KLASSIFIKATION: Der Patient wird als **{final_class_text}** eingestuft.\n")
    print("="*70)

    #   test_plaintext_classification.py