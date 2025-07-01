# =================================================================================
# SERVER-SEITIGE LOGIK FÜR GMLVQ (homomorph mit CKKS)
# =================================================================================
# Dieser Code simuliert die serverseitige Komponente einer Client-Server-Anwendung
# für eine  Klassifikation mit dem GMLVQ-Algorithmus.
#
# Die Hauptaufgaben dieses Servers sind:
# 1. Einmaliges Trainieren eines GMLVQ-Modells beim ersten Start und Bereitstellen
#    der notwendigen, nicht-geheimen "Assets" (Scaler, Feature-Namen).
# 2. Bereitstellen einer API-ähnlichen Funktion (`process_encrypted_request`), die
#    Anfragen vom Client entgegennimmt.
# 3. Empfangen eines homomorph verschlüsselten Patientendaten-Vektors vom Client.
# 4. Durchführung der  GMLVQ-Distanzberechnung auf den verschlüsselten
#    Daten. Dies ist eine "blinde" Berechnung, da der Server die Daten nie entschlüsselt.
# 5. Zurücksenden der verschlüsselten Ergebnisse und der globalen Modell-Relevanzen an den Client.
#
# Der Server kennt zu keinem Zeitpunkt die geheimen Patientendaten oder das finale
# Klassifikationsergebnis. Die Sicherheit wird durch das CKKS-Schema gewährleistet.
# =================================================================================


# --- Bibliotheken importieren ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GmlvqModel
import tenseal as ts


@st.cache_resource
def get_server_assets():
    """
    Simuliert das Laden und Trainieren der serverseitigen Assets.
    Diese Funktion wird dank Caching nur einmal ausgeführt. In einer realen Anwendung
    würde man hier ein bereits fertig trainiertes Modell laden, anstatt es neu zu trainieren.
    Gibt die Prototypen, deren Labels, den Scaler, die Feature-Namen, die
    Omega-Matrix und die berechneten Merkmals-Relevanzen zurück.
    """
    # --- Schritt 1: Daten laden und vorverarbeiten ---
    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    df = pd.read_csv("../heart_data_pretty.csv", sep='\s+')
    X = df.drop(columns=["target"]).copy()
    y = (df["target"] > 0).astype(int)
    feature_names = X.columns.tolist()

    X.replace('?', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    # --- Schritt 2: Daten skalieren ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binary_np = y.to_numpy()

    # --- Schritt 3: GMLVQ-Modelltraining ---
    print("--- SERVER: Trainiere GMLVQ-Modell... ---")
    gmlvq = GmlvqModel(prototypes_per_class=3, regularization=0.35, random_state=42) # Params wurden durch Kreuzvalidierung ausgewählt
    gmlvq.fit(X_scaled, y_binary_np)

    # --- Schritt 4: Relevante Modell-Parameter extrahieren ---
    prototypes = gmlvq.w_
    proto_labels = gmlvq.c_w_
    omega = gmlvq.omega_

    # Berechne die Merkmals-Relevanzen aus der Omega-Matrix   
    # Die Relevanzmatrix Lambda (Λ) ist definiert als ΩT * Ω.
    # Die Diagonale von Lambda gibt die Wichtigkeit jedes einzelnen Merkmals an. 
    lambda_matrix = omega.T @ omega
    relevances = np.diag(lambda_matrix) #  (wird als Säulendiagramm im UI angezeigt - Erklärbarkeit)

    print("--- SERVER: GMLVQ-Assets und Relevanzen geladen. ---")
    return prototypes, proto_labels, scaler, feature_names, omega, relevances


def process_encrypted_request(serialized_vector, serialized_public_context):
    """
    Nimmt vom Client den verschlüsselten Patientenvektor und den öffentlichen CKKS -Kontext
    entgegen, wendet homomorph die Omega-Transformation an und berechnet die Distanzen
    zu den GMLVQ-Prototypen. Rückgabe: Liste verschlüsselter Distanzen, Klassenlabel
    und die globalen Merkmals-Relevanzen.
    """
    # Lade die Server-Assets (Modell-Parameter) aus dem Cache.
    # Empfange auch die 'relevances' (Erklärbarkeit im UI)
    prototypes, proto_labels, _, _, omega, relevances = get_server_assets()

    # Rekonstruiere den öffentlichen CKKS-Kontext aus den vom Client gesendeten Daten.
    public_context = ts.context_from(serialized_public_context)

    # Sicherheits-Check: Stelle sicher, dass der Kontext öffentlich ist und keinen geheimen Schlüssel enthält.
    print("\n--- SERVER: Kontextüberprüfung ---")
    print("-> Public Key vorhanden:", public_context.has_public_key())
    print("-> Secret Key vorhanden (sollte False sein):", public_context.has_secret_key())
    print("----------------------------------------------------")

    # Rekonstruiere den verschlüsselten (serializiert) Vektor aus den vom Client gesendeten Daten.
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_vector)

    # --- A) Homomorphe Projektion des Patientendatenvektors: Enc(Ωξ) ---
    encrypted_embedded = []
    for omega_row in omega:
        dot_product = encrypted_patient_vector.dot(omega_row.tolist())
        encrypted_embedded.append(dot_product)

    # --- B) Berechnung der Distanz zu jedem Prototyp im projizierten Raum ---
    encrypted_distances = []
    for i, proto in enumerate(prototypes):
        embedded_proto = omega @ proto
        diff_enc = []
        for enc_val, p_val in zip(encrypted_embedded, embedded_proto):
            diff_enc.append(enc_val - p_val)
        sq_diff = [x.pow(2) for x in diff_enc]

        enc_distance = sq_diff[0]
        for j in range(1, len(sq_diff)):
            enc_distance += sq_diff[j]
        encrypted_distances.append(enc_distance)

    # --- Ergebnis für den Rückversand an den Client vorbereiten ---
    serialized_results = []
    for vec, label in zip(encrypted_distances, proto_labels):
        serialized_results.append((vec.serialize(), label))

    print("--- SERVER: Distanzberechnung abgeschlossen. ---")
    
    # Gib die verschlüsselten Distanzen UND die Klartext-Relevanzen zurück.
    # .tolist() wandelt das NumPy-Array in eine normale Python-Liste um, was für APIs üblich ist.
    return serialized_results, relevances.tolist()