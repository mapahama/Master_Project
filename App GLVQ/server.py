
# =================================================================================
# SERVER-SEITIGE LOGIK 
# =================================================================================
#
# Dieser Code simuliert die serverseitige Komponente einer Client-Server-Anwendung.
# Die Hauptaufgaben dieses Servers sind:
# 1. Einmaliges Trainieren eines GLVQ-Machine-Learning-Modells, um "Prototypen"
#    für die Klassen "gesund" und "krank" zu lernen. Diese Prototypen sind die
#    einzige Wissensbasis des Servers.
# 2. Bereitstellen einer Funktion (`process_encrypted_request`), die eine Anfrage
#    von einem Client entgegennimmt.
# 3. Diese Anfrage enthält homomorph verschlüsselte Patientendaten. Der Server
#    führt "blind" Berechnungen (Distanzmessungen) auf diesen verschlüsselten
#    Daten durch.
# 4. Der Server sendet die verschlüsselten Ergebnisse zurück an den Client, ohne
#    jemals die originalen Patientendaten oder das finale Klassifikationsergebnis
#    zu kennen.
#
# Die Sicherheit wird dadurch gewährleistet, dass der Server nie Zugriff auf den
# geheimen Schlüssel (Secret Key) des Clients hat.
# =================================================================================


# --- Bibliotheken importieren ---
import streamlit as st # Wird hier nur für die Caching-Funktion @st.cache_resource genutzt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GlvqModel
import tenseal as ts

@st.cache_resource
def get_server_assets():
    """
    Simuliert das Laden der serverseitigen Assets (trainiertes Modell).
    Diese Funktion wird nur einmal ausgeführt und die Ergebnisse werden zwischengespeichert,
    um das teure Modelltraining nicht bei jeder Anfrage wiederholen zu müssen.
    In einer echten Anwendung würde man hier ein bereits fertig trainiertes Modell
    von der Festplatte laden.
    """
    # --- Schritt 1: Daten laden und vorverarbeiten ---
    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    df = pd.read_csv("heart_data_pretty.csv", sep='\s+')
    X = df.drop(columns=["target"]).copy()
    y = (df["target"] > 0).astype(int)
    # Feature-Namen extrahieren, damit der Client sie erhalten kann
    feature_names = X.columns.tolist()
    
    X.replace('?', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    # --- Schritt 2: Modelltraining ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binary_np = y.to_numpy()

    print("--- SERVER: Trainiere GLVQ-Modell... ---")
    server_model = GlvqModel(prototypes_per_class=3, beta=2, random_state=42)
    server_model.fit(X_scaled, y_binary_np)
    
    prototypes = server_model.w_
    proto_labels = server_model.c_w_
    
    print("--- SERVER: Modell trainiert und Assets geladen. ---")
    # Gibt alle Assets zurück, die von anderen Teilen der Anwendung benötigt werden könnten. // prototypes werden vom Client NICHT abgerufen
    return prototypes, proto_labels, scaler, feature_names

def process_encrypted_request(serialized_encrypted_patient_vector, serialized_public_ckks_context):
    """
    Simuliert einen Server-API-Endpunkt.
    Diese Funktion ist die zentrale Logik des Servers, die auf Anfragen des Clients reagiert.
    Sie nimmt serialisierte, verschlüsselte Daten entgegen, verarbeitet sie blind und
    gibt serialisierte, verschlüsselte Ergebnisse zurück.
    """
    # 1. Lade die Server-Assets (Prototypen und deren Labels)
    # Die Funktion gibt 4 Werte zurück. Wir benötigen hier nur die ersten beiden
    # und ignorieren den Rest mit dem Unterstrich-Platzhalter "_".
    prototypes, proto_labels, _, _ = get_server_assets()

    # 2. Rekonstruiere den öffentlichen CKKS-Kontext aus den vom Client gesendeten Daten
    public_context = ts.context_from(serialized_public_ckks_context)
    
    # Sicherheits- und Funktions-Check des erhaltenen Kontexts
    print("\n--- SERVER: Überprüfe den vom Client erhaltenen Kontext... ---")
    if public_context.has_public_key():
        print("-> STATUS: ✅ Der Kontext enthält die benötigten öffentlichen Schlüssel (Galois-Keys)")
    else:
        print("-> STATUS: ❌ WARNUNG: Der Kontext enthält keine öffentlichen Schlüssel")
    
    if not public_context.has_secret_key():
        print("-> STATUS: ✅ Der Kontext enthält wie erwartet KEINEN geheimen Schlüssel.")
    else:
        print("-> STATUS: ❌ SICHERHEITSRISIKO: Der Kontext enthält fälschlicherweise einen geheimen Schlüssel!")
    print("----------------------------------------------------\n")

    # 3. Rekonstruiere den verschlüsselten Vektor aus den vom Client gesendeten Daten
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_encrypted_patient_vector)
    
    # 4. Führe die homomorphe Distanzberechnung für jeden Prototyp durch
    encrypted_distances = []
    for p_vector in prototypes:
        enc_diff = encrypted_patient_vector - p_vector
        enc_squared_diff = enc_diff.pow(2)
        enc_distance = enc_squared_diff.sum()
        encrypted_distances.append(enc_distance)

        # Debug-Ausgaben, um den Prozess zu verfolgen. TODO: In Produktion zu entfernen.
        print("\n--- encrypted_patient_vector:", encrypted_patient_vector)
        print("\n--- prototype vector:", p_vector)
        print("\n--- Aktuelle quadrierte Differenz:", enc_squared_diff)
        print("\n--- Aktuelle quadrierte Distanz Datenpunkt/Vektor:", enc_distance)
        print("\n-----------------")

    # 5. Bereite die Ergebnisse für die Rücksendung an den Client vor. (Serializieren von CKKS-Vektoren ist notwendig für Client/Server Kommunikation über das Netz)
    serialized_results = []
    for vec, label in zip(encrypted_distances, proto_labels):
        serialized_results.append((vec.serialize(), label))


    print("--- SERVER: Distanzberechnung abgeschlossen, sende Ergebnisse zurück. ---")
    return serialized_results