# Um die App zu starten müssen 2 separate Konsolen verwendet werden (eine für Client und die andere für Server)
# Erstmal den Server starten, durch den Befehl:
# uvicorn server:app --reload

# =================================================================================
# SERVER-SEITIGE LOGIK FÜR GMLVQ (homomorph mit CKKS)
# =================================================================================
# Dieser Code implementiert die serverseitige Komponente als echten FastAPI-Webserver
#  für eine Client-Server App, die Patientendaten-Klassifikation mit dem GMLVQ-Algorithmus durchführt.
# Die Hauptaufgaben dieses Servers sind:
# 1. Einmaliges Trainieren eines GMLVQ-Modells beim ersten Start und Bereitstellen
#    der notwendigen, nicht-geheimen "Assets" (Scaler, Feature-Namen).
# 2. Bereitstellen einer API-Funktion (`process_encrypted_request`), die
#    Anfragen vom Client entgegennimmt.
# 3. Empfangen eines homomorph verschlüsselten Patientendaten-Vektors vom Client.
# 4. Durchführung der GMLVQ-Distanzberechnung auf den verschlüsselten
#    Daten. Dies ist eine "blinde" Berechnung, da der Server die Daten nie entschlüsselt.
# 5. Zurücksenden der verschlüsselten Ergebnisse und der globalen Modell-Relevanzen an den Client.
#
# Der Server kennt zu keinem Zeitpunkt die geheimen Patientendaten oder das finale
# Klassifikationsergebnis. Die Sicherheit wird durch das CKKS-Schema gewährleistet.
# =================================================================================


# --- Bibliotheken importieren ---
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tenseal as ts
import os
import joblib
from io import BytesIO
import base64

from sklearn.preprocessing import MinMaxScaler
from sklearn_lvq import GmlvqModel
from sklearn.ensemble import IsolationForest


def get_server_assets():
    """
    Simuliert das Laden und Trainieren der serverseitigen Assets.
    Diese Funktion wird dank Caching nur einmal ausgeführt. In einer realen Anwendung
    würde man hier ein bereits fertig trainiertes Modell laden, anstatt es neu zu trainieren.
    Gibt die Prototypen, deren Labels, den Scaler, die Feature-Namen, die
    Lambda-Matrix und die berechneten Merkmals-Relevanzen zurück.
    """
    # --- Schritt 1: Daten laden und vorverarbeiten ---
    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    df = pd.read_csv("../heart_data_pretty.csv", sep='\s+')

    # Daten vorverarbeiten
    X = df.drop(columns=["target"]).copy()
    y = (df["target"] > 0).astype(int)
    feature_names = X.columns.tolist()

    X.replace('?', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    #---------------------
    # AUSREISSERERKENNUNG
    #---------------------

    # --- AUSREISSERERKENNUNG Schritt 1: Ausreißererkennung mit Isolation Forest (Multivariat) ---
    # Der Isolation Forest entscheidet WELCHE  Ausreißer entfernt werden sollten
    print("\n--- Wende Isolation Forest an ---")
    # Die Daten werden hier nur für den Zweck der Ausreißererkennung skaliert 
    scaler_for_iso = MinMaxScaler(feature_range=(-1, 1))
    X_scaled_for_iso = scaler_for_iso.fit_transform(X)

    # Isolation Forest initialisieren und anwenden
    # contamination legt den erwarteten Anteil der Ausreißer fest
    iso_forest = IsolationForest(contamination=0.08, random_state=42) # 8%
    predictions = iso_forest.fit_predict(X_scaled_for_iso) # -1 für Ausreißer, 1 für Inlier

    # Indizes der als Ausreißer markierten Datenpunkte
    outlier_indices = np.where(predictions == -1)[0]
    print(f"Anzahl der erkannten Ausreißer: {len(outlier_indices)}")

    # --- AUSREISSERERKENNUNG Schritt 2: Entfernen der Ausreißer ---
    print("\n--- Entferne Ausreißer aus dem Datensatz ---")
    print(f"Originale Datenform (X): {X.shape}")
    # y- target
    print(f"Originale Datenform (y): {y.shape}")

    # Entferne die Ausreißer aus X und y
    X_clean = X.drop(X.index[outlier_indices])
    y_clean = y.drop(y.index[outlier_indices])

    print(f"Neue Datenform nach Außreiser-Bereinigung (X_clean): {X_clean.shape}")
    print(f"Neue Datenform nach Außreiser-Bereinigung (y_clean): {y_clean.shape}")


    # --- Schritt 2: Daten skalieren ---
    scaler = MinMaxScaler(feature_range=(-1, 1))   # Alle Features in einem kleinen Wertebereich 
    X_scaled = scaler.fit_transform(X_clean)
    y_binary_np = y_clean.to_numpy()

    # --- Schritt 3: GMLVQ-Modelltraining ---
    print("--- SERVER: Trainiere GMLVQ-Modell... ---")
    gmlvq = GmlvqModel(prototypes_per_class=2, regularization=0.35, gtol=1e-5, random_state=42) # Params wurden durch Kreuzvalidierung ausgewählt
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


# --- Globale Variablen für die Assets ---
PROTOTYPES, PROTO_LABELS, SCALER, FEATURE_NAMES, OMEGA, RELEVANCES = get_server_assets()

# --- Server-Modell für die Anfrage ---
class EncryptedDataRequest(BaseModel):
    # Die Daten werden als Base64-kodierte Strings erwartet
    serialized_encrypted_patient_vector: str
    serialized_public_ckks_context: str
    
# --- FastAPI App ---
app = FastAPI()

# Endpunkt zum Abrufen der initialen Assets
@app.get("/assets_gmlvq") # Endpunkt: /assets_gmlvq
def get_assets_gmlvq():
    """
    Stellt den fitten Scaler und die Feature-Namen bereit (für Client HTTP-Anfrage).
    """
    print("--- SERVER: Sende initiale Assets an Client... ---")
    scaler_bytes = BytesIO()
    joblib.dump(SCALER, scaler_bytes)
    scaler_base64 = base64.b64encode(scaler_bytes.getvalue()).decode('utf-8')
    return {
        "scaler": scaler_base64,
        "feature_names": FEATURE_NAMES
    }

@app.post("/classify_gmlvq") # Endpunkt: /classify_gmlvq
def process_encrypted_request_api(request: EncryptedDataRequest):
    """
    API-Endpunkt zur Verarbeitung verschlüsselter Daten.
    """
    # Dekodierung der Base64-Strings zurück in Bytes (Patientendaten und CKKS-Context)
    serialized_vector_bytes = base64.b64decode(request.serialized_encrypted_patient_vector)
    serialized_context_bytes = base64.b64decode(request.serialized_public_ckks_context)
    
    # Rekonstruiere den öffentlichen CKKS-Kontext
    public_context = ts.context_from(serialized_context_bytes)
    

    # Sicherheits-Check: Stelle sicher, dass der Kontext öffentlich ist und keinen geheimen Schlüssel enthält.
    print("\n--- SERVER: Kontextüberprüfung ---")
    print("-> Public Key vorhanden:", public_context.has_public_key()) # Output: Ja
    print("-> Relinearisierungsschlüssel vorhanden :", public_context.has_relin_keys()) # Output: Ja
    print("-> Secret Key vorhanden (sollte False sein):", public_context.has_secret_key()) # Output: Nein
    print("----------------------------------------------------")

    # Rekonstruiere den verschlüsselten Vektor
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_vector_bytes)
    
    # --- A) Homomorphe Projektion des Patientendatenvektors: Enc(Ωξ) ---
    encrypted_embedded = []
    for omega_row in OMEGA:
        dot_product = encrypted_patient_vector.dot(omega_row.tolist())
        encrypted_embedded.append(dot_product)

    # --- B) Berechnung der Distanz zu jedem Prototyp im projizierten Raum ---
    encrypted_distances = []
    for i, proto in enumerate(PROTOTYPES):
        embedded_proto = OMEGA @ proto
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
    for vec, label in zip(encrypted_distances, PROTO_LABELS):
        # Jedes serialisierte Ergebnis muss ebenfalls als Base64-kodierter String gespeichert werden
        serialized_vec = base64.b64encode(vec.serialize()).decode('utf-8')
        serialized_results.append([serialized_vec, int(label)])

    print("--- SERVER: Distanzberechnung abgeschlossen. ---")
    
    # Gib die verschlüsselten Distanzen UND die Klartext-Relevanzen zurück.
    return {"serialized_results": serialized_results, "relevances": RELEVANCES.tolist()}

# Starte  den Server in der Konsole:
# uvicorn server:app --reload