
# Um die App zu starten müssen 2 separate Konsolen verwendet werden (eine für Client und die andere für Server)
# Erstmal den Server starten, durch den Befehl:
# uvicorn server:app --reload

# =================================================================================
# SERVER-SEITIGE LOGIK
# =================================================================================
#
# Dieser Code implementiert die serverseitige Komponente als echten FastAPI-Webserver.
# Die Hauptaufgaben dieses Servers sind:
# 1. Einmaliges Trainieren eines GLVQ-Machine-Learning-Modells, um "Prototypen"
#    für die Klassen "gesund" und "krank" zu lernen. Diese Prototypen sind die
#    einzige Wissensbasis des Servers.
# 2. Bereitstellen von zwei API-Endpunkten:
#    - /assets: Gibt die initialen Daten (Scaler, Feature-Namen) zurück.
#    - /classify: Nimmt verschlüsselte Patientendaten an und gibt verschlüsselte
#                 Klassifikationsergebnisse zurück.
# 3. Diese Anfrage enthält homomorph verschlüsselte Patientendaten. Der Server
#    führt "blind" Berechnungen (Distanzmessungen) auf diesen verschlüsselten
#    Daten durch.
# 4. Der Server sendet die verschlüsselten Ergebnisse zurück an den Client, ohne
#    jemals die originalen Patientendaten oder das finale Klassifikationsergebnis
#    zu kennen.
#
#  Wichtig: Der Server läuft als eigenständiger Prozess und kommuniziert über HTTP.

#  Wichtig: Die Sicherheit wird dadurch gewährleistet, dass der Server nie Zugriff auf den
#  geheimen Schlüssel (Secret Key) des Clients hat.
#
# =================================================================================

# --- Bibliotheken importieren ---
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tenseal as ts
from sklearn.preprocessing import MinMaxScaler
from sklearn_lvq import GlvqModel
from sklearn.ensemble import IsolationForest
import base64 # Import für die Base64-Kodierung
import joblib #  Zum Serialisieren des Scalers
from io import BytesIO

# --- Modell-Training und Asset-Laden ---
# Diese Funktion wird einmalig beim Start des Servers ausgeführt
def get_server_assets():
    """
    Simuliert das Laden der serverseitigen Assets (trainiertes Modell).
    """
    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    # Pfad zum Datensatz 
    df = pd.read_csv("../heart_data_pretty.csv", sep='\s+')
    X = df.drop(columns=["target"]).copy()
    y_original = (df["target"] > 0).astype(int)
    # Feature-Namen extrahieren, damit der Client sie erhalten kann
    feature_names = X.columns.tolist()
    
    # Daten vorverarbeiten
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
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled_for_iso) # -1 für Ausreißer, 1 für Inlier

    # Indizes der als Ausreißer markierten Datenpunkte
    outlier_indices = np.where(predictions == -1)[0]
    print(f"Anzahl der erkannten Ausreißer: {len(outlier_indices)}")

    # --- AUSREISSERERKENNUNG Schritt 2: Entfernen der Ausreißer ---
    print("\n--- Entferne Ausreißer aus dem Datensatz ---")
    print(f"Originale Datenform (X): {X.shape}")
    # y- target
    print(f"Originale Datenform (y): {y_original.shape}")

    # Entferne die Ausreißer aus X und y
    X_clean = X.drop(X.index[outlier_indices])
    y_clean = y_original.drop(y_original.index[outlier_indices])
    
    print(f"Neue Datenform nach Außreiser-Bereinigung (X_clean): {X_clean.shape}")
    print(f"Neue Datenform nach Außreiser-Bereinigung (y_clean): {y_clean.shape}")

    ##################################################
    # --- Schritt 2: Modelltraining ---
    ##################################################

    # Ziel: alle Merkmale  auf eine ähnliche Skala (Bereich [-1,1]) zu bringen!
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Skaliere die BEREINIGTEN Daten 'X_clean'
    X_scaled = scaler.fit_transform(X_clean)
    # Verwende die BEREINIGTE Zielvariable 'y_clean'
    y_binary_np = y_clean.to_numpy()
    
    print("--- SERVER: Trainiere GLVQ-Modell... ---")
    server_model = GlvqModel(prototypes_per_class=1, beta=3, gtol=1e-5, random_state=42)
    server_model.fit(X_scaled, y_binary_np)
    
    prototypes = server_model.w_
    proto_labels = server_model.c_w_
    
    print("--- SERVER: Modell trainiert und Assets geladen. ---")
     # Gibt alle Assets zurück, die von anderen Teilen der Anwendung benötigt werden könnten. // prototypes werden vom Client NICHT abgerufen
    return prototypes, proto_labels, scaler, feature_names

# --- Globale Variable für die Assets ---
PROTOTYPES, PROTO_LABELS, SCALER, FEATURE_NAMES = get_server_assets()

# --- Server-Modell für die HTTP-Anfrage ---
class EncryptedDataRequest(BaseModel):
    # Die Daten werden jetzt als Base64-kodierte Strings erwartet
    serialized_encrypted_patient_vector: str
    serialized_public_ckks_context: str
    
# --- FastAPI App ---
app = FastAPI()


# Endpunkt zum Abrufen der initialen Assets
@app.get("/assets") # Endpunkt. /assets
def get_assets():
    """
    Stellt den fitten Scaler und die Feature-Namen bereit.
    Der Scaler wird serialisiert, um ihn über HTTP senden zu können.
    """
    print("--- SERVER: Sende initiale Assets an Client... ---")
    scaler_bytes = BytesIO()
    joblib.dump(SCALER, scaler_bytes)
    scaler_base64 = base64.b64encode(scaler_bytes.getvalue()).decode('utf-8')
    return {
        "scaler": scaler_base64,
        "feature_names": FEATURE_NAMES
    }

@app.post("/classify")
def process_encrypted_request(request: EncryptedDataRequest):
    """
    Diese Funktion ist die zentrale Logik des Servers, die auf HTTP-Anfragen des Clients reagiert.
    Sie nimmt serialisierte, verschlüsselte Daten entgegen, verarbeitet sie blind und
    gibt serialisierte, verschlüsselte Ergebnisse zurück.
    """
    # Dekodierung der Base64-Strings zurück in Bytes
    # Patienten-Vektor  und CKKS-Kontext vom Client
    serialized_vector_bytes = base64.b64decode(request.serialized_encrypted_patient_vector)
    serialized_context_bytes = base64.b64decode(request.serialized_public_ckks_context)
    
    # Rekonstruiere den öffentlichen CKKS-Kontext vom Client
    public_context = ts.context_from(serialized_context_bytes)
    
    # 2. Rekonstruiere den verschlüsselten Patienten-Vektor vom Client
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_vector_bytes)
    
    # Sicherheits- und Funktions-Check des erhaltenen Kontexts
    print("\n--- SERVER: Überprüfe den vom Client erhaltenen Kontext... ---")
    if public_context.has_public_key():
        print("-> STATUS: ✅ Der Kontext enthält die benötigten öffentlichen Schlüssel")
    else:
        print("-> STATUS: ❌ WARNUNG: Der Kontext enthält keine öffentlichen Schlüssel")
    
    if not public_context.has_secret_key():
        print("-> STATUS: ✅ Der Kontext enthält wie erwartet KEINEN geheimen Schlüssel.")
    else:
        print("-> STATUS: ❌ SICHERHEITSRISIKO: Der Kontext enthält fälschlicherweise einen geheimen Schlüssel!")
    print("----------------------------------------------------\n")


    # 3. Führe die homomorphe Distanzberechnung für jeden Prototyp durch
    encrypted_distances = []
    for p_vector in PROTOTYPES:
        enc_diff = encrypted_patient_vector - p_vector
        enc_squared_diff = enc_diff.pow(2)
        enc_distance = enc_squared_diff.sum()  # dies ist die Quadrierte Euklidische Distanz // Alle 13 Merkmale innerhalb des Vektors "enc_squared_diff" werden zu einem einzigen Wert addiert.
        encrypted_distances.append(enc_distance)
    
    # 4. Bereite die Ergebnisse für die Rücksendung vor (Serializieren von CKKS-Vektoren ist notwendig für Client/Server Kommunikation über das Netz)
    serialized_results = []
    for vec, label in zip(encrypted_distances, PROTO_LABELS):
        # Jedes serialisierte Ergebnis muss ebenfalls als Base64-kodierter String gespeichert werden
        serialized_vec = base64.b64encode(vec.serialize()).decode('utf-8')
        serialized_results.append([serialized_vec, int(label)])
        
    print("--- SERVER: Distanzberechnung abgeschlossen, sende Ergebnisse zurück. ---")
    return {"serialized_results": serialized_results}

# den Server in der Konsole starten:
# uvicorn server:app --reload