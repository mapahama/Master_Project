# Tschebyscheff Polynome - Server


# Um die App zu starten müssen 2 separate Konsolen verwendet werden (eine für Client und die andere für Server)
# Erstmal den Server starten, durch den Befehl:
#  .\venv311\Scripts\activate
# uvicorn server:app --reload

# =================================================================================
# SERVER-SEITIGE LOGIK für GLVQ (min. Distanz Bestimmung mit Polynom-Approximation - Tschebyscheff Polynome)
# =================================================================================

# =================================================================================
# Der Server führt die komplette homomorphe Klassifikation (blind) durch.
# Min. Distanz und Label werden durch Polynomapproximation mit Tschebyscheff Polynomen bestimmt
# Der Server empfängt einen verschlüsselten Vektor (Patientendaten) vom Client und gibt den verschlüsselten
# Gewinner (Label und Distanz) zurück.
# ================================================================================

# --- Bibliotheken importieren ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tenseal as ts
import os
import joblib
from io import BytesIO
import base64
from typing import List

from sklearn.preprocessing import MinMaxScaler
from sklearn_lvq import GlvqModel
from sklearn.ensemble import IsolationForest

# Variable für den Dateipfad des CKKS-Kontexts
CONTEXT_FILE_PATH = "ckks_context_glvq.bin"

# Variable für Assets und CKKS-Cache ---
ckks_context_cache = {}

# --- Modell-Training und Asset-Laden ---
def get_server_assets():
    """
    Simuliert das Laden der serverseitigen Assets (ein bereits trainiertes Modell).
    """
    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    df = pd.read_csv("../heart_data_pretty.csv", sep='\s+')
    X_full = df.drop(columns=["target"]).copy()
    X = X_full.iloc[:, :13].copy() # Alle  13 Features verwenden
    y_original = (df["target"] > 0).astype(int)
    feature_names = X.columns.tolist()
    
    # Daten bereinigen
    X.replace('?', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    print("\n--- Wende Isolation Forest an ---")
    scaler_for_iso = MinMaxScaler(feature_range=(-1, 1))
    X_scaled_for_iso = scaler_for_iso.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled_for_iso)
    outlier_indices = np.where(predictions == -1)[0]
    print(f"Anzahl der erkannten Ausreißer: {len(outlier_indices)}")

    print("\n--- Entferne Ausreißer aus dem Datensatz ---")
    X_clean = X.drop(X.index[outlier_indices])
    y_clean = y_original.drop(y_original.index[outlier_indices])
    
    # Daten in [-1,1] Bereich transformieren
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_clean)
    y_binary_np = y_clean.to_numpy()
    
    print("--- SERVER: Trainiere GLVQ-Modell... ---")
    server_model = GlvqModel(prototypes_per_class=1, beta=3, gtol=1e-5, random_state=42) # Bestimmt durch Cross Validation
    server_model.fit(X_scaled, y_binary_np)
    
    prototypes = server_model.w_
    proto_labels = server_model.c_w_
    
    print("--- SERVER: Modell trainiert und Assets geladen. ---")
    return prototypes, proto_labels, scaler, feature_names

# Wenn CKKS Kontext vorhanden - Funktion zum Laden des CKKS-Kontexts aus einer lokalen Datei
def load_ckks_context_from_file():
    if not os.path.exists(CONTEXT_FILE_PATH):
        raise FileNotFoundError(f"FEHLER: CKKS-Kontext-Datei '{CONTEXT_FILE_PATH}' nicht gefunden. Bitte starten Sie zuerst den Client, um die Datei zu erstellen.")
    
    with open(CONTEXT_FILE_PATH, "rb") as f:
        context_bytes = f.read()
    
    context = ts.context_from(context_bytes)
    return context

# Variablen für die Assets ---
PROTOTYPES, PROTO_LABELS, SCALER, FEATURE_NAMES = get_server_assets()

# --- Initialisiere den Kontext direkt bei Server-Start und speichere ihn im Cache ---
print(f"--- SERVER: Versuche CKKS-Kontext aus '{CONTEXT_FILE_PATH}' zu laden und zu cachen. ---")
try:
    ckks_context_cache["public_context"] = load_ckks_context_from_file()
    print("--- SERVER: CKKS-Kontext erfolgreich geladen. ---")
except FileNotFoundError as e:
    print(str(e))

# =================================================================================
# Polynom-Approximation und Homomorphe Helper-Funktionen
# =================================================================================
CHEBYSHEV_DEGREE = 10  # !!! entscheidend für die Präzision der homomorphen Berechnungen
MAX_EXPECTED_SQUARED_DISTANCE = 30.0 # TODO  dynamisch machen
x_vals = np.linspace(-1, 1, 2000)
sign_vals = np.sign(x_vals)
cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_vals, sign_vals, CHEBYSHEV_DEGREE)
monomial_coeffs = cheb_poly.convert(kind=np.polynomial.Polynomial).coef
print("Monomial-Koeffizienten für sgn(x): ", monomial_coeffs)

def homomorphic_sgn(a_minus_b):
    """
    Wendet die Polynom-Approximation der Signum-Funktion auf einen
    verschlüsselten Vektor an.
    Args:
        a_minus_b (ts.CKKSVector): Ein verschlüsselter Vektor, dessen Vorzeichen
                                   bestimmt werden soll.
    Returns:
        ts.CKKSVector: Das verschlüsselte Ergebnis der Polynom-Auswertung,
                       das entweder nahe bei +1 oder -1 liegt.
    """
    coeffs = monomial_coeffs
    return a_minus_b.polyval(coeffs)

def homomorphic_min_with_label(enc_val_a, enc_label_a, enc_val_b, enc_label_b):
    """
    Vergleicht zwei verschlüsselte Werte (a und b) und gibt den kleineren Wert
    sowie das zugehörige Label zurück, ohne die Werte zu entschlüsseln.
    Nutzt die Formel: min(a, b) = 0.5 * [(a+b) - sgn(a-b)*(a-b)]
    """
    val_a_minus_b = enc_val_a - enc_val_b
    sgn_a_minus_b = homomorphic_sgn(val_a_minus_b)
    val_a_plus_b = enc_val_a + enc_val_b
    min_val = (val_a_plus_b - (sgn_a_minus_b * val_a_minus_b)) * 0.5

    # Die gleiche Logik wird auf die Labels angewendet, um das korrekte Label auszuwählen.
    label_a_plus_b = enc_label_a + enc_label_b
    label_a_minus_b = enc_label_a - enc_label_b
    min_label = (label_a_plus_b - (sgn_a_minus_b * label_a_minus_b)) * 0.5
    return min_val, min_label

def find_min_and_label_tournament(encrypted_pairs):
    """
    Führt ein "Turnier" durch, um das Paar mit dem minimalen Wert aus einer
    Liste von verschlüsselten (Wert, Label)-Paaren zu finden.
    In jeder Runde werden benachbarte Paare verglichen, bis nur noch der
    Gesamtsieger übrig ist.
    """
    current_level = encrypted_pairs
    while len(current_level) > 1:
        next_level = []
        # Vergleiche Paare (1 vs 2, 3 vs 4, etc.)
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                val1, label1 = current_level[i]
                val2, label2 = current_level[i + 1]
                winner_val, winner_label = homomorphic_min_with_label(val1, label1, val2, label2)
                next_level.append((winner_val, winner_label))
            else:
                # Füge das letzte, ungerade Element für die nächste Runde hinzu.
                next_level.append(current_level[i])
        current_level = next_level
    return current_level[0]

# --- FastAPI App und Endpunkte ---
app = FastAPI()

class EncryptedDataRequest(BaseModel):
    """ Definiert die erwartete Struktur der JSON-Anfrage vom Client. """
    serialized_encrypted_patient_vector: str

@app.get("/assets")
def get_assets():
    """ Stellt die initialen Assets bereit. """
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
    Nimmt verschlüsselte Daten entgegen und führt die komplette Klassifikation homomorph durch.
    """
    public_context = ckks_context_cache.get("public_context")
    if public_context is None:
        raise HTTPException(status_code=500, detail=f"CKKS-Kontext konnte nicht geladen werden. Sicherstellen, dass '{CONTEXT_FILE_PATH}' existiert.")

    serialized_vector_bytes = base64.b64decode(request.serialized_encrypted_patient_vector)
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_vector_bytes)

    # Schritt 1: Homomorphe Distanzberechnung
    encrypted_distances = []
    for p_vector in PROTOTYPES:
        enc_diff = encrypted_patient_vector - p_vector
        enc_squared_diff = enc_diff.pow(2)

        # Rescale nach Potenzierung zur Vermeidung von Overflow
        rescaled_sq_diff = enc_squared_diff * 1.0 # hier wird Rescaling von TenSEAL aktiviert (Multiplikation)
        enc_distance = rescaled_sq_diff.sum()
        encrypted_distances.append(enc_distance)
    print("--- SERVER: Homomorphe Distanzberechnung abgeschlossen. ---")

    # Schritt 2: Homomorphe Klassifikation (Finden der Min. Distanz)
    norm_factor = 1.0 / MAX_EXPECTED_SQUARED_DISTANCE
    normalized_distances = [d * norm_factor for d in encrypted_distances]
    
    encrypted_pairs = []
    for dist, label in zip(normalized_distances, PROTO_LABELS):
        # Konvertiere Label in float für CKKS und verschlüssele es
        enc_label = ts.ckks_vector(public_context, [float(label)])
        encrypted_pairs.append((dist, enc_label))
    
    enc_winner_val_norm, enc_winner_label = find_min_and_label_tournament(encrypted_pairs)
    
    # Denormalisiere den Gewinner-Wert für den Client zurück
    enc_winner_val = enc_winner_val_norm * (1.0 / norm_factor)
    print("--- SERVER: Homomorphe Klassifikation abgeschlossen. ---")
    
    # Schritt 3: Ergebnisse für die Rücksendung vorbereiten
    serialized_distances_b64 = [base64.b64encode(d.serialize()).decode('utf-8') for d in encrypted_distances]
    serialized_winner_label_b64 = base64.b64encode(enc_winner_label.serialize()).decode('utf-8')
    serialized_winner_val_b64 = base64.b64encode(enc_winner_val.serialize()).decode('utf-8')
    
    return {
        "distances": serialized_distances_b64,
        "proto_labels": PROTO_LABELS.tolist(),
        "winner_label": serialized_winner_label_b64,
        "winner_val": serialized_winner_val_b64
    }