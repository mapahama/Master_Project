# (GLVQ   - Proof of Concept - Verschlüsselte Klassifikation / Unverschlüsselte Klassifikation) 
# Der Code testet Klassifikationsgenauigkeit sowohl auf verschlüsselte Daten als auch auf Klartext-Daten

# !!! Anmerkung: Die Präzision der verschlüsselten Daten hängt von den CKKS-Parametern (wie global_scale) ab.

import time
import numpy as np
import tenseal as ts
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn_lvq import GlvqModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def prepare_data():
    """
    Lädt und bereinigt den Heart Disease Datensatz.
    Diese Funktion übernimmt das Laden, die Bereinigung (NaNs) und die binäre Umwandlung der Zielvariablen.
    """
    print("1. Lade und bereite Heart Disease Datensatz vor...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    # Wandle die Zielvariable in ein binäres Format um (0 = Gesund, 1 = Krank)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y_binary = (y > 0).astype(int)

    # Bereinige Merkmale: Ersetze '?' durch NaN, dann fülle mit dem Spaltenmedian
    X.replace('?', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)
    
    print("Datensatz vorbereitet.\n")
    return X, y_binary

def train_and_evaluate_plaintext(X, y):
    """
    Trainiert ein GLVQ-Modell und bewertet dessen Genauigkeit auf standardmäßigen, unverschlüsselten Testdaten.
    """
    print("2. Trainiere GLVQ-Modell und evaluiere auf Klartext-Daten...")
    
    # Teile den Datensatz in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Skaliere die Daten, um sie in einen Bereich von [-1, 1] zu bringen
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialisiere und trainiere das GLVQ-Modell
    glvq = GlvqModel(prototypes_per_class=1, random_state=42)
    glvq.fit(X_train_scaled, y_train)

    # Mache Vorhersagen
    y_pred_plaintext = glvq.predict(X_test_scaled)
    
    # Berechne Metriken
    accuracy_plaintext = accuracy_score(y_test, y_pred_plaintext)
    precision_plaintext = precision_score(y_test, y_pred_plaintext, zero_division=0)
    recall_plaintext = recall_score(y_test, y_pred_plaintext, zero_division=0)
    f1_plaintext = f1_score(y_test, y_pred_plaintext, zero_division=0)

    print(f"  - Genauigkeit auf Klartext-Testdaten: {accuracy_plaintext:.9f}")
    print(f"  - Precision auf Klartext-Testdaten: {precision_plaintext:.9f}")
    print(f"  - Recall auf Klartext-Testdaten: {recall_plaintext:.9f}")
    print(f"  - F1-Score auf Klartext-Testdaten: {f1_plaintext:.9f}")
    print("Evaluation auf Klartext-Daten abgeschlossen.\n")
    
    # Gebe das trainierte Modell, die skalierten Testdaten und die Test-Labels zurück
    return glvq, X_test_scaled, y_test.values

def evaluate_encrypted(glvq_model, X_test_scaled, y_test):
    """
    Führt Vorhersagen auf CKKS-verschlüsselten Daten durch und bewertet die Genauigkeit.
    """
    print("3. Richte TenSEAL-Kontext für CKKS ein...")
    
    # Initialisiere den CKKS-Kontext mit den notwendigen Parametern
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[50, 40, 40, 50]
    )
    # Erzeuge Galois-Schlüssel, die für Rotationsoperationen benötigt werden
    context.generate_galois_keys()
    # Setze den globalen Skalierungsfaktor für die Dezimalgenauigkeit
    context.global_scale = 2**40
    
    print("TenSEAL-Kontext erstellt.\n")
    print("4. Evaluiere das Modell auf verschlüsselten Testdaten...")
    
    print("  - Verschlüssele Testdaten...")
    # Verschlüssele jeden Datenpunkt im Test-Set in einen CKKS-Vektor
    encrypted_X_test = [ts.ckks_vector(context, x.tolist()) for x in X_test_scaled]
    print(f"  - {len(encrypted_X_test)} Testvektoren verschlüsselt.")

    # Rufe die Prototypen und deren Labels vom trainierten Modell ab
    prototypes = glvq_model.w_
    prototype_labels = glvq_model.c_w_
    
    # Konvertiere Prototypen einmal in eine Liste von Listen, um eine
    # wiederholte Konvertierung in der Schleife zu vermeiden.
    prototypes_list = [p.tolist() for p in prototypes]

    y_pred_encrypted = []
    
    print("  - Starte die Vorhersage-Schleife auf verschlüsselten Daten...")
    start_time = time.time()

    # Iteriere über jeden verschlüsselten Datenpunkt
    for enc_vector in encrypted_X_test:
        decrypted_distances = []
        # Berechne die Distanz zu jedem Prototyp
        for proto in prototypes_list: # Verwende die vorab konvertierte Liste
            # Subtrahiere den Klartext-Prototyp vom verschlüsselten Vektor
            enc_diff = enc_vector - proto
            # Berechne die quadrierte euklidische Distanz homomorph
            enc_sq_diff = enc_diff.pow(2)
            enc_sq_dist = enc_sq_diff.sum()
            # Entschlüssele die Distanz und füge sie der Liste hinzu
            decrypted_distances.append(enc_sq_dist.decrypt()[0])

        # Finde den Index des Prototyps mit der geringsten Distanz
        closest_proto_idx = np.argmin(decrypted_distances)
        # Weist die entsprechende Klasse als Vorhersage zu
        prediction = prototype_labels[closest_proto_idx]
        y_pred_encrypted.append(prediction)

    end_time = time.time()
    
    # Berechne die Metriken der verschlüsselten Vorhersagen
    accuracy_encrypted = accuracy_score(y_test, y_pred_encrypted)
    precision_encrypted = precision_score(y_test, y_pred_encrypted, zero_division=0)
    recall_encrypted = recall_score(y_test, y_pred_encrypted, zero_division=0)
    f1_encrypted = f1_score(y_test, y_pred_encrypted, zero_division=0)

    print("  - Vorhersage auf verschlüsselten Daten abgeschlossen.")
    print(f"  - Benötigte Zeit für verschlüsselte Vorhersagen: {end_time - start_time:.2f} Sekunden.")
    print(f"  - Genauigkeit auf verschlüsselten Testdaten: {accuracy_encrypted:.9f}")
    print(f"  - Precision auf verschlüsselten Testdaten: {precision_encrypted:.9f}")
    print(f"  - Recall auf verschlüsselten Testdaten: {recall_encrypted:.9f}")
    print(f"  - F1-Score auf verschlüsselten Testdaten: {f1_encrypted:.9f}")
    print("Evaluation auf verschlüsselten Daten abgeschlossen.\n")


if __name__ == '__main__':
    features, labels = prepare_data()
    model, X_test, y_test_data = train_and_evaluate_plaintext(features, labels)
    evaluate_encrypted(model, X_test, y_test_data)

    print("--- Proof of Concept abgeschlossen ---")