# =================================================================================
# SERVER-SEITIGE LOGIK FÜR GMLVQ (homomorph mit CKKS)
# =================================================================================
# Dieser Code simuliert die serverseitige Komponente einer Client-Server-Anwendung
# für eine  Klassifikation mit dem GMLVQ-Algorithmus.
#
# Die Hauptaufgaben dieses Servers sind:
# 1. Einmaliges Trainieren eines GMLVQ-Modells beim ersten Start und Bereitstellen
#    der notwendigen, nicht-geheimen "Assets" (Scaler, Feature-Namen).
# 2. Bereitstellen einer API-ähnlichen Funktion (`process_encrypted_request`), die
#    Anfragen vom Client entgegennimmt.
# 3. Empfangen eines homomorph verschlüsselten Patientendaten-Vektors vom Client.
# 4. Durchführung der  GMLVQ-Distanzberechnung auf den verschlüsselten
#    Daten. Dies ist eine "blinde" Berechnung, da der Server die Daten nie entschlüsselt.
# 5. Zurücksenden der verschlüsselten Ergebnisse an den Client.
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
    Gibt die Prototypen, deren Labels, den Scaler, die Feature-Namen und die
    trainierte Omega-Matrix zurück.
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
    gmlvq = GmlvqModel(prototypes_per_class=3, regularization=0.35, random_state=42)
    gmlvq.fit(X_scaled, y_binary_np)

    # --- Schritt 4: Relevante Modell-Parameter extrahieren ---
    prototypes = gmlvq.w_      # Die gelernten Prototypen-Vektoren
    proto_labels = gmlvq.c_w_   # Die Klassen-Labels der Prototypen (0 oder 1)
    omega = gmlvq.omega_        # Die gelernte Transformationsmatrix (Omega)

    print("--- SERVER: GMLVQ-Assets geladen. ---")
    return prototypes, proto_labels, scaler, feature_names, omega


def process_encrypted_request(serialized_vector, serialized_public_context):
    """
    Nimmt vom Client den verschlüsselten Patientenvektor und den öffentlichen CKKS -Kontext
    entgegen, wendet homomorph die Omega-Transformation an und berechnet die Distanzen
    zu den GMLVQ-Prototypen. Rückgabe: Liste verschlüsselter Distanzen + Klassenlabel.
    """
    # Lade die Server-Assets (Modell-Parameter) aus dem Cache.
    prototypes, proto_labels, _, _, omega = get_server_assets()

    # Rekonstruiere den öffentlichen CKKS-Kontext aus den vom Client gesendeten Daten.
    public_context = ts.context_from(serialized_public_context)

    # Sicherheits-Check: Stelle sicher, dass der Kontext öffentlich ist und keinen geheimen Schlüssel enthält.
    print("\n--- SERVER: Kontextüberprüfung ---")
    print("-> Public Key vorhanden:", public_context.has_public_key())
    print("-> Secret Key vorhanden (sollte False sein):", public_context.has_secret_key())
    print("----------------------------------------------------")

    # Rekonstruiere den verschlüsselten (serializiert) Vektor aus den vom Client gesendeten Daten.
    encrypted_patient_vector = ts.ckks_vector_from(public_context, serialized_vector)


    # ==============================================================================
    # GMLVQ-DISTANZBERECHNUNG
    # ==============================================================================
    # Die GMLVQ-Distanz d² ist  die quadrierte euklidische Distanz.
    # Wir müssen  zuerst beide Vektoren (Patient ξ und
    # Prototyp w) mit der Matrix Ω (omega) in einen neuen "Relevanz-Raum" projizieren und
    # DANN dort den Abstand berechnen.

    # --- A) Homomorphe Projektion des Patientendatenvektors: Enc(Ωξ) ---
    # Wir führen die Matrix-Vektor-Multiplikation Ω * Enc(ξ) homomorph durch.
    # Das Ergebnis ist ein neuer, verschlüsselter Vektor im projizierten Raum.
    # Dies geschieht, indem wir den verschlüsselten Vektor mit jeder einzelnen (Klartext-)Zeile
    # der Omega-Matrix per Skalarprodukt multiplizieren.
    encrypted_embedded = []
    for omega_row in omega:
        # Berechnet das Skalarprodukt: Enc(ξ) ⋅ ω_zeile = Enc(ξ ⋅ ω_zeile)
        dot_product = encrypted_patient_vector.dot(omega_row.tolist())
        encrypted_embedded.append(dot_product)


    # --- B) Berechnung der Distanz zu jedem Prototyp im projizierten Raum ---
    encrypted_distances = []
    for i, proto in enumerate(prototypes):

        # B.1) Projiziere den Klartext-Prototyp in den Relevanz-Raum: Ωw
        embedded_proto = omega @ proto

        # B.2) Berechne die Differenz: Enc(Ωξ) - Ωw
        # Dies geschieht elementweise zwischen dem verschlüsselten und dem Klartext-Vektor.
        diff_enc = []
        for enc_val, p_val in zip(encrypted_embedded, embedded_proto):
            diff_enc.append(enc_val - p_val)

        # B.3) Quadriere die Differenzen homomorph
        # Jeder Wert im Differenzvektor wird quadriert.
        sq_diff = [x.pow(2) for x in diff_enc]

        # B.4) Summiere die quadrierten Differenzen
        # Dies ergibt die finale quadrierte euklidische Distanz im projizierten Raum.
        enc_distance = sq_diff[0]
        for j in range(1, len(sq_diff)):
            enc_distance += sq_diff[j]

        encrypted_distances.append(enc_distance)

        # Debug-Ausgaben 
        print(f"\n--- Prototyp {i}")
        print("Embedded prototype:", embedded_proto)
        print("Quadr. Differenz (verschlüsselt):", sq_diff) 
        print("Verschlüsselte Distanz:", enc_distance)


    # --- Ergebnis für den Rückversand an den Client vorbereiten ---
    # Die berechneten, verschlüsselten Distanz-Skalare werden serialisiert (Notwendig für CKKS-Daten bei Client/Server kommunikation über ein Netz)
    serialized_results = []
    for vec, label in zip(encrypted_distances, proto_labels):
        # Jeder verschlüsselte Vektor wird in Bytes umgewandelt und mit seinem
        # Klartext-Label als Tupel gespeichert.
        serialized_results.append((vec.serialize(), label))

    print("--- SERVER: Distanzberechnung abgeschlossen. ---")
    return serialized_results