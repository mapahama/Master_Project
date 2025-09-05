# Tschebyscheff Polynome - Client


#  .\venv311\Scripts\activate
# Start App  via      streamlit run client.py
# client und server müssen in 2 separaten Konsolen gestartet werden

# =================================================================================
# CLIENT-SEITIGE LOGIK für GLVQ (Min. Distanz bestimmen  min Polynom-Approximation - Tschebyscheff Polynome)
# =================================================================================
#
# =================================================================================
# Die Logik ist nun identisch zur GMLVQ-Anwendung:
# 1. Der Client verschlüsselt die Patientendaten und sendet sie an den Server.
# 2. Der Server findet den Gewinner-Prototyp (min. Distanz & Label dazu) homomorph (durch Polynom Approximation)
#    und sendet das verschlüsselte Ergebnis zurück an Client.
# 3. Der Client entschlüsselt nur noch das finale Ergebnis und muss die min.Distanz (Klasse Gesund oder Krank) NICHT bestimmen
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import tenseal as ts
import time
import plotly.express as px
import requests
import base64
import joblib
from io import BytesIO

# Globale Variable für den Dateipfad des CKKS-Kontexts
CONTEXT_FILE_PATH = "ckks_context_glvq.bin"

# --- Server-Kommunikation und Kontext-Setup ---
@st.cache_resource
def setup_client_environment():
    """
    Bereitet die Client-Umgebung vor, holt Assets vom Server und erstellt den CKKS-Kontext.
    Der öffentliche Kontext wird für den Server in eine Datei geschrieben.
    """
    print("--- CLIENT: Frage Scaler und Feature-Namen vom Server an... ---")
    try:
        response = requests.get("http://localhost:8000/assets")
        response.raise_for_status()
        assets = response.json()
       
        scaler_bytes = base64.b64decode(assets["scaler"])
        scaler = joblib.load(BytesIO(scaler_bytes))
        feature_names = assets["feature_names"]
       
        print("--- CLIENT: Scaler und Feature-Namen vom Server erhalten. ---")
    except requests.exceptions.RequestException as e:
        st.error(f"FEHLER: Konnte keine Verbindung zum Server herstellen. Details: {e}")
        st.stop()

    print("--- CLIENT: Erstelle CKKS-Kontext und Schlüsselpaar... ---")
    # Der Kontext muss tief genug für die Polynom-Approximation auf dem Server sein
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384, #32768,
        coeff_mod_bit_sizes=[50] + [30]*10 + [50] # Tiefe für Polynom-Berechnungen, reicht für Tscheb. Poly Degree (10)
    )
    context.generate_galois_keys()
    context.global_scale = 2**30
   
    # Speichern den öffentlichen CKKS Kontext in eine Datei, die der Server lesen kann
    context_for_server = context.copy()
    context_for_server.make_context_public(generate_galois_keys=False)
    with open(CONTEXT_FILE_PATH, "wb") as f:
        f.write(context_for_server.serialize())
    print(f"--- CLIENT: Öffentlicher CKKS-Kontext in '{CONTEXT_FILE_PATH}' für Server gespeichert. ---")

    return scaler, context, feature_names

# ==============================================================================
# BENUTZEROBERFLÄCHE (UI)
# ==============================================================================
st.set_page_config(layout="wide", page_title="Privacy-Preserving GLVQ")
st.title("🩺 Privacy-Preserving Heart Disease Classification (GLVQ)")
st.write("Der **Client** verschlüsselt die Daten. Der **Server** führt die Klassifikation durch (min. Distanz Bestimmung), ohne die Daten oder das Ergebnis zu kennen.")

try:
    scaler, context, feature_names = setup_client_environment()
except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist bei der Initialisierung aufgetreten: {e}")
    st.stop()

st.sidebar.header("Client: Patientendaten eingeben")
user_input = {}
default_values = [54.4, 0.68, 3.16, 131.6, 246.7, 0.14, 0.99, 149.6, 0.32, 1.0, 1.6, 0.67, 4.73]
for i, feature in enumerate(feature_names):
    user_input[feature] = st.sidebar.number_input(
        label=f"{i+1}. {feature}", value=default_values[i], step=1.0, format="%.2f"
    )

# #############################################################################################
# NEU: Funktion zur Überprüfung der Sicherheit der Klassifikation
# #############################################################################################
def check_classification_certainty(decrypted_label):
    """
    Überprüft, ob der entschlüsselte Label-Wert in einem der definierten
    "unsicheren" Bereiche liegt.

    Args:
        decrypted_label (float): Der entschlüsselte Fließkommawert des Gewinner-Labels.

    Returns:
        bool: True, wenn die Klassifikation als sicher gilt, andernfalls False.
    """
    # Definiere die unsicheren Bereiche
    is_uncertain = (
        decrypted_label < -0.35 or
        (0.37 <= decrypted_label <= 0.75) or
        decrypted_label > 1.35
    )
    return not is_uncertain # True, wenn sicher; False, wenn unsicher

if st.sidebar.button("Klassifikation durchführen", type="primary"):
   
    # === Schritt 1: CLIENT - Daten aufbereiten und verschlüsseln ===
    st.header("1. Client-Aktionen")
    patient_df = pd.DataFrame([user_input])
    st.write("**Aktion:** Rohdaten des Patienten werden gesammelt.")
    st.dataframe(patient_df)
   
    scaled_patient_vector = scaler.transform(patient_df)[0]
    encrypted_patient_vector = ts.ckks_vector(context, scaled_patient_vector)
    st.info("🔒 Die Patientendaten wurden skaliert und sicher verschlüsselt.")
   
    # === Schritt 2: CLIENT -> SERVER - Anfrage senden ===
    st.header("2. Client-Server-Interaktion")
   
    serialized_patient_vector_b64 = base64.b64encode(encrypted_patient_vector.serialize()).decode('utf-8')
    payload = {"serialized_encrypted_patient_vector": serialized_patient_vector_b64}
   
    start_zeit = time.time()
   
    with st.spinner('Warte auf die homomorphe Berechnung des Servers...'):
        try:
            response = requests.post("http://localhost:8000/classify", json=payload)
            response.raise_for_status()
            response_json = response.json()
           
            # Empfange alle Distanzen (für die Anzeige) und das Gewinner-Label
            all_distances_encrypted = [ts.ckks_vector_from(context, base64.b64decode(d)) for d in response_json["distances"]]
            proto_labels = response_json["proto_labels"]
            winner_label_encrypted = ts.ckks_vector_from(context, base64.b64decode(response_json["winner_label"]))
            winner_val_encrypted = ts.ckks_vector_from(context, base64.b64decode(response_json["winner_val"]))

        except requests.exceptions.RequestException as e:
            st.error(f"Fehler bei der Server-Kommunikation: {e}")
            st.stop()

    end_zeit = time.time()
    verstrichene_zeit = end_zeit - start_zeit
    st.success(f"Antwort vom Server nach **{verstrichene_zeit:.4f}** Sekunden erhalten.")

    # === Schritt 3: CLIENT - Antwort entschlüsseln und Ergebnis anzeigen ===
    st.header("3. Ergebnis auf der Client-Seite")
    st.write("**Aktion:** Client entschlüsselt die Antwort mit seinem privaten Schlüssel.")

    secret_key = context.secret_key()
    decrypted_distances = [d.decrypt(secret_key)[0] for d in all_distances_encrypted]
    decrypted_winner_label = winner_label_encrypted.decrypt(secret_key)[0]
    decrypted_winner_val = winner_val_encrypted.decrypt(secret_key)[0]
   
    # Runde das Ergebnis, um das exakte Label wiederherzustellen (0=gesund, 1=krank)
    final_winner_label = int(round(decrypted_winner_label))
    print(f"--- CLIENT: 💎 decrypted_winner_label:  '{decrypted_winner_label}' ---")
    print(f"--- CLIENT: 💎 decrypted_winner_val '{decrypted_winner_val}' ---")
    print(f"--- CLIENT: 💎 final_winner_label (gerundet) '{final_winner_label}' ---")


    # Anzeige der Distanztabelle (Erklärbarkeit)
    results_df = pd.DataFrame({
        "Prototyp-Klasse": ['GESUND' if label == 0 else 'KRANK' for label in proto_labels],
        "Entschlüsselte Distanz (quadriert)": decrypted_distances
    }, index=np.arange(1, len(decrypted_distances) + 1))
    results_df.index.name = "Prototyp-Nr."

    def highlight_min(row):
        is_min = np.isclose(row['Entschlüsselte Distanz (quadriert)'], decrypted_winner_val)
        return ['background-color: #636363'] * len(row) if is_min else [''] * len(row)

    st.table(results_df.style.apply(highlight_min, axis=1).format({"Entschlüsselte Distanz (quadriert)": "{:.4f}"}))

    # Anzeige der finalen Klassifikation
    st.subheader("Finale Klassifikation:")

    # #############################################################################################
    # NEU: Überprüfung und Anzeige der Sicherheit der Klassifikation
    # #############################################################################################
    is_certain = check_classification_certainty(decrypted_winner_label)
    if not is_certain:
        st.warning(
            f"**Warnung: Unsichere Modell-Entscheidung**\n\n"
            f"Der entschlüsselte Ergebniswert (Label) von **{decrypted_winner_label:.4f}** liegt in einem Bereich, "
            f"in dem die Zuverlässigkeit der Klassifikation gering ist. "
            f"Das Ergebnis sollte mit Vorsicht interpretiert werden.",
            icon="⚠️"
        )
    # #############################################################################################

    if final_winner_label == 1:
        st.error("Der Patient wird als **KRANK** eingestuft.", icon="💔")
    else:
        st.success("Der Patient wird als **GESUND** eingestuft.", icon="💚")

    # Grafische Darstellung der Distanzen als Säulendiagramm (Erklärbarkeit)
    st.divider()
    st.subheader("📊 Grafischer Vergleich der Distanzen")
    st.write("Jeder Balken repräsentiert einen Prototyp des Modells. "
             "Die Klassifikation basiert auf dem Prototyp mit der kürzesten Distanz.")

    plot_df = pd.DataFrame({
        "Prototyp-Nr.": [f"Proto {i+1}" for i in range(len(proto_labels))],
        "Klasse": ["GESUND" if l == 0 else "KRANK" for l in proto_labels],
        "Quadrierte Distanz": decrypted_distances
    })

    color_map = {
        'GESUND': 'mediumseagreen', 'KRANK': 'indianred',
        'GEWINNER (GESUND)': 'darkgreen', 'GEWINNER (KRANK)': 'darkred'
    }
   
    plot_df['Legenden-Kategorie'] = plot_df['Klasse']
    min_dist_idx = np.argmin(decrypted_distances) # Finde den Index für die Grafik
    winning_class_label = plot_df.loc[min_dist_idx, 'Klasse']
    winning_category = f'GEWINNER ({winning_class_label})'
    plot_df.loc[min_dist_idx, 'Legenden-Kategorie'] = winning_category

    fig = px.bar(plot_df, x="Prototyp-Nr.", y="Quadrierte Distanz", color='Legenden-Kategorie',
                 color_discrete_map=color_map, title="Vergleich der Distanzen zu allen Prototypen")
    fig.update_layout(legend_title_text='Status')
    fig.update_xaxes(categoryorder='array', categoryarray=plot_df['Prototyp-Nr.'])
    st.plotly_chart(fig, use_container_width=True)