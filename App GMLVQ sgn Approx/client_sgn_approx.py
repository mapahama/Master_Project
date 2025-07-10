#    streamlit run client_sgn_approx.py
#    .\venv311\Scripts\activate

# =================================================================================
# CLIENT-SEITIGE LOGIK & UI für GMLVQ + CKKS
# =================================================================================
# 1. Der Nutzer gibt seine Daten über die UI ein
# 2. Der Client skaliert die Werte im Bereich [-1,1] und sendet diese als ein CKKS-verschlüsselten Vektor an den Server
#    Der Client sendet auch den CKKS-Context (ohne Priv Key und Galouis Keys) an den Server
# 3. Der Client empfängt die vom Server homomorph ermittelte Gewinner-Klasse (krank oder gesund), entschlüsselt sie und zeigt sie auf der UI an.
#
# Erklärbarkeit: Zusätzlich werden zur Transparenz die (entschlüsselten) Distanzen zu allen Prototypen 
#                und ein Diagramm zur Wichtigkeit (Relevanz) der einzelnen Merkmale angezeigt. 
#                Dies hilft, die Entscheidung des KI-Modells nachvollziehbar zu machen (Explainable AI).
# =================================================================================
# HINWEIS: Serialisierung/Deserialisierung wurde entfernt (wegen Performanz - es ist Proof of Concept) 
#          Die Kommunikatio mit dem Server (in diesem Code) erfolgt nun über direkte Funktionsaufrufe mit TenSEAL-Objekten.
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tenseal as ts
import time
import altair as alt

# !!! Importiert die Server-Funktionen. In einer echten Anwendung wäre dies ein API-Aufruf (z.B. via REST)
from server_sgn_approx import get_server_assets, process_encrypted_request

# =================================================================================
# Bereich 1: Client-Setup und Initialisierung der Kryptographie
# =================================================================================

@st.cache_resource # Der Decorator sorgt dafür, dass dieser aufwändige Setup-Prozess nur einmal ausgeführt wird
def setup_client_environment():
    """
    Initialisiert die gesamte Client-Umgebung. Dies beinhaltet das Abrufen von öffentlichen
    Server-Assets (wie dem Scaler) und die Erstellung des kryptografischen Kontexts,
    der sowohl den öffentlichen als auch den geheimen (privaten) Schlüssel enthält.
    Returns:
        tuple: Ein Tupel, das den Scaler, den vollständigen CKKS-Kontext und die Feature-Namen enthält.
    """
    print("--- CLIENT: Lade Server-Assets und initialisiere Umgebung... ---")

    # Ruft öffentliche Assets vom Server ab. Der Scaler ist wichtig, damit der Client die Daten
    # auf exakt dieselbe Weise wie das Trainings-Set des Servers vorbereiten kann (im Bereich [-1,1])
    _, _, scaler, feature_names, _, _ = get_server_assets()

    # --- Erstellung des CKKS-Kontexts ---
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, # Verschlüsselungsschema CKKS
        poly_modulus_degree=32768,  # Ein Sicherheitsparameter. Größer = sicherer, aber langsamer  (16384 getestet, geht nicht mehr nach 12 Bit-Moduln)
        # Definiert die Kette von Moduln, die das "Rauschbudget" für Multiplikationen bestimmen.
        # Jede Multiplikation verbraucht einen der mittleren Moduln. Die Anzahl muss für die
        # Komplexität des Polynoms ausreichen.
        coeff_mod_bit_sizes=[50] + [40]*13 + [50]  # 13 * 40-Bit-Moduln sind hier für Grad 7 ausreichend (Bei 12 -> Fehler - scale out of bounds)
    )
    context.generate_galois_keys() # für Rotationen und Dot-Produkte
    context.global_scale = 2**40 # höhere Skala bedeutet höhere Präzision für die Dezimalzahlen
    context.auto_rescale = True
    return scaler, context, feature_names

# =================================================================================
# Bereich 2: Benutzeroberfläche (UI) mit Streamlit
# =================================================================================

# Konfiguriert das Seitenlayout der Web-App
st.set_page_config(layout="wide", page_title="Privacy-Preserving GMLVQ")
st.title("🩺 Privacy-Preserving Heart Disease Classification (GMLVQ)")
st.markdown(
    '<p style="color:#a7a1a1;">Dies ist eine Simulation einer getrennten Client-Server-Architektur. '
    'Der <b>Client</b> (diese UI) verschlüsselt die Daten. Der <b>Server</b> (eine separate Logik) '
    'führt die Klassifikation durch, ohne die Daten oder das Ergebnis zu kennen.</p>',
    unsafe_allow_html=True
)

try:
    scaler, context, feature_names = setup_client_environment() # die Client-Umgebung  initialisieren
except FileNotFoundError:
    st.error("FEHLER: Die Datensatz-Datei 'heart_data_pretty.csv' wurde auf der Serverseite nicht gefunden.")
    st.stop()

# Erstellt Sidebar mit einem Formular, wo die Nutzer ihre Daten eingeben können (und an den Server abschicken)
st.sidebar.header("Client: Patientendaten eingeben")
user_input = {}
default_values = [54.4, 0.68, 3.16, 131.6, 246.7, 0.14, 0.99, 149.6, 0.32, 1.0, 1.6, 0.67, 4.73] # default Werte
for i, feature in enumerate(feature_names):
    default_val = default_values[i] if i < len(default_values) else 0.0
    user_input[feature] = st.sidebar.number_input(f"{i+1}. {feature}", value=default_val, step=1.0, format="%.2f")


# =================================================================================
# Bereich 3: Hauptlogik - nach Button-Klick (Patientendaten skallieren, verschlüsseln und an den Server senden)
# =================================================================================

if st.sidebar.button("Klassifikation durchführen", type="primary"):
    # === Schritt 1: CLIENT - Daten normalisieren [-1,1] und verschlüsseln  ===
    st.header("1. Client-Aktionen")
    patient_df = pd.DataFrame([user_input])
    st.dataframe(patient_df)
    # Wendet den vom Server erhaltenen Scaler an, um die Patientendaten zu normalisieren.
    scaled_patient_vector = scaler.transform(patient_df)[0]
    # verschlüsseln
    encrypted_patient_vector = ts.ckks_vector(context, scaled_patient_vector)
    st.info("🔒 Die Patientendaten sind jetzt sicher und können das Gerät verlassen.")
    print("--- Skalierter Patienten-Vektor ---") 
    print(scaled_patient_vector)                 

    # === Schritt 2: CLIENT -> SERVER - Anfrage senden ===
    st.header("2. Simulation der Interaktion")
    
    # Wichtig! # Erstellt eine Kopie des Kontexts für den Versand an den Server - ohne Private Key (Sicherheit), ohne Galois Keys (Performanz)
    context_for_server = context.copy()  
    context_for_server.make_context_public(generate_galois_keys=False) # hier Priv Key und Galois Keys aus dem Kontext entfernen

    with st.spinner('Warte auf Antwort vom Server...'):
        time.sleep(1)

        # Dies ist der simulierte API-Aufruf an den Server.
        # Der Client sendet den verschlüsselten Vektor und den öffentlichen Kontext.
        # Er empfängt die verschlüsselten Ergebnisse zurück, vom Server.
        all_distances, clean_proto_labels, winner_label_encrypted, winner_val_encrypted, relevances = process_encrypted_request(
            encrypted_patient_vector, context_for_server
        )
    
    # UI 
    st.markdown('<p style="color:#a7a1a1;">📤 <b>Client an Server:</b> Sende verschlüsselten Datenvektor und öffentlichen Kontext.</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a7a1a1;">... Server berechnet Distanzen und finde die kleinste (mit ihrer Klasse) ...</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a7a1a1;">📥 <b>Server an Client:</b> Sende verschlüsselte Distanzen, verschlüsselten Gewinner-LABEL und die Feature-Relevanzen.</p>', unsafe_allow_html=True)

    # === Schritt 3: CLIENT - Server Antwort entschlüsseln und Ergebnis auf der UI anzeigen ===
    st.header("3. Ergebnis auf Client-Seite")
    st.markdown(
        '<p style="color:#a7a1a1;"><b>Aktion:</b> Client empfängt die Daten, entschlüsselt sie mit seinem privaten Schlüssel und bestimmt das finale Ergebnis.</p>',
        unsafe_allow_html=True
    )
    
    # Entschlüssele die Liste der Distanzen für die Tabellenanzeige (Explainable AI)
    decrypted_distances = [d.decrypt(context.secret_key())[0] for d in all_distances]
    
    # 1. Entschlüssele das einzelne Gewinner-Label (das Ergebnis ist verrauscht, z.B. 0.998 oder -0.999)
    decrypted_winner_label = winner_label_encrypted.decrypt(context.secret_key())[0]
    # Decrypt winner Distanz
    decrypted_winner_val = winner_val_encrypted.decrypt(context.secret_key())[0]
    print(f"--- Client erhält vom Server: Entschlüsseltes Gewinner-Label (verrauscht): {decrypted_winner_label} ---") 
    # print(f"--------> !!! Client erhält vom Server: Entschlüsselte kleinste Distanz (verrauscht): {decrypted_winner_val} ---") 

    # 2. Runde das Ergebnis zur nächsten ganzen Zahl (-1 oder 1), um Rauschen zu entfernen
    final_winner_label = int(round(decrypted_winner_label))
    print(f"--- Client bestimmt: Finales sauberes Label ist {final_winner_label} ---")
    
    # Stellt die entschlüsselten Distanzen in einer übersichtlichen Tabelle dar (AI Erklärbarkeit)
    results_df = pd.DataFrame({
        "Prototyp-Klasse": ['KRANK' if label == 1 else 'GESUND' for label in clean_proto_labels],
        "Entschlüsselte Distanz (quadriert)": decrypted_distances
    }, index=np.arange(1, len(decrypted_distances) + 1))
    results_df.index.name = "Prototyp-Nr."
    st.table(results_df)

    # Zeigt die finale Klassifikation basierend auf dem entschlüsselten und gerundeten Label an.
    st.subheader("Finale Klassifikation (vom Server ermittelt):")
    if final_winner_label == 1:
        st.error("Der Patient wird als **KRANK** eingestuft.", icon="💔")
    else:
        st.success("Der Patient wird als **GESUND** eingestuft.", icon="💚")
    
    # Schritt 4 - ERKLÄRBARKEIT - Relevanz Matrix-Daten  (aus der Diagonale)  anzeigen (welche Patientendaten waren für die Klassifikation entscheidend)
    st.divider()
    st.subheader("💡 Erklärbarkeit des GMLVQ-Modells")
    st.markdown(
        '<p style="color:#a7a1a1;">GMLVQ lernt, welche Merkmale für die Klassifikation wichtig sind...</p>',
        unsafe_allow_html=True
    )
    relevance_df = pd.DataFrame({"Merkmal": feature_names, "Relevanz": relevances}).sort_values(by="Relevanz", ascending=False)
    chart = alt.Chart(relevance_df).mark_bar().encode(x=alt.X('Merkmal', sort=None), y='Relevanz').properties(title='Wichtigkeit der Merkmale')
    st.altair_chart(chart, use_container_width=True)