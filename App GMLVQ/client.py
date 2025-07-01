# 	streamlit run client.py
#   .\venv311\Scripts\activate

# =================================================================================
# CLIENT-SEITIGE LOGIK & UI für GMLVQ + CKKS
# =================================================================================
# Dieser Code implementiert die Client-Anwendung mit einer Streamlit-Benutzeroberfläche.
# Die Hauptaufgaben dieses Clients sind:
# 1. Bereitstellen einer UI, damit ein Nutzer seine Patientendaten eingeben kann.
# 2. Vorbereiten (Skalieren) dieser Daten, damit sie für das Modell des Servers
# 	verständlich sind.
# 3. Erzeugen eines Verschlüsselungskontexts (CKKS) und Halten des geheimen Schlüssels.
# 4. Verschlüsseln der Patientendaten und Senden der Anfrage an den Server.
# 5. Empfangen der verschlüsselten Ergebnisse und der Modell-Relevanzen vom Server.
# 6. Entschlüsseln der Ergebnisse und Treffen der finalen Klassifikationsentscheidung.
# 7. Visualisierung der Modellerklärung durch Darstellung der gelernten Merkmals-Relevanzen.
#
# Der Client kennt das Machine-Learning-Modell (GMLVQ) oder dessen Prototypen zu keinem Zeitpunkt.
# =================================================================================


# --- Bibliotheken importieren ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tenseal as ts
import time
import altair as alt  # Säulendiagramm

# --- Server-Kommunikation simulieren ---
# Importiert die "API-Funktion" aus der server.py Datei.
from server import get_server_assets, process_encrypted_request


@st.cache_resource
def setup_client_environment():
    """
    Initialisiert den Client: Lädt Scaler, Feature-Namen und erstellt den CKKS-Kontext mit Schlüsselpaar.
    Diese Funktion wird nur einmal ausgeführt und die Ergebnisse werden gecached.
    
    WICHTIG: Der Client kennt weder die Prototypen noch die Omega-Matrix des GMLVQ-Modells.
    """
    # === Schritt 1: Lade benötigte Assets vom Server ===
    print("--- CLIENT: Frage Scaler und Feature-Namen vom Server an... ---")
    # Wir ignorieren alle Modell-Parameter außer Scaler und Feature-Namen
    _, _, scaler, feature_names, _, _ = get_server_assets()

    # === Schritt 2: Generiere den Verschlüsselungs-Kontext ===
    print("--- CLIENT: Generiere CKKS-Kontext und Schlüsselpaar... ---")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    print("--- CLIENT: Umgebung initialisiert. ---")
    return scaler, context, feature_names


# ==============================================================================
# BENUTZEROBERFLÄCHE (UI)
# ==============================================================================
st.set_page_config(layout="wide", page_title="Privacy-Preserving GMLVQ")
st.title("🩺 Privacy-Preserving Heart Disease Classification (GMLVQ)")
st.markdown(
    '<p style="color:#a7a1a1;">Dies ist eine Simulation einer getrennten Client-Server-Architektur. '
    'Der <b>Client</b> (diese UI) verschlüsselt die Daten. Der <b>Server</b> (eine separate Logik) '
    'berechnet die Distanzen, ohne die Daten oder das Ergebnis zu kennen.</p>',
    unsafe_allow_html=True
)

try:
    scaler, context, feature_names = setup_client_environment()
except FileNotFoundError:
    st.error("FEHLER: Die Datensatz-Datei 'heart_data_pretty.csv' wurde auf der Serverseite nicht gefunden.")
    st.stop()

st.sidebar.header("Client: Patientendaten eingeben")
user_input = {}
default_values = [54.4, 0.68, 3.16, 131.6, 246.7, 0.14, 0.99, 149.6, 0.32, 1.0, 1.6, 0.67, 4.73]

for i, feature in enumerate(feature_names):
    user_input[feature] = st.sidebar.number_input(
        label=f"{i+1}. {feature}", value=default_values[i], step=1.0, format="%.2f"
    )

# --- Haupt-Workflow: Wird ausgeführt, wenn der Nutzer auf den Button klickt ---
if st.sidebar.button("Klassifikation durchführen", type="primary"):
    
    # === Schritt 1: CLIENT - Daten aufbereiten und verschlüsseln ===
    st.header("1. Client-Aktionen")
    patient_df = pd.DataFrame([user_input])
    st.markdown('<p style="color:#a7a1a1;"><b>Aktion:</b> Rohdaten des Patienten werden gesammelt.</p>', unsafe_allow_html=True)
    st.dataframe(patient_df)

    scaled_patient_vector = scaler.transform(patient_df)[0]
    st.markdown('<p style="color:#a7a1a1;"><b>Aktion:</b> Daten werden normiert, um für das Modell kompatibel zu sein.</p>', unsafe_allow_html=True)
    
    encrypted_patient_vector = ts.ckks_vector(context, scaled_patient_vector)
    st.markdown('<p style="color:#a7a1a1;"><b>Aktion:</b> Normierte Daten werden homomorph verschlüsselt.</p>', unsafe_allow_html=True)
    st.info("🔒 Die Patientendaten sind jetzt sicher und können das Gerät verlassen.")

    # === Schritt 2: CLIENT -> SERVER - Anfrage senden (simulierter API-Aufruf) ===
    st.header("2. Simulation der Interaktion")
    
    serialized_patient_vector = encrypted_patient_vector.serialize()
    
    context_for_server = context.copy()
    context_for_server.make_context_public()
    serialized_public_ckks_context = context_for_server.serialize()

    with st.spinner('Warte auf Antwort vom Server...'):
        time.sleep(1)
        # Der Client empfängt jetzt ZWEI Werte vom Server:
        # 1. Die verschlüsselten Ergebnisse (Distanzen und Labels)
        # 2. Die Klartext-Relevanzen für die Erklärbarkeit (wird als Säulendiagramm angezeigt)
        serialized_results_from_server, relevances = process_encrypted_request(serialized_patient_vector, serialized_public_ckks_context)
    
    st.markdown('<p style="color:#a7a1a1;">📤 <b>Client an Server:</b> Sende verschlüsselten Datenvektor und öffentlichen Kontext.</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a7a1a1;">... Server arbeitet blind auf den Daten ...</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#a7a1a1;">📥 <b>Server an Client:</b> Sende Liste von (verschlüsselten Distanzen, Klassen) UND die Merkmals-Relevanzen.</p>', unsafe_allow_html=True)

    # === Schritt 3: CLIENT - Antwort entschlüsseln und Ergebnis analysieren ===
    st.header("3. Ergebnis auf Client-Seite")
    st.markdown(
        '<p style="color:#a7a1a1;"><b>Aktion:</b> Client empfängt die Liste und entschlüsselt die Distanzen mit seinem privaten Schlüssel.</p>',
        unsafe_allow_html=True
    )
    
    secret_key = context.secret_key()
    decrypted_distances = []
    labels = []
    for ser_dist, label in serialized_results_from_server:
        enc_dist = ts.ckks_vector_from(context, ser_dist)
        dist = enc_dist.decrypt(secret_key)[0]
        decrypted_distances.append(dist)
        labels.append(label)

    min_dist_idx = int(np.argmin(decrypted_distances))
    predicted_class = labels[min_dist_idx]
    
    results_df = pd.DataFrame({
        "Prototyp-Klasse": ['GESUND' if l == 0 else 'KRANK' for l in labels],
        "Entschlüsselte Distanz (quadriert)": decrypted_distances
    }, index=np.arange(1, len(labels) + 1))
    results_df.index.name = "Prototyp-Nr."
    
    def highlight_min(row):
        return ['background-color: #636363'] * len(row) if row.name == (min_dist_idx + 1) else [''] * len(row)
    st.table(results_df.style.apply(highlight_min, axis=1))

    st.subheader("Finale Klassifikation:")
    if predicted_class == 1:
        st.error("Der Patient wird als **KRANK** eingestuft.", icon="💔")
    else:
        st.success("Der Patient wird als **GESUND** eingestuft.", icon="💚")

    
    # ==============================================================================
    # Schritt 4 - ERKLÄRBARKEIT DES GMLVQ MODELLS 
    # ==============================================================================
    st.divider()
    st.subheader("💡 Erklärbarkeit des GMLVQ-Modells")
    st.markdown(
        '<p style="color:#a7a1a1;">GMLVQ lernt, welche Merkmale für die Klassifikation wichtig sind. '
        'Das folgende Diagramm zeigt die vom Modell gelernte Wichtigkeit (Relevanz) für jedes Merkmal. '
        'Hohe Balken bedeuten, dass das Merkmal einen großen Einfluss auf das Klassifikations-Ergebnis hat.</p>',
        unsafe_allow_html=True
    )

    # 1. Bereite die empfangenen Relevanz-Daten für das Diagramm vor
    relevance_df = pd.DataFrame({
        "Merkmal": feature_names,
        "Relevanz": relevances # Diagonal-Werte aus der Lambda-Matrix
    })

    # 2. Sortiere den DataFrame, um die wichtigsten Merkmale zuerst anzuzeigen
    relevance_df = relevance_df.sort_values(by="Relevanz", ascending=False)

    # 3. Erstelle das Diagramm mit Altair-Bibliothek
    chart = alt.Chart(relevance_df).mark_bar().encode(
        x=alt.X('Merkmal', sort=None, title="Merkmal"),
        y=alt.Y('Relevanz', title="Relevanz")
    ).properties(
        title='Wichtigkeit der Merkmale'
    ).configure_title(
        color='rgb(75, 104, 159)', 
        fontSize=16    
    )

    # 4. Zeige das Diagramm in der Streamlit-App an
    st.altair_chart(chart, use_container_width=True)