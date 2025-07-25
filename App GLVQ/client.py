#  .\venv311\Scripts\activate

# Start App  via      streamlit run client.py 
#
#
# =================================================================================
# CLIENT-SEITIGE LOGIK & BENUTZEROBERFLÄCHE
# =================================================================================
# Dieser Code implementiert die Client-Anwendung mit einer Streamlit-Benutzeroberfläche.
# Die Hauptaufgaben dieses Clients sind:
# 1. Bereitstellen einer UI, damit ein Nutzer seine Patientendaten eingeben kann.
# 2. Vorbereiten (Skalieren) dieser Daten, damit sie für das Modell des Servers
#    verständlich sind.
# 3. Erzeugen eines Verschlüsselungskontexts (CKKS) und Halten des geheimen Schlüssels.
# 4. Verschlüsseln der Patientendaten und Senden der echten HTTP-Anfrage an den FastAPI-Server.
#    Zur Client-Server Kommunikation verwenden wir JSON-Objekte.
#    Sie können jedoch keine rohen Binärdaten (bytes) verarbeiten , 
#    die durch die Serialisierung der TenSEAL-Objekte entstehen.
#    Um dies zu beheben, müssen wir die Binärdaten in eine JSON-kompatible Zeichenkette umwandeln. 
#    Wir verwenden dafür **Base64-Kodierung**
#
# 5. Empfangen der verschlüsselten Ergebnisse vom Server (eukl. Distanzen + Klassen-Labels).
# 6. Entschlüsseln der Ergebnisse und Treffen der finalen Klassifikationsentscheidung
#    ("gesund" oder "krank").
#
# Der Client kennt das Machine-Learning-Modell oder dessen Prototypen zu keinem Zeitpunkt.
# Die gesamte Interaktion mit dem Server erfolgt über verschlüsselte Daten.
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tenseal as ts
import time
import plotly.express as px
import requests
import base64 # Neu: Import für die Base64-Kodierung

# --- Server-Kommunikation simulieren ---
@st.cache_resource
def setup_client_environment():
    """
    Bereitet die client-seitige Umgebung vor, indem der Scaler und Feature-Namen
    vom Server angefordert werden. Auch der CKKS-Kontext wird hier generiert.

    WICHTIG: Der Client kennt das Modell oder die Prototypen NICHT.
    Er benötigt nur den Scaler, um Daten korrekt vorzubereiten, und den CKKS-Kontext
    für die Ver- und Entschlüsselung.
    
    """

    # === Schritt 1: Lade benötigte Assets vom Server ===
    # Der Client "fragt" den FastAPI-Server durch eine echte HTTP-Anfrage  nach den öffentlichen Assets, die er benötigt.
    # Hier sind das der gefittete Scaler und die Namen der Features für die UI.
    # Die Prototypen und ihre Labels werden bewusst ignoriert (mit _), da der
    # Client sie nicht kennen darf.
    print("--- CLIENT: Frage Scaler und Feature-Namen vom Server an... ---")
    from server_fastapi import get_server_assets
    _, _, scaler, feature_names = get_server_assets()

    # === Schritt 2: Generiere den Verschlüsselungs-Kontext ===
    # Dieser Kontext enthält alle Parameter für die CKKS-Verschlüsselung.
    print("--- CLIENT: Generiere CKKS-Kontext und Schlüsselpaar... ---")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    # Setzt den globalen Skalierungsfaktor für die Präzision der Berechnungen.
    context.global_scale = 2**40
    # Generiert die öffentlichen Galois-Schlüssel, die der Server für Operationen wie .sum() benötigt.
    context.generate_galois_keys()
    print("--- CLIENT: Umgebung initialisiert. ---")

    # Gib die für den Client notwendigen Objekte zurück.
    return scaler, context, feature_names

# ==============================================================================
# BENUTZEROBERFLÄCHE (UI)
# ==============================================================================
# Konfiguriert die grundlegenden Seiteneinstellungen der Streamlit-App.
st.set_page_config(layout="wide", page_title="Privacy-Preserving GLVQ")
# Setzt den Haupttitel der Anwendung.
st.title("🩺 Privacy-Preserving Heart Disease Classification")
# # Einleitungstext, der die Architektur erklärt.
st.write(
    "Dies ist eine  getrennte Client-Server-Architektur. "
    "Der **Client** (diese UI) verschlüsselt die Daten und sendet sie über eine HTTP-Anfrage an einen separaten FastAPI-Server."
    "Der **Server** berechnet die Distanzen, ohne die Daten oder das Ergebnis zu kennen."
)

#  Initialisiere die Client-Umgebung.
try:
    scaler, context, feature_names = setup_client_environment()
except FileNotFoundError:
    st.error("FEHLER: Kommunikation mit Server fehlgeschlagen")
    st.stop()

# Erstellt eine Seitenleiste für die Dateneingabe durch den Nutze    
st.sidebar.header("Client: Patientendaten eingeben")
# Initialisiert ein Dictionary, um die 13 Eingabewerte zu speichern
user_input = {}
# Defaultwerte für die Eingabefelder, um die Nutzung zu erleichter
default_values = [54.4, 0.68, 3.16, 131.6, 246.7, 0.14, 0.99, 149.6, 0.32, 1.0, 1.6, 0.67, 4.73]

# Erzeugt dynamisch 13 numerische Eingabefelder in der Seitenleist
for i, feature in enumerate(feature_names):
    user_input[feature] = st.sidebar.number_input(
        label=f"{i+1}. {feature}", value=default_values[i], step=1.0, format="%.2f"
    )

# --- Haupt-Workflow: Wird ausgeführt, wenn der Nutzer auf den Button klickt --
if st.sidebar.button("Klassifikation durchführen", type="primary"):
    
    # === Schritt 1: CLIENT - Daten aufbereiten und verschlüsseln ==
    st.divider() # Abstand
    st.header("1. Client-Aktionen")
    # Wandelt die Nutzereingaben in einen Pandas DataFrame um.
    patient_df = pd.DataFrame([user_input])
    st.write("**Aktion:** Rohdaten des Patienten werden gesammelt.")
    st.dataframe(patient_df)

    # Skaliere die Eingabedaten mit dem vom Server erhaltenen, fertig trainierten Scaler.
    scaled_patient_vector = scaler.transform(patient_df)[0]
    st.write("**Aktion:** Daten werden normiert, um für das Modell kompatibel zu sein.")

    # Verschlüssele den normierten Patienten-Vektor mit dem CKKS-Kontext des Clients.
    encrypted_patient_vector = ts.ckks_vector(context, scaled_patient_vector)
    st.write("**Aktion:** Normierte Daten werden homomorph verschlüsselt.")
    st.info("🔒 Die Patientendaten sind jetzt sicher und können das Gerät verlassen.")
    
    # === Schritt 2: CLIENT -> SERVER - Anfrage senden (HTTP-Anfrage an FastAPI-Server) ==
    st.divider() # Abstand
    st.header("2. HTTP-Anfrage an FastAPI-Server")
    
    # !! Kodierung der Binärdaten in Base64-Strings  
    #  der Client sendet eine echte HTTP-Anfrage an den FastAPI-Server
    #  JSON-Objekte können keine rohen Binärdaten (bytes) verarbeiten , die durch die Serialisierung der TenSEAL-Objekte entstehen.
    #  Um dies zu beheben, müssen wir die Binärdaten in eine JSON-kompatible Zeichenkette umwandeln. 
    #  Wir verwenden dafür Base64-Kodierung

    # Vorbereitung für den "Versand": Die komplexen Objekte müssen serialisiert (in Bytes umgewandelt) werden.
    # Danach Byte64-Kodierung für eine JSON-kompatible Zeichenkette
    serialized_patient_vector_b64 = base64.b64encode(encrypted_patient_vector.serialize()).decode('utf-8')
    context_for_server = context.copy() #  # Erstelle eine öffentliche Kopie des Kontexts, die sicher an den Server gesendet werden kann.
            # Wir erstellen einen öffentlichen Kontext, der die riesigen Galois-Schlüssel und den Privaten Schlüssel NICHT enthält.
            # Das macht die Serialisierung und den Transfer wesentlich schneller. 
    context_for_server.make_context_public(generate_galois_keys=False)  # Private Key und Galois keys entfernen!!!
    serialized_public_ckks_context_b64 = base64.b64encode(context_for_server.serialize()).decode('utf-8')

    payload = {
        "serialized_encrypted_patient_vector": serialized_patient_vector_b64,
        "serialized_public_ckks_context": serialized_public_ckks_context_b64
    }
    
    # --- ECHTE ZEITMESSUNG BEGINNT HIER ---
    start_zeit = time.time()
    
    with st.spinner('Warte auf Antwort vom Server...'):
        try:
            # HTTP Anfrage an den FastAPI-Server
            response = requests.post("http://localhost:8000/classify", json=payload)
            response.raise_for_status()
            
            # Dekodierung der Base64-Strings aus der Server-Antwort
            response_json = response.json()
            serialized_results_from_server = []
            for ser_dist_b64, label in response_json["serialized_results"]:
                serialized_results_from_server.append(
                    (base64.b64decode(ser_dist_b64), label)
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Fehler bei der Server-Kommunikation: {e}")
            st.stop()
            
    # --- ECHTE ZEITMESSUNG ENDET HIER ---
    end_zeit = time.time()
    verstrichene_zeit = end_zeit - start_zeit
    st.info(f"Gesamtzeit für Client-Server-Kommunikation: **{verstrichene_zeit:.6f}** Sekunden.")
    
    st.write("📤 **Client an Server:** Sende verschlüsselten Datenvektor und öffentlichen Kontext.")
    st.write("... Server arbeitet blind auf den Daten ...")
    st.write("📥 **Server an Client:** Sende Liste von (verschlüsselten Distanzen, Prototyp-Klassen).")

    # === Schritt 3: CLIENT - Antwort entschlüsseln und Ergebnis analysieren ===
    st.divider()
    st.header("3. Ergebnis auf Client-Seite")
    st.write("**Aktion:** Client empfängt die verschlüsselten Ergebnisse und entschlüsselt sie mit seinem privaten Schlüssel.")
    
    # Entschlüsselungsprozess:
    # Hole den geheimen Schlüssel, den nur der Client besitzt.
    secret_key = context.secret_key()
    decrypted_distances = []
    labels = []
    # Iteriere durch die vom Server erhaltene Ergebnisliste (Distanzen/Klassen).
    for ser_dist, label in serialized_results_from_server:
        enc_dist = ts.ckks_vector_from(context, ser_dist)
        # Entschlüssle den Wert mit dem Private Key.
        decrypted_dist = enc_dist.decrypt(secret_key)[0]
        decrypted_distances.append(decrypted_dist)
        labels.append(label)
    
    # Finde den Index der kleinsten Distanz. Dies bestimmt die Klassenzugehörigkeit.
    min_dist_idx = np.argmin(decrypted_distances)
    # Wähle das Label des Prototyps mit der geringsten Distanz als finale Vorhersage.
    predicted_class = labels[min_dist_idx]
    
    # Bereite die Ergebnisse für eine übersichtliche Tabellenanzeige vor.
    results_df = pd.DataFrame({
        "Prototyp-Klasse": ['GESUND' if l == 0 else 'KRANK' for l in labels],
        "Entschlüsselte Distanz (quadriert)": decrypted_distances
    }, index=np.arange(1, len(labels) + 1))

    # Den Index benennen, damit er in der Tabelle einen Titel hat.
    results_df.index.name = "Prototyp-Nr."
    
    # Definiere eine Funktion, um die Zeile mit der minimalen Distanz farblich hervorzuheben.
    def highlight_min(row):
        return ['background-color: #636363'] * len(row) if row.name == (min_dist_idx + 1) else [''] * len(row)
    # Zeige die formatierte Tabelle an.
    st.table(results_df.style.apply(highlight_min, axis=1))
    
    # Zeige die finale, für den Nutzer verständliche Klassifikation an.
    st.subheader("Finale Klassifikation:")
    if predicted_class == 1:
        st.error("Der Patient wird als **KRANK** eingestuft.", icon="💔")
    else:
        st.success("Der Patient wird als **GESUND** eingestuft.", icon="💚")
    

    #################################
    # Balkendiagramm  / Erklärbarkeit
    #################################

    # === Schritt 4: GRAFISCHE DARSTELLUNG DES ERGEBNISSES  ===
    st.divider() # Abstand
    st.subheader("📊 Grafischer Vergleich der Distanzen")
    st.write("Jeder Balken repräsentiert einen Prototyp des Modells. "
             "Die Klassifikation basiert auf dem Prototyp mit der kürzesten Distanz.")
    
    # Bereite die Daten für das Diagramm vor
    plot_df = pd.DataFrame({
        "Prototyp-Nr.": [f"Proto {i+1}" for i in range(len(labels))],
        "Klasse": ["GESUND" if l == 0 else "KRANK" for l in labels],
        "Quadrierte Distanz": decrypted_distances
    })

    # 4.1 Definiere alle möglichen Zustände und ihre Farben
    color_map = {
        'GESUND': 'mediumseagreen',
        'KRANK': 'indianred',
        'GEWINNER (GESUND)': 'darkgreen', # Dunkleres Grün für den Gewinner (Gesund)
        'GEWINNER (KRANK)': 'darkred'     # Dunkleres Rot für den Gewinner (Krank)
    }

    # 4.2 Erstelle eine neue Spalte, die den Zustand für die Legende und Farbe steuert.
    #    Fülle sie zuerst mit den Basis-Klassen ('GESUND' oder 'KRANK').
    plot_df['Legenden-Kategorie'] = plot_df['Klasse']

    # 4.3 Bestimme die spezielle Kategorie für den Gewinner-Prototyp
    winning_class_label = plot_df.loc[min_dist_idx, 'Klasse']
    winning_category = f'GEWINNER ({winning_class_label})'

    # 4.4 Setze diese spezielle Kategorie für die Gewinner-Zeile im DataFrame.
    #    Wir verwenden .loc mit dem Index-Wert, um die richtige Zeile zu ändern.
    plot_df.loc[min_dist_idx, 'Legenden-Kategorie'] = winning_category

    # Erstelle das Balkendiagramm mit Plotly, basierend auf der neuen Kategorie-Spalte.
    fig = px.bar(
        plot_df,
        x="Prototyp-Nr.",
        y="Quadrierte Distanz",
        color='Legenden-Kategorie',
        color_discrete_map=color_map, # Weist  Farbpalette zu
        title="Vergleich der Distanzen zu allen Prototypen"
    )
    fig.update_layout(legend_title_text='Status')

    # Zwinge die X-Achse, die Reihenfolge aus dem DataFrame beizubehalten
    fig.update_xaxes(categoryorder='array', categoryarray=plot_df['Prototyp-Nr.'])
    # Zeige die interaktive Grafik in Streamlit an
    st.plotly_chart(fig, use_container_width=True)