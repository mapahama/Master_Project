# 	streamlit run client.py

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
# 5. Empfangen der verschlüsselten Ergebnisse vom Server (eukl. Distanzen + Klassen-Labels).
# 6. Entschlüsseln der Ergebnisse und Treffen der finalen Klassifikationsentscheidung
# 	("gesund" oder "krank").
#
# Der Client kennt das Machine-Learning-Modell (GMLVQ) oder dessen Prototypen zu keinem Zeitpunkt.
# Die gesamte Interaktion mit dem Server erfolgt über verschlüsselte Daten.
# =================================================================================


# --- Bibliotheken importieren ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tenseal as ts
import time

# --- Server-Kommunikation simulieren ---
# Importiert die "API-Funktion" aus der server.py Datei.
# Importiert den Scaler aus server.py
# In einer echten Anwendung wäre dies ein HTTP-Request an eine URL.
from server import get_server_assets, process_encrypted_request


@st.cache_resource
def setup_client_environment():
    """
    Initialisiert den Client: Lädt Scaler, Feature-Namen und erstellt den CKKS-Kontext mit Schlüsselpaar.
    Diese Funktion wird nur einmal ausgeführt und die Ergebnisse werden gecached.
    
    WICHTIG: Der Client kennt weder die Prototypen noch die Omega-Matrix des GMLVQ-Modells.
    """
    # === Schritt 1: Lade benötigte Assets vom Server ===
    # Der Client "fragt" den Server nach den öffentlichen Assets, die er benötigt.
    print("--- CLIENT: Frage Scaler und Feature-Namen vom Server an... ---")
    _, _, scaler, feature_names, _ = get_server_assets()

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
st.set_page_config(layout="wide", page_title="Privacy-Preserving GMLVQ")
# Setzt den Haupttitel der Anwendung.
st.title("🩺 Privacy-Preserving Heart Disease Classification (GMLVQ)")
# Einleitungstext, der die Architektur erklärt.
st.write(
    "Dies ist eine Simulation einer getrennten Client-Server-Architektur. "
    "Der **Client** (diese UI) verschlüsselt die Daten. Der **Server** (eine separate Logik) "
    "berechnet die Distanzen, ohne die Daten oder das Ergebnis zu kennen."
)

# Initialisiere die Client-Umgebung.
# Der try-except-Block fängt Fehler ab, falls z.B. die CSV-Datei fehlt.
try:
    scaler, context, feature_names = setup_client_environment()
except FileNotFoundError:
    st.error("FEHLER: Die Datensatz-Datei 'heart_data_pretty.csv' wurde auf der Serverseite nicht gefunden.")
    st.stop()

# Erstellt eine Seitenleiste für die Dateneingabe durch den Nutzer.
st.sidebar.header("Client: Patientendaten eingeben")
# Initialisiert ein Dictionary, um die 13 Eingabewerte zu speichern.
user_input = {}
# Defaultwerte für die Eingabefelder, um die Nutzung zu erleichtern.
default_values = [54.4, 0.68, 3.16, 131.6, 246.7, 0.14, 0.99, 149.6, 0.32, 1.0, 1.6, 0.67, 4.73] # Nutzer können andere Werte eingeben

# Erzeugt dynamisch 13 numerische Eingabefelder in der Seitenleiste.
for i, feature in enumerate(feature_names):
    user_input[feature] = st.sidebar.number_input(
        label=f"{i+1}. {feature}", value=default_values[i], step=1.0, format="%.2f"
    )

# --- Haupt-Workflow: Wird ausgeführt, wenn der Nutzer auf den Button klickt ---
if st.sidebar.button("Klassifikation durchführen", type="primary"):
    
    # === Schritt 1: CLIENT - Daten aufbereiten und verschlüsseln ===
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

    # === Schritt 2: CLIENT -> SERVER - Anfrage senden (simulierter API-Aufruf) ===
    st.header("2. Simulation der Interaktion")
    
    # Vorbereitung für den "Versand": Die komplexen Objekte müssen serialisiert (in Bytes umgewandelt) werden.
    serialized_patient_vector = encrypted_patient_vector.serialize()
    
    # Erstelle eine öffentliche Kopie des Kontexts, die sicher an den Server gesendet werden kann.
    context_for_server = context.copy()
    context_for_server.make_context_public() # WICHTIG: Entfernt den geheimen Schlüssel aus der Kopie !!!
    serialized_public_ckks_context = context_for_server.serialize()

    # Simuliere den API-Aufruf und die Wartezeit auf eine Antwort.
    with st.spinner('Warte auf Antwort vom Server...'):
        time.sleep(1) # Simuliert Netzwerklatenz für ein realistischeres Gefühl.
        # Hier findet der eigentliche "API-Aufruf" an die Server-Logik statt.
        serialized_results_from_server = process_encrypted_request(serialized_patient_vector, serialized_public_ckks_context)
    
    st.write("📤 **Client an Server:** Sende verschlüsselten Datenvektor und öffentlichen Kontext.")
    st.write("... Server arbeitet blind auf den Daten ...")
    st.write("📥 **Server an Client:** Sende Liste von (verschlüsselten Distanzen, Prototyp-Klassen).")

    # === Schritt 3: CLIENT - Antwort entschlüsseln und Ergebnis analysieren ===
    st.header("3. Ergebnis auf Client-Seite")
    st.write("**Aktion:** Client empfängt die verschlüsselten Ergebnisse und entschlüsselt sie mit seinem privaten Schlüssel.")
    
    # Entschlüsselungsprozess:
    # Hole den geheimen Schlüssel, den nur der Client besitzt.
    secret_key = context.secret_key()
    decrypted_distances = []
    labels = []
    # Iteriere durch die vom Server erhaltene Ergebnisliste (Distanzen/Klassen).
    for ser_dist, label in serialized_results_from_server:
        # Deserialisiere die Distanz, um wieder ein funktionierendes CKKS-Vektor-Objekt zu erhalten.
        enc_dist = ts.ckks_vector_from(context, ser_dist)
        # Entschlüssle den Wert mit dem geheimen Schlüssel.
        dist = enc_dist.decrypt(secret_key)[0]
        decrypted_distances.append(dist)
        labels.append(label)

    # Finde den Index der kleinsten Distanz. Dies bestimmt die Klassenzugehörigkeit.
    min_dist_idx = int(np.argmin(decrypted_distances))
    # Wähle das Label des Prototyps mit der geringsten Distanz als finale Vorhersage.
    predicted_class = labels[min_dist_idx]
    
    # Bereite die Ergebnisse für eine übersichtliche Tabellenanzeige vor.
    results_df = pd.DataFrame({
        "Prototyp-Klasse": ['GESUND' if l == 0 else 'KRANK' for l in labels],
        "Entschlüsselte Distanz (quadriert)": decrypted_distances
    }, index=np.arange(1, len(labels) + 1)) 
    
    #Den Index benennen, damit er in der Tabelle einen Titel hat.
    results_df.index.name = "Prototyp-Nr."
    
    # Definiere eine Funktion, um die Zeile mit der minimalen Distanz farblich hervorzuheben.
    def highlight_min(row):
        # Wir vergleichen den 1-basierten Tabellen-Index (row.name) mit dem 
        # 0-basierten Ergebnis von np.argmin, zu dem wir 1 addieren.
        return ['background-color: #636363'] * len(row) if row.name == (min_dist_idx + 1) else [''] * len(row)
    # Zeige die formatierte Tabelle an.
    st.table(results_df.style.apply(highlight_min, axis=1))

    # Zeige die finale, für den Nutzer verständliche Klassifikation an.
    st.subheader("Finale Klassifikation:")
    if predicted_class == 1:
        st.error("Der Patient wird als **KRANK** eingestuft.", icon="💔")
    else:
        st.success("Der Patient wird als **GESUND** eingestuft.", icon="💚")