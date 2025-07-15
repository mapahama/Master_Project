
# =================================================================================
# SERVER-SEITIGE LOGIK FÜR GMLVQ (homomorph mit CKKS)
# =================================================================================
# Der Server führt  die komplette homomorphe Klassifikation (blind) durch,
# inklusive der Bestimmung der minimalen Distanz und des zugehörigen Labels
# mittels der "argmin"-Turniermethode und polynomiale Approximation der Sign Funktion (Distanzen vergleichen -> min finden)
# =================================================================================
# HINWEIS: ! Serialisierung/Deserialisierung wurde entfernt (wegen Performanz - es ist ein Proof of Concept)
#          ! Zu keinem Zeitpunkt kennt der Server die wahren Werte der Patientendaten.
#          ! Zu keinem Zeitpunkt kennt der Server Private Key des Clients
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import tenseal as ts
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn_lvq import GmlvqModel
from sklearn.ensemble import IsolationForest



# =================================================================================
# Bereich 1: Globale Konfiguration und Laden der Modell-Assets
# =================================================================================

# Die homomorphe Signum-Approximation funktioniert am besten im Wertebereich [-1, 1]
# Daher sollen die Eingaben in diesem Wertebereich sein
# Wir normalisieren die verschlüsselten Distanzen  mit:   enc(distanz) * (1 / MAX_EXPECTED_SQUARED_DISTANCE)
MAX_EXPECTED_SQUARED_DISTANCE = 5.0 # TODO: dynamisch machen

# Diese Funktion lädt alle notwendigen Server-Assets. Der Decorator sorgt dafür,
# dass dieser aufwändige Prozess (Daten laden, Modell trainieren) nur einmal beim
# allerersten Start der Anwendung ausgeführt wird. Das Ergebnis wird im Cache gespeichert. (zum Proof of Concept, da die Anwendung nicht deployed ist)
@st.cache_resource


def get_server_assets():
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Diese Funktion simuliert den einmaligen Setup-Prozess auf dem Server.
    Sie ist dafür verantwortlich, den Datensatz zu laden, ihn vorzubereiten und das GMLVQ-Modell
    zu trainieren. Die resultierenden Modellparameter (Prototypen, Relevanzen etc.) werden
    zurückgegeben und für alle zukünftigen Anfragen wiederverwendet.

    Returns:
        tuple: Ein Tupel, das die trainierten Prototypen, ihre Labels, den Skalierer,
               die Feature-Namen, die Omega-Matrix und die Feature-Relevanzen enthält.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    print("--- SERVER: Lade Datensatz für das einmalige Training... ---")
    df = pd.read_csv("../heart_data_pretty.csv", sep='\s+') # Dataset laden - in einen Pandas DataFrame.
    X_full = df.drop(columns=["target"]).copy()
    y = (df["target"] > 0).astype(int)
    # GMLVQ (in diesem Code) erwartet Labels als -1 und 1, daher wird 0 (gesund) durch -1 (gesund) ersetzt.
    y.replace(0, -1, inplace=True) 
    
    X = X_full.copy()   # X = X_full.iloc[:, :6].copy()   
    feature_names = X.columns.tolist()
    
    # --- Datenbereinigung und Skalierung ---
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

    # Initialisiert einen Skalierer, der alle Feature-Werte in den Bereich [-1, 1] transformiert.
    # Dies ist wichtig für die Stabilität des GMLVQ-Trainings und die Genauigkeit der sgn-Approximation. (erwartet Werte in [-1,1])
    scaler = MinMaxScaler(feature_range=(-1, 1))   # Alle Features in einem kleinen Wertebereich wegen sign Funk 
    X_scaled = scaler.fit_transform(X_clean)
    y_binary_np = y_clean.to_numpy()
    print(f"--- SERVER: Trainiere GMLVQ-Modell mit {len(feature_names)} Features... ---")

    # Initialisiert das GMLVQ-Modell mit spezifischen Hyperparametern.  (Ausgewählt durch Cross Validation)
    gmlvq = GmlvqModel(prototypes_per_class=2, regularization=0.35, gtol=1e-5, random_state=42)
    gmlvq.fit(X_scaled, y_binary_np)
    # --- Extraktion der gelernten Modellparameter nach dem Training ---
    prototypes = gmlvq.w_     # Die Positionen der gelernten Prototypen im Feature-Raum.
    proto_labels = gmlvq.c_w_ # Die zugehörigen Klassenlabels für jeden Prototyp (-1 oder 1).
    omega = gmlvq.omega_      # !!!Die gelernte Relevanzmatrix (gibt an, wie Features zu transformieren sind)
    lambda_matrix = omega.T @ omega  # !!!Die quadratische Relevanzmatrix, deren Diagonale die Wichtigkeit der Features angibt.
    relevances = np.diag(lambda_matrix)  # Extrahiert diese Wichtigkeiten (Feature-Relevanzen)  - wird im UI angezeigt (Interpretierbarkeit der Ergebnisse)
    print("--- SERVER: GMLVQ-Assets und Relevanzen geladen und bereit. ---")
    return prototypes, proto_labels, scaler, feature_names, omega, relevances

# =================================================================================
# Bereich 2: Polynom-Approximation der Signum-Funktion
# =================================================================================

# Grad des Tschebyscheff-Polynoms. 7 (getestet) ist Kompromiss zwischen Genauigkeit und Rechenaufwand
CHEBYSHEV_DEGREE = 7 # Ein höherer Grad -> genauere Approximation, erhöht aber auch Multiplikationstiefe, Rauschen und Berechnungszeit in CKKS
# Erstellt ein Array von 2000 gleichmäßig verteilten Zahlen im Intervall [-1, 1].
# Dies ist der Bereich, in dem unsere Approximation der Signum-Funktion trainiert (gut) sein muss.
x_vals = np.linspace(-1, 1, 2000) 

# Berechnet die -wahren- Werte der Signum-Funktion für jeden Wert in x_vals.
# Das Ergebnis ist -1 für negative, +1 für positive und 0 für x=0.
sign_vals = np.sign(x_vals) #  Diese Werte sind das, was unser Polynom lernen soll zu imitieren.

# !!! Polynomfit im Tschebyscheff-System !!!
# Die Methode (Chebyshev.fit) findet das bestmögliche Polynom vom Grad CHEBYSHEV_DEGREE, ,
#  das die durch (x_vals, sign_vals) gegebenen Punkte beschreibt.
cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_vals, sign_vals, CHEBYSHEV_DEGREE) # das gefundene Tschebyscheff Polynom
monomial_coeffs = cheb_poly.convert(kind=np.polynomial.Polynomial).coef #konvertiert das  gefundene Tschebyscheff-Polynom (cheb_poly) in  Standardform und extrahiert  Koeffizienten (.coef).
print("monomial_coeffs: ", monomial_coeffs) 
# zum Testen
# monomial_coeffs = [0, 5.38063021, 0, -1.97093792, 0, 3.07160644, 0, -1.56558738] # <-- Höhste Treffquote Grad 7


# =================================================================================
# Bereich 3: Homomorphe Helper-Funktionen  (CKKS Werte vergleichen und min Wert finden)
# =================================================================================

def homomorphic_sgn(a_minus_b):
    """
    Verwendet Tschebyscheff Polynom zur Approximierung der Sign Funktion
    Args:
        a_minus_b (ts.CKKSVector): Ein verschlüsselter Vektor, der die Differenz zweier Werte enthält.
    Returns:
        ts.CKKSVector: Das verschlüsselte Ergebnis der Polynomauswertung. 
    """
    print("In Server - Funktion - homomorphic_sgn")
    coeffs = monomial_coeffs 
    return a_minus_b.polyval(coeffs) #(bei -1  a ist kleiner als b,  bei +1 a ist größer als b)


def homomorphic_min_with_label(enc_val_a, enc_label_a, enc_val_b, enc_label_b):
    """
    Findet homomorph das ->Minimum<- von zwei verschlüsselten Distanzen (a, b) und gibt
    das zugehörige Label zurück. Wird in der Tournier-Methode verwendet.
    !!! Implementiert die Formel: min(a,b) = 0.5 * (a+b - sgn(a-b)*(a-b))
    
    Args:
        enc_val_a (ts.CKKSVector): Der erste verschlüsselte Wert.
        enc_label_a (ts.CKKSVector): Das zum ersten Wert gehörende verschlüsselte Label.
        enc_val_b (ts.CKKSVector): Der zweite verschlüsselte Wert.
        enc_label_b (ts.CKKSVector): Das zum zweiten Wert gehörende verschlüsselte Label.
    Returns:
        tuple: Ein Tupel mit (verschlüsseltes Minimum, verschlüsseltes zugehöriges Label).
    """
    print("In Server - Funktion - homomorphic_min_with_label")

    val_a_minus_b = enc_val_a - enc_val_b
    sgn_a_minus_b = homomorphic_sgn(val_a_minus_b) # Wendet die homomorphe Signum-Approximation an, um sgn(a-b) zu erhalten.
    val_a_plus_b = enc_val_a + enc_val_b
    # Minimum-Formel anwenden: min(a,b) = 0.5 * (a+b - sgn(a-b)*(a-b))
    min_val = (val_a_plus_b - (sgn_a_minus_b * val_a_minus_b)) * 0.5 # Alle Operationen sind homomorph
    
    # Führt  die gleiche  Operation auf den Labels durch.
    # Dadurch wird asch das korrekte Label ausgewählt, das zum Minimum gehört
    label_a_plus_b = enc_label_a + enc_label_b
    label_a_minus_b = enc_label_a - enc_label_b
    min_label = (label_a_plus_b - (sgn_a_minus_b * label_a_minus_b)) * 0.5

    return min_val, min_label   # gibt das Paar aus minimaler Distanz und zugehörigem Label zurück

def find_min_and_label_tournament(encrypted_pairs):
    """
    Führt ein "Turnier" durch, um das Paar (Distanz, Label) mit der kleinsten
    Distanz aus einer Liste zu finden. Vergleicht rekursiv Paare, bis ein Sieger übrig bleibt (getestet: reduziert Multiplikations-Tiefe !!!)
    Args:
        encrypted_pairs (list): Eine Liste von Tupeln, wobei jedes Tupel ein
                                (verschlüsselte Distanz, verschlüsseltes Label) ist.
    Returns:
        tuple: Das Gewinner-Tupel (verschlüsselte kleinste Distanz, verschlüsseltes zugehöriges Label)
    """
    print("In Server - Funktion - find_min_and_label_tournament")
    # Startet mit der vollen Liste von Paaren.
    current_level = encrypted_pairs
    # Fall 1: Die Liste ist leer, es gibt nichts zu tun
    if not current_level:
        return None, None
    # Fall 2: Die Liste hat nur ein Element, es ist automatisch der Gewinner.
    if len(current_level) == 1:
        return current_level[0]

    # Führt so lange Runden durch, bis nur noch ein Element in der Liste ist
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                val1, label1 = current_level[i]
                val2, label2 = current_level[i+1]
                # führe den homomorphen Vergleich durch, um den Gewinner zu ermitteln
                winner_val, winner_label = homomorphic_min_with_label(val1, label1, val2, label2)
                # füge den Gewinner der nächsten Runde hinzu.
                next_level.append((winner_val, winner_label))

            # Wenn ein Element am Ende übrig bleibt (ungerade Anzahl), kommt es automatisch eine Runde weiter.
            else:
                next_level.append(current_level[i])
        # Die Gewinner der aktuellen Runde werden zu den Teilnehmern der nächsten Runde
        current_level = next_level

    return current_level[0] # Gesamtsieger des Turniers (min Distanz und ihrer Label)

# =================================================================================
# Bereich 4: Haupt-Verarbeitungsfunktion
# =================================================================================

def process_encrypted_request(encrypted_patient_vector, public_context):
    """
    Dies ist die zentrale Funktion des Servers. Sie nimmt eine verschlüsselte Anfrage
    vom Client entgegen und orchestriert die gesamte homomorphe Klassifikation.
    Args:
        encrypted_patient_vector (ts.CKKSVector): Der verschlüsselte Vektor mit den Patientendaten.
        public_context (ts.Context): Der öffentliche CKKS-Kontext vom Client (ohne privaten Schlüssel und Galois Keys).
    Returns:
        tuple: Ein Tupel mit Ergebnissen für den Client, inkl. dem verschlüsselten Gewinner-Label.
    """
    print("In Server - Funktion - process_encrypted_request")

    prototypes, proto_labels, _, _, omega, relevances = get_server_assets() # Lädt die trainierten Modell-Assets (Prototypen, Omega-Matrix etc.)

    # --- Schritt 1: Homomorphe Distanzberechnung  ---
    print("--- SERVER: Schritt 1 - Berechne alle Distanzen... ---")
    # Projiziert den verschlüsselten Patientenvektor mit der Relevanzmatrix Omega
    # Dies ist der erste Schritt der Distanzberechnung
    encrypted_embedded = [encrypted_patient_vector.dot(omega_row.tolist()) for omega_row in omega]
    encrypted_distances = []
    # Iteriert durch jeden  Prototyp, den der Server kennt
    for proto in prototypes:
        embedded_proto = omega @ proto # Projiziert den  Prototyp ebenfalls mit der Omega-Matrix
        # Berechnet die Differenz zwischen dem projizierten Patientenvektor und dem projizierten GMLVQ-Prototyp
        # Dies ist eine Operation zwischen einem verschlüsselten und einem klaren Vektor !
        diff_enc = [enc_val - p_val for enc_val, p_val in zip(encrypted_embedded, embedded_proto)]
        sq_diff = [x.pow(2) for x in diff_enc] # Distanzen der Features quadrieren
        enc_distance = sum(sq_diff) # alle quadrierten Feature-Distanzen summieren
        encrypted_distances.append(enc_distance) # Fügt die verschlüsselte Distanz zur Ergebnisliste hinzu
    print("--- SERVER: Homomorphe Distanzberechnung abgeschlossen. ---")

    # --- Schritt 2: NEU - Tiefen-optimierte Klassifikation ---
    print("--- SERVER: Schritt 2 - Starte tiefen-optimierte Klassifikation... ---")

    # a) Normalisiere Distanzen, um sie in den optimalen Bereich [-1, 1] der sgn-Approximation zu bringen.
    norm_factor = 1.0 / MAX_EXPECTED_SQUARED_DISTANCE
    normalized_distances = [d * norm_factor for d in encrypted_distances]

    # b) Gruppiere Distanzen und ihre Labels nach Klasse
    # Klasse -1 = gesund, Klasse 1 = krank
    class_minus_1_distances = []
    class_1_distances = []
    for dist, label in zip(normalized_distances, proto_labels):
        if label == -1: # gesund
            class_minus_1_distances.append(dist)
        else: # krank
            class_1_distances.append(dist)

    # c) Stufe 1 des Turniers: Finde die minimale Distanz FÜR JEDE KLASSE separat.
    # Wir erstellen Paare aus (Distanz, Klassenlabel), um das Turnier zu nutzen.
    # Das Label ist hier ein Skalar (-1 oder 1)  #  One-Hot-Vektor getestet -> erhöht Multiplikationstiefe
    print("--- SERVER: Finde Minimum für Klasse -1 ('gesund')... ---")
    pairs_class_minus_1 = [(dist, ts.ckks_vector(public_context, [-1])) for dist in class_minus_1_distances]
    min_dist_c_minus_1, label_c_minus_1 = find_min_and_label_tournament(pairs_class_minus_1)

    print("--- SERVER: Finde Minimum für Klasse 1 ('krank')... ---")
    pairs_class_1 = [(dist, ts.ckks_vector(public_context, [1])) for dist in class_1_distances]
    min_dist_c1, label_c1 = find_min_and_label_tournament(pairs_class_1)

    # d) Stufe 2 des Turniers: Vergleiche die beiden Klassen-Gewinner und bestimme das finale Label.
    # Behandelt den Fall, dass eine Klasse keine Prototypen hat
    if min_dist_c_minus_1 is None:
        enc_winner_label = label_c1
    elif min_dist_c1 is None:
        enc_winner_label = label_c_minus_1
    else:
        # Führt den finalen ((entscheidenden)) Vergleich zwischen dem besten "gesund"- und dem besten "krank"-Prototyp durch
        print("--- SERVER: Führe finalen Vergleich der Klassen-Minima durch... ---")
        enc_winner_val, enc_winner_label = homomorphic_min_with_label(min_dist_c_minus_1, label_c_minus_1, min_dist_c1, label_c1)
    
    print("--- SERVER: Homomorphe Klassifikation abgeschlossen. ---")

    # --- Schritt 3: Ergebnisse für den Client vorbereiten ---
    # Gibt alle relevanten Informationen an den Client zurück. Das wichtigste ist `enc_winner_label`. ((also die  Klassifikation))
    return encrypted_distances, proto_labels.tolist(), enc_winner_label, enc_winner_val, relevances.tolist()
