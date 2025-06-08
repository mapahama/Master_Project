# === Bibliotheken importieren ===
import tenseal as ts # ckks
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# === === === === === === === === === === === === ===
#   1. DATEN LADEN UND VORBEREITEN 
# === === === === === === === === === === === === ===

print("Lade und verarbeite Heart Disease Datensatz...")
# Datensatz laden
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()

# Vorverarbeitung
X.replace('?', np.nan, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.median(), inplace=True)

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# EINEN Datapunkt zur Verschlüsselung auswählen (den ersten Patienten)
datapoint_to_encrypt = X_scaled[0]
print(f"\nDatenpunkt, der verschlüsselt wird (13 Features):\n{datapoint_to_encrypt}")
print("-" * 40)


# === === === === === === === === === === === === ===
#   2. CKKS KONFIGURIEREN UND KONTEXT ERSTELLEN
# === === === === === === === === === === === === ===
# Der "Kontext" definiert alle Parameter für die homomorphe Verschlüsselung.

print("Erstelle CKKS-Kontext...")
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

# Setzt die globale Skala für die Präzision der Gleitkommazahlen
context.global_scale = 2**40

# Generiert die notwendigen kryptographischen Schlüssel
# Der Private Key  
# Der Public Key  
context.generate_galois_keys()
secret_key = context.secret_key()
public_key = context.public_key()
print("Kontext und Schlüssel wurden generiert.")
print("-" * 40)


# === === === === === === === === === === ===
#   3. DATENPUNKT VERSCHLÜSSELN
# === === === === === === === === === === ===
# Jetzt wird der Vektor mit den 13 Features des Patienten verschlüsselt.

print("Verschlüssele den Datenpunkt...")
# ts.ckks_vector verschlüsselt einen Vektor (13 Features)
# Es wird der öffentliche Schlüssel für die Verschlüsselung verwendet
encrypted_datapoint = ts.ckks_vector(context, datapoint_to_encrypt)

print("Verschlüsselung erfolgreich!")
print(f"Typ des verschlüsselten Objekts: {type(encrypted_datapoint)}")
# Ein Blick auf den Ciphertext zeigt, dass er nicht lesbar ist
print(f"Verschlüsselte Daten (Ausschnitt): {encrypted_datapoint.serialize()[:200]}...")
print("-" * 40)


# === === === === === === === === === === ===
#   4. DATENPUNKT ENTSCHLÜSSELN (zur Überprüfung)
# === === === === === === === === === === ===
# Um zu beweisen, dass die Verschlüsselung funktioniert hat
# Hierfür ist der private Schlüssel notwendig

print("Entschlüssele den Datenpunkt zur Überprüfung...")
decrypted_datapoint = encrypted_datapoint.decrypt(secret_key)

print("Entschlüsselung erfolgreich!")
print("-" * 40)


# === === === === === === === === === === ===
#   5. ERGEBNIS VERGLEICHEN
# === === === === === === === === === === ===
# vergleichen den ursprünglichen Vektor mit dem entschlüsselten Vektor.
# Aufgrund der Natur von CKKS kann es zu  Rundungsfehlern kommen

print("VERGLEICH:")
print("Originaler Datenpunkt (erste 5 Features):")
print(np.round(datapoint_to_encrypt[:5], decimals=5))

print("\nEntschlüsselter Datenpunkt (erste 5 Features):")
print(np.round(decrypted_datapoint[:5], decimals=5))

# Überprüfung, ob die Werte sehr nahe beieinander liegen
are_close = np.allclose(datapoint_to_encrypt, decrypted_datapoint)
print(f"\nSind die originalen und entschlüsselten Werte (nahezu) identisch? -> {are_close}")