

#   .\venv311\Scripts\activate

#3  turnier methode
# =============================================================================
# Ziel des Codes: Demonstration der Minimum-Findung auf verschlüsselten Daten
#
# Was der Code macht:
# Dieser Code implementiert eine Funktion zur Bestimmung des Minimums einer Liste
# von Zahlen, die mit dem homomorphen Verschlüsselungsverfahren CKKS verschlüsselt sind.
# Da CKKS nur Addition und Multiplikation unterstützt, kann eine direkte
# Minimum-Funktion nicht trivial implementiert werden.
#
# Methode:
# 1. Mathematische Umformung: Die Funktion min(a, b) wird durch eine rein
#    polynomielle Formel ausgedrückt: min(a, b) = 0.5 * [a + b - sgn(a - b) * (a - b)].
# 2. Approximation der Vorzeichenfunktion (sgn): Die nicht-polynomielle
#    Vorzeichenfunktion sgn(x) wird durch ein Polynom 7. Grades angenähert.
#    Diese Approximation ist der rechenintensivste und ungenaueste Teil.
# 3. Iterative Anwendung: Der Code verschlüsselt eine Liste von Zahlen und
#    wendet die homomorphe Minimum-Funktion iterativ an, um das kleinste
#    Element der gesamten Liste zu finden. Durch die !!! Tourniermethode !!! kann eine kleinere Multiplikationstiefe verwendet werden.
#
# Demonstrationszweck:
# Der Code dient als Machbarkeitsbeweis (Proof of Concept) dafür, wie komplexe
# nicht-lineare Funktionen (wie ein Vergleich) im Rahmen der homomorphen
# Verschlüsselung durch polynomiale Approximationen realisiert werden können.
#
# =============================================================================


import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

################ Start Polynom ####################

# 1. Tschebyscheff-Approximation der Signumfunktion
CHEBYSHEV_DEGREE = 7 # Grad des Tschebyscheff-Polynoms. Ein höherer Grad -> genauere Approximation, erhöht aber auch Multiplikationstiefe, Rauschen und Berechnungszeit in CKKS
x_vals = np.linspace(-1, 1, 2000) # Erstellt ein Array von 2000 gleichmäßig verteilten Zahlen im Intervall von -1 bis +1

sign_vals = np.sign(x_vals) #  Berechnet Signumfunktion (Vorzeichenfunktion) für jeden Wert in x_vals
                            #. Es wird ausgegeben: für negative x-Werte -1, für positive x-Werte +1 und für x=0 den Wert 0 
                            #  Diese Werte sind das, was unser Polynom lernen soll zu imitieren.

# !!! Polynomfit im Tschebyscheff-System !!!
# Die Methode (Chebyshev.fit) findet das bestmögliche Polynom vom Grad CHEBYSHEV_DEGREE,
#  das die durch x_vals und sign_vals gegebenen Punkte beschreibt.
cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_vals, sign_vals, CHEBYSHEV_DEGREE) # das gefundene Tschebyscheff Polynom
monomial_coeffs = cheb_poly.convert(kind=np.polynomial.Polynomial).coef #konvertiert das  gefundene Tschebyscheff-Polynom (cheb_poly) in  Standardform und extrahiert  Koeffizienten (.coef).
print("monomial_coeffs: ", monomial_coeffs)
# Output:  [ 1.92753281e-16  3.32927197e+00 -2.04970584e-15 -4.80418856e+00 1.88666005e-15  3.23961423e+00 -5.34773567e-16 -9.35694607e-01 4.92353972e-17  9.59289761e-02]


# ================================================================
# ANZEIGE DES POLYNOMS
# ================================================================
print("--- Approximierendes Polynom (gerundet) ---")

# Erzeugt ein neues Polynom-Objekt in der Standard-Basis zur Anzeige
standard_poly = cheb_poly.convert(kind=np.polynomial.Polynomial)
# Gibt das Polynom in einer lesbaren Form auf der Konsole aus 
print(standard_poly)
print("\n")
# ================================================================


# 2. Visualisierung der Approximation in einem Diagramm
plt.figure(figsize=(10, 6))
plt.plot(x_vals, sign_vals, "--", label="sign(x)") # Zeichnet die originale Signumfunktion
plt.plot(x_vals, cheb_poly(x_vals), label=f"Chebyshev Approx (deg={CHEBYSHEV_DEGREE})") # Zeichnet unser approximiertes Polynom
plt.legend(); plt.title("Tschebyscheff-Approximation der Signumfunktion"); plt.grid(True)
plt.ylim(-1.5, 1.5); plt.show()

max_error = np.max(np.abs(sign_vals - cheb_poly(x_vals)))
print(f"Maximaler Approximationsfehler: {max_error:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(x_vals, np.abs(sign_vals - cheb_poly(x_vals)), color="red", label="|sign(x) - approx(x)|")
plt.title("Approximationsfehler des Tschebyscheff-Polynomans")
plt.xlabel("x"); plt.ylabel("Fehler")
plt.grid(True); plt.legend(); plt.show()
########### Ende Polynom ##############





context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384 ,# 32768, # 8192 ->error  # Grad des Polynommoduls. Ein höherer Grad erhöht die Sicherheit und die Kapazität des Ciphertextes, verlangsamt aber die Berechnung.
        # Kette von Primzahlen-Moduln - Multiplikationstiefe. Wen mehrere Moduln, dann langsamere Berechnung. 
        # Erlaubt aber mehrere Multiplikationen. 
        # Ziel: möglichst kleine aber ausreichende für die Multiplikationen Tiefe, sonst Error: scale out of bounds
    coeff_mod_bit_sizes=[50] + [25]*10 + [50] #  Kette von Primzahlen-Moduln. 
)

# Galois Keys. Generiert spezielle Schlüssel (Galois-Schlüssel), die für Rotationen in Vektoren benötigt werden.
# Die Funktion .polyval() nutzt diese intern, um die Potenzen von x (x^2, x^3, ...) effizient zu berechnen.
context.generate_galois_keys()
# Setzt den globalen Skalierungsfaktor. Dies ist ein entscheidender Parameter bei CKKS, der die Genauigkeit der Fließkommazahlen bestimmt. 2^40 ist ein üblicher hoher Wert für gute Präzision.
context.global_scale = 2**25
#context.auto_rescale = True # nicht nötig !

# liste im Bereich [0, 1] normieren, damit Differenz von 2 Zahlen im Bereich [-1, 1] liegt
def min_max_scale(values):
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.0 for _ in values]  # alle Werte gleich, daher 0
    return [(v - min_v) / (max_v - min_v) for v in values] 
    #return [v / 50  for v in values] -  keine gute Idee, da die approximierte Sgn-Func  sehr kleine Werte nah 0 (0,001)  ungenau berechnet


def homomorphic_sgn(a_minus_b):
    # Coefficients for the degree 7 polynomial approximation
    # Die Nullen stehen für die fehlenden geradzahligen Potenzen
    # Nur die ungeradzahligen Potenzen (0, x^1, 0, x^3, 0, x^5, 0, x^7) sind mit echten Koeffizienten belegt.
    # !! warum? Die Signumfunktion sgn(x) ist ungerade: -> Polynome, die ungerade Funktionen approximieren, enthalten nur ungeradzahlige Potenzen
    
    '''
    # getestete Polynome
    '''
    # coeffs = monomial_coeffs # grad 9    (ist Min Wert: 0.33   --  soll Min Wert: 0.36) 
                               # grad 9    (ist Min Wert: -0.03  --  soll Min Wert: 0.0)  
    # coeffs = [0, 3.4375, 0, -4.6875, 0, 2.8125, 0, -0.5625] # grad 7    (ist Min Wert:  0.29  --  soll Min Wert: 0.36)
                                                              # grad 7    (ist Min Wert: -0.06  --  soll Min Wert: 0.0)
    coeffs = [0, 1.5708, 0, -0.6459, 0,  0.0796, 0, -0.0046] # grad 7   (ist Min Wert: 0.37 -- soll Min Wert: 0.36)  <-
                                                               # grad 7   (ist Min Wert: 0.01  -- soll Min Wert: 0.0)  <-
    # coeffs = [0, 1.5708,  0, -0.6704, 0, 0.1346, 0, -0.0125, 0, 0.0005] # grad 9   (ist Min Wert: 0.367  -- soll Min Wert: 0.36) <-
                                                                          # grad 9   (ist Min Wert: 0.007   -- soll Min Wert: 0.0) <-
    # coeffs = [0, 4.375, 0, -8.203125, 0, 8.4375, 0, -3.9375, 0, 0.6875] # grad 9   (ist Min Wert: 0.22  -- soll Min Wert: 0.36)
                                                                          # grad 9   (ist Min Wert: -0.13   -- soll Min Wert: 0.0)
    
    # Führt die Polynom-Auswertung homomorph aus (also auf verschlüsselten Zahlen), ohne zu entschlüsseln.
    # Die Eingabe muss auf [−1,1] normiert sein, damit die Approximation stimmt!
    return a_minus_b.polyval(coeffs)  

'''
# --- Step 3: Homomorphic Minimum Funktion ---
def homomorphic_min(enc_a, enc_b):
    """
    Minimum Funktion anwenden
    min(a, b) = 0.5 * [a + b - sgn(a - b) * (a - b)]
    Alle Berechnungen hier sind homomorph.
    """

    # Berechnet (a - b) auf den verschlüsselten Daten.
    a_minus_b = enc_a - enc_b #  a-b muss im Bereich [-1, 1] liegen, sonst ist das Ergebnis der sgn-Func ungenau!
    # Berechnet (a + b) auf den verschlüsselten Daten.
    a_plus_b = enc_a + enc_b
    
    # Ruft unsere homomorphe sgn-Funktion auf dem verschlüsselten Ergebnis von (a - b) auf.
    sgn_a_minus_b = homomorphic_sgn(a_minus_b)
    
    # Implementiert den Teil sgn(a - b) * (a - b) der Formel
    min_val = a_plus_b - (sgn_a_minus_b * a_minus_b)
    
    # Multipliziert das Endergebnis mit dem Klartext-Wert 0.5.
    return min_val * 0.5
'''
'''
# !!!!! Turniermethoden sind in CKKS sehr vorteilhaft, da sie die Multiplikationstiefe reduzieren
# normale for each schleife // ohne Turniermethode // Multiplikative Tiefe = 18 und Berechnungszeit = 23 sec
# mit Tourniermethode - Multiplikative Tiefe = 12 und Berechnungszeit = 6.61 sec
def find_min_tournament(encrypted_list):
    
    #Findet das Minimum in einer Liste von verschlüsselten Werten
    #unter Verwendung einer Turniermethode zur Reduzierung der multiplikativen Tiefe.
    
    if not encrypted_list:
        return None
    
    current_level = encrypted_list
    
    # Das "Turnier" läuft, solange es mehr als einen "Spieler" (Wert) gibt
    while len(current_level) > 1:
        next_level = []
        # Vergleiche die Werte paarweise
        for i in range(0, len(current_level), 2):
            # Wenn es noch einen "Gegner" gibt
            if i + 1 < len(current_level):
                p1 = current_level[i]
                p2 = current_level[i+1]
                winner = homomorphic_min(p1, p2)
                next_level.append(winner)
            # Wenn ein Wert ohne "Gegner" ist (ungerade Anzahl)
            else:
                next_level.append(current_level[i])
        
        # Die Gewinner ziehen in die nächste Runde ein
        current_level = next_level
        
    # Der letzte verbleibende Wert ist das globale Minimum
    return current_level[0]
'''
def homomorphic_min_with_label(enc_a, enc_label_a, enc_b, enc_label_b):
    a_minus_b = enc_a - enc_b
    a_plus_b = enc_a + enc_b
    label_diff = enc_label_a - enc_label_b
    label_sum = enc_label_a + enc_label_b
    sgn = homomorphic_sgn(a_minus_b)
    min_val = (a_plus_b - sgn * a_minus_b) * 0.5
    min_label = (label_sum - sgn * label_diff) * 0.5
    return min_val, min_label


# !!!!! Turniermethoden sind in CKKS sehr vorteilhaft, da sie die Multiplikationstiefe reduzieren
# normale for each schleife // ohne Turniermethode // Multiplikative Tiefe = 18 und Berechnungszeit = 23 sec
# mit Tourniermethode - Multiplikative Tiefe = 12 und Berechnungszeit = 6.61 sec
def find_min_tournament_with_labels(enc_label_list):
    if not enc_label_list:
        return None, None
    current_level = enc_label_list
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                val1, label1 = current_level[i]
                val2, label2 = current_level[i+1]
                winner_val, winner_label = homomorphic_min_with_label(val1, label1, val2, label2)
                next_level.append((winner_val, winner_label))
            else:
                next_level.append(current_level[i])
        current_level = next_level
    return current_level[0]



# Demo
plain_values = [6.1, 6.2, 6.3, 6.01]
labels_plain = ["krank", "gesund", "krank", "gesund"]
label_map = {"krank": 1.0, "gesund": 0.0}
print(f"Originale Liste: {plain_values}")
print(f"Zugehörige Labels: {labels_plain}")


norm_plain_values = min_max_scale(plain_values)
print(f"Normierte Liste: {norm_plain_values}")
print(f"Erwartetes Minimum: {min(norm_plain_values)}")

norm_labels = [label_map[lbl] for lbl in labels_plain]
print(f"Normierte Labels: {norm_labels}")
encrypted_values = [ts.ckks_vector(context, [v]) for v in norm_plain_values]
encrypted_labels = [ts.ckks_vector(context, [l]) for l in norm_labels]
enc_label_list = list(zip(encrypted_values, encrypted_labels))
print(f"enc_label_list: {enc_label_list}")

start_time = time.time()
result_encrypted, result_label = find_min_tournament_with_labels(enc_label_list)
result_decrypted = result_encrypted.decrypt()
label_decrypted = result_label.decrypt()
final_result = round(result_decrypted[0], 4)
final_label = round(label_decrypted[0], 4)
class_result = "krank" if final_label >= 0.5 else "gesund"

print(f"Homomorph berechnetes Min (entschlüsselt): {final_result}")
print(f"Zugehörige Klasse (entschlüsselt): {final_label} → {class_result}")
print(f"Berechnungszeit: {time.time() - start_time:.2f} Sekunden")




