

#   .\venv311\Scripts\activate
import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

################ Start Polynom ####################

# 1. Tschebyscheff-Approximation der Signumfunktion
CHEBYSHEV_DEGREE = 9 # Grad des Tschebyscheff-Polynoms. Ein höherer Grad -> genauere Approximation, erhöht aber auch Rauschen in CKKS
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
    poly_modulus_degree=32768, # 16384, 8192 -> error
    coeff_mod_bit_sizes=[60] + [40]*18 + [60]  # Multiplikative Tiefe < 15  -> error   # 15 ok für Polynomgrad 7
)
context.generate_galois_keys()
context.global_scale = 2**40
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
    # coeffs = [0, 1.5708, 0, -0.6459, 0,  0.0796, 0, -0.0046] # grad 7   (ist Min Wert: 0.37 -- soll Min Wert: 0.36)  <-
                                                               # grad 7   (ist Min Wert: 0.01  -- soll Min Wert: 0.0)  <-
    coeffs = [0, 1.5708,  0, -0.6704, 0, 0.1346, 0, -0.0125, 0, 0.0005] # grad 9   (ist Min Wert: 0.367  -- soll Min Wert: 0.36) <-
                                                                          # grad 9   (ist Min Wert: 0.007   -- soll Min Wert: 0.0) <-
    # coeffs = [0, 4.375, 0, -8.203125, 0, 8.4375, 0, -3.9375, 0, 0.6875] # grad 9   (ist Min Wert: 0.22  -- soll Min Wert: 0.36)
                                                                          # grad 9   (ist Min Wert: -0.13   -- soll Min Wert: 0.0)
    
    # Führt die Polynom-Auswertung homomorph aus (also auf verschlüsselten Zahlen), ohne zu entschlüsseln.
    # Die Eingabe muss auf [−1,1] normiert sein, damit die Approximation stimmt!
    return a_minus_b.polyval(coeffs)  

# --- Step 3: Homomorphic Minimum Funktion ---
def homomorphic_min(enc_a, enc_b):
    """
    Minimum Funktion anwenden
    min(a, b) = 0.5 * [a + b - sgn(a - b) * (a - b)]
    """


    a_minus_b = enc_a - enc_b #  a-b muss im Bereich [-1, 1] liegen, sonst ist das Ergebnis der sgn-Func ungenau!
    a_plus_b = enc_a + enc_b
    
    
    sgn_a_minus_b = homomorphic_sgn(a_minus_b)
    
    # Hauptformel
    min_val = a_plus_b - (sgn_a_minus_b * a_minus_b)
    
    return min_val * 0.5

def find_min_tournament(encrypted_list):
    """
    Findet das Minimum in einer Liste von verschlüsselten Werten
    unter Verwendung einer Turniermethode zur Reduzierung der multiplikativen Tiefe.
    """
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


plain_values = [6.1, 6.2, 6.3, 6.01]   # Differenz jeder 2 Zahlen aus der Liste muss im Bereich [-1, 1] liegen -> normieren
norm_plain_values = min_max_scale(plain_values)
print(f"Original list: {norm_plain_values}")
print(f"Expected minimum: {min(norm_plain_values)}")

# Die ganze Liste mit Vektoren veschlüsseln
encrypted_values = [ts.ckks_vector(context, [v]) for v in norm_plain_values]

start_time = time.time()

#result_encrypted = encrypted_values[0]
#for i in range(1, len(encrypted_values)):
#    result_encrypted = homomorphic_min(result_encrypted, encrypted_values[i])
    

result_encrypted = find_min_tournament(encrypted_values)
result_decrypted = result_encrypted.decrypt()

final_result = round(result_decrypted[0], 4)

print(f"Homomorph berechnetes Min (entschlüsselt): {final_result}")
print(f"Berechnungszeit: {time.time() - start_time:.2f} Sekunden")


