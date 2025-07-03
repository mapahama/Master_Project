# .\venv311\Scripts\activate

# ============================================================================================
# Homomorpher Vergleich von Zahlen mittels Polynomapproximation der Signumfunktion
# ============================================================================================
#
# --- ZWECK DES CODES ---
#
# Dieses Skript demonstriert, wie ein Vergleich von zwei Zahlen (z. B. ist a > b?) auf
# homomorph verschlüsselten Daten mit dem CKKS-Schema durchgeführt werden kann.
# Da homomorphe Verschlüsselung keine direkten Vergleichs-Operationen 
# erlaubt, wird ein mathematischer Ansatz angewendet:
#
# Der Vergleich "a > b" wird in die äquivalente Form "sign(a - b) == 1" umgewandelt.
# Die unstetige Signumfunktion (Vorzeichenfunktion) wird wiederum durch ein
# stetiges Polynom angenähert, das homomorph ausgewertet werden kann.
#
# --- ABLAUF ---
#
# 1.  **Approximation (Mathematik):**
#     Ein Tschebyscheff-Polynom wird erzeugt, das die Signumfunktion im
#     Intervall [-1, 1] bestmöglich nachahmt. Der Grad des Polynoms ist ein
#     entscheidender Parameter für die Genauigkeit. 
#
# 2.  **Homomorphe Konfiguration (Kryptografie):**
#     Ein -->CKKS-Kontext<-- wird mit der `tenseal`-Bibliothek initialisiert. Die Parameter
#     (insb. `poly_modulus_degree` und `coeff_mod_bit_sizes`) definieren die Sicherheit
#     und das "Rauschbudget", das für die Komplexität des Polynoms ausreichen muss.
#
# 3.  **Evaluierung (Anwendung):**
#     - Ein Referenzwert (`reference`) und mehrere Testwerte (`test_values`) werden definiert.
#     - Die Differenz der verschlüsselten Werte wird berechnet.
#     - Das angenäherte Signum-Polynom wird auf dieser verschlüsselten Differenz
#       ausgewertet (`.polyval`).
#     - Das Ergebnis wird entschlüsselt und die Genauigkeit der Vorhersage
#       (war das Vorzeichen korrekt?) wird überprüft.
#
# --- KERNKONFLIKT ---
#
# Der Code illustriert den zentralen Kompromiss der homomorphen Berechnung:
# - **Hoher Polynomgrad:** Führt zu einer besseren mathematischen Annäherung.
# - **Hoher Polynomgrad:** Erfordert mehr homomorphe Multiplikationen, was zu mehr Rauschen
#   führt und die kryptografische Präzision senkt oder ein größeres Rauschbudget (tiefere `coeff_mod_bit_sizes`) erfordert.
#
# ============================================================================================



import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Tschebyscheff-Approximation der Signumfunktion
CHEBYSHEV_DEGREE = 10 # Grad des Tschebyscheff-Polynoms. Ein höherer Grad -> genauere Approximation, erhöht aber auch Rauschen in CKKS
x_vals = np.linspace(-1, 1, 2000) # Erstellt ein Array von 2000 gleichmäßig verteilten Zahlen im Intervall von -1 bis +1

sign_vals = np.sign(x_vals) #  Berechnet Signumfunktion (Vorzeichenfunktion) für jeden Wert in x_vals
                            #. Es wird ausgegeben: für negative x-Werte -1, für positive x-Werte +1 und für x=0 den Wert 0 
                            #  Diese Werte sind das, was unser Polynom lernen soll zu imitieren.

# !!! Polynomfit im Tschebyscheff-System !!!
# Die Methode (Chebyshev.fit) findet das bestmögliche Polynom vom Grad CHEBYSHEV_DEGREE,
#  das die durch x_vals und sign_vals gegebenen Punkte beschreibt.
cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_vals, sign_vals, CHEBYSHEV_DEGREE) # das gefundene Tschebyscheff Polynom
monomial_coeffs = cheb_poly.convert(kind=np.polynomial.Polynomial).coef #konvertiert das  gefundene Tschebyscheff-Polynom (cheb_poly) in  Standardform und extrahiert  Koeffizienten (.coef).


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


# 3. CKKS-Kontext initialisieren
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    # "poly_modulus_degre" definiert die Größe der Polynomringe, in denen gerechnet wird. 
    #    Ein größerer Wert bedeutet mehr Sicherheit und mehr Kapazität für Rauschen, aber auch langsamere Berechnungen.
    #    Ein niedrigerer Wert bedeutet schnellere aber ungenauere Berechnungen
    poly_modulus_degree= 16384, # 32768   # für Polynomgrad 10
    # "coeff_mod_bit_sizes"  ist Parameter für das "Rauschbudget" - Multiplikationstiefe
    #    Jede Multiplikation auf verschlüsselten Daten "verbraucht" einen der mittleren Moduln (hier die 40er).
    #    Die Anzahl dieser mittleren Moduln  bestimmt die multiplikative Tiefe – 
    #    also wie viele Multiplikationen hintereinander ausgeführt werden können
    coeff_mod_bit_sizes= [60, 40, 40, 40, 40, 60]   # [60, 40, 40, 40, 40, 40, 40, 60]  - multiplikationstiefe hängt von dem Polynomgrad ab
)
context.generate_galois_keys() #  ermöglichen "Rotationen" (Drehungen) von verschlüsselten Vektoren. (z.B. für Multiplikation, Summieren)
# "context.global_scale" - CKKS kann nicht direkt mit Dezimalzahlen (wie 5.1) rechnen, sondern nur mit ganzen Zahlen.
#    Um trotzdem Gleitkommazahlen zu unterstützen, skaliert CKKS sie zu großen ganzen Zahlen (z. B. 5.1 → 51).
#    global_scale bestimmt, wie stark skaliert wird.
context.global_scale = 2 ** 40 

##################################################################
#  Test durchführen
#
#  Wir verschlüsseln Werte, führen die Berechnung (sign(val - ref)) homomorph durch und 
#  entschlüsseln das Ergebnis, um die Genauigkeit zu überprüfen.
#
###################################################################

# 4. Vergleich: festgelegter Wert vs. deterministische Testwerte
reference = 0.5 # Der Referenzwert, mit dem wir vergleichen wollen.    z.B.   ob   0.5 > 1
test_values = np.linspace(-1, 1, 100) # Eine neue Reihe von 100 Testwerten im Intervall [-1;1], um unsere Methode zu evaluieren.

results = []
for val in test_values:
    # Schritt 1: Verschlüsselung des Testwerts
    enc_val = ts.ckks_vector(context, [val])

    # Schritt 2: Subtraktion im Ciphertext = Differenz val - reference
    enc_diff = enc_val - reference # encrypted - unencrypted  in diesem Fall möglich

    # Schritt 3: Anwendung des Signum-Approximationspolynoms auf die Differenz
    # Die .polyval()-Methode wertet unser zuvor berechnetes Polynom (gegeben durch monomial_coeffs) 
    # auf dem verschlüsselten Wert enc_diff aus. Hier kann das Rauschen waschen.
    enc_sign_approx = enc_diff.polyval(monomial_coeffs)

    # Schritt 4: Entschlüsselung (zur Evaluation)
    decrypted_result = enc_sign_approx.decrypt()[0]

    # Schritt 5: Auswertung – war das Vorzeichen korrekt?
    predicted = decrypted_result > 0  # Wenn der entschlüsselte (rauschende) Wert größer als 0 ist,  ist die Vorhersage TRUE
    expected = val > reference # Wir berechnen das korrekte, erwartete Ergebnis in dem unverschlüsselten Szenario.
    results.append({
        "Testwert": round(val, 4),
        "Erwartet": expected,
        "Sign_approx": round(decrypted_result, 4),
        "Vorhersage": predicted,
        "Korrekt?": predicted == expected
    })

# 5. Analyse der Ergebnisse
df = pd.DataFrame(results)
accuracy = df["Korrekt?"].mean() * 100

print("\n--- Auswertung der homomorphen Vergleichsapproximation ---")
print(df.to_string(index=False))
print(f"\nTrefferquote: {accuracy:.2f}% bei Tschebyscheff-Grad {CHEBYSHEV_DEGREE}")
# Solange das Vorzeichen nach der sign-function  korrekt ist (negativ für val < 0.5 und positiv für val > 0.5), ist die Vorhersage richtig. 


############################################
#   Ergebnisse
############################################

#   Polynomgrad 7                                   <-
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=32768                          
#
# Ergebnis: Genauigkeit 77%      Geschwindigkeit: 19 sec  (Vergleich von 100 Werten)

#   Polynomgrad 7                                   
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60] 
#   poly_modulus_degree=16384                       <-
#
# Ergebnis: Genauigkeit 77%      Geschwindigkeit: 17 sec  (Vergleich von 100 Werten)

#   Polynomgrad 10                                   <-
#   context.global_scale = 2 ** 20                   <-
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=32768  
#
# Ergebnis: Genauigkeit 45%     Geschwindigkeit: 16 sec  (Vergleich von 100 Werten)

#   Polynomgrad 10 
#   context.global_scale = 2 ** 40                   <-
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=32768
#
# Ergebnis: Genauigkeit 100%   Geschwindigkeit: 24 sec  (Vergleich von 100 Werten)

#   Polynomgrad 10 
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=16384                         <-
#
# Ergebnis: Genauigkeit 100%   Geschindigkeit: 25 sec  (Vergleich von 100 Werten)

#   Polynomgrad 10 
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 60]          <-
#   poly_modulus_degree=16384                         
#
# Ergebnis: Error: Scale out of bounds (wegen nicht ausreichende Multiplikationstiefe)

#   Polynomgrad 15                                    <-
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=16384   
#
# Ergebnis: Genauigkeit 76%    Geschwindigkeit: 38 sec  (Vergleich von 100 Werten)

#   Polynomgrad 15                                    
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
#   poly_modulus_degree=32768                          <-
#
# Ergebnis: Genauigkeit 76%      Geschwindigkeit: 43 sec  (Vergleich von 100 Werten)

#   Polynomgrad 15                                    
#   context.global_scale = 2 ** 40
#   coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]  <-
#   poly_modulus_degree=32768                          
#
# Ergebnis: Genauigkeit 76%      Geschwindigkeit: 74 sec  (Vergleich von 100 Werten)