# Test Rückgabewerte der approximierten Sign-Funktion durch ein zusammengesetztes Polynom
# Erwartete Ergebnisse:  nahe 1, nahe 0,  oder nahe -1
# Die finalen Ergebnisse werden als Kommentar im unteren Bereich nach dem Code dargestellt

import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt

# 1. Definition der Polynomkoeffizienten in der Potenzbasis (x^0, x^1, x^2, ...)
# Polynom: P_0(x) = 1.5*x - 0.5*x^3
# Diese Koeffizienten sind für die Anwendung in einer Kette vorgesehen,
# um die Approximation schrittweise zu verfeinern.
poly_coeffs_single_step = [
    0,     # Koeffizient für x^0
    1.5,   # Koeffizient für x^1
    0,     # Koeffizient für x^2
    -0.5   # Koeffizient für x^3
]

# Anzahl der Kompositionen/Iterationen
NUM_COMPOSITIONS = 7 # Anzahl der Male, die das Polynom angewendet wird

# 2. CKKS-Kontext initialisieren
def setup_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[50] + [40]*14 + [50]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

context = setup_ckks_context()
print("CKKS-Kontext initialisiert.")

# 3. homomorphe Signum-Approximation mit Polynomkomposition
def homomorphic_sgn_composed(a_minus_b, num_compositions=NUM_COMPOSITIONS):
    """
    Verwendet ein Polynom in einer Kette (Komposition) zur Approximierung der Sign Funktion.
    Args:
        a_minus_b (ts.CKKSVector): Ein verschlüsselter Vektor, der die Differenz zweier Werte enthält.
        num_compositions (int): Die Anzahl der Male, die das Polynom hintereinander angewendet wird.
    Returns:
        ts.CKKSVector: Das verschlüsselte Ergebnis der Polynomauswertung.
    """
    print(f"In homomorphic_sgn_composed - {num_compositions} Kompositionen.")
    
    current_approx = a_minus_b
    for i in range(num_compositions):
        # Bei jeder Iteration wird das Polynom auf das Ergebnis der vorherigen angewendet.
        current_approx = current_approx.polyval(poly_coeffs_single_step)
        # Nach jeder Multiplikation muss der Skalierungsfaktor angepasst werden
        print(f"  Komposition {i+1} abgeschlossen.")
    
    return current_approx


# 4. Testfunktion für die homomorphe Signum-Approximation mit Komposition
def homomorphic_sgn_test_composed(value, ckks_context):
    # Sicherstellen, dass der Wert im Approximationsbereich [-1, 1] liegt
    if not -1.0 <= value <= 1.0:
        print(f"WARNUNG: Eingabewert {value} liegt außerhalb des optimalen Bereichs [-1, 1] für die Approximation.")

    # Verschlüsseln des Wertes
    enc_value = ts.ckks_vector(ckks_context, [value])

    # Homomorphe Polynomauswertung (Signum-Approximation) mit Komposition
    enc_sgn_approx = homomorphic_sgn_composed(enc_value)

    # Entschlüsseln des Ergebnisses
    dec_sgn_approx = enc_sgn_approx.decrypt()[0]

    # Erwarteter Signum-Wert (echte Signum-Funktion)
    expected_sgn = np.sign(value)

    return value, expected_sgn, dec_sgn_approx

# 5. Testdaten und Durchführung der Tests
test_values = [0.8, -0.73, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.05]

print("\n--- Test der homomorphen Signum-Approximation mit Komposition ---")
results_composed = []
for val in test_values:
    original_val, expected, decrypted_approx = homomorphic_sgn_test_composed(val, context)
    results_composed.append({
        "Originalwert": original_val,
        "Erwartetes Signum": expected,
        "Entschlüsselte Approx.": decrypted_approx,
        "Differenz": abs(expected - decrypted_approx)
    })

# 6. Ergebnisse ausgeben
print("\nTestergebnisse (mit Polynom-Komposition):")
print(f"{'Originalwert':<15} {'Erwartetes Signum':<20} {'Entschlüsselte Approximation':<28} {'Differenz':<15}")
print("-" * 78)
for res in results_composed:
    print(f"{res['Originalwert']:<15.4f} {res['Erwartetes Signum']:<20.4f} {res['Entschlüsselte Approx.']:<28.4f} {res['Differenz']:<15.4e}")

# 7. Kurze Bewertung
max_diff_composed = max(res['Differenz'] for res in results_composed)
if max_diff_composed < 0.01:
    print("\nBewertung: Die Approximation mit Komposition funktioniert sehr gut, die Differenzen sind gering.")
elif max_diff_composed < 0.1:
    print("\nBewertung: Die Approximation mit Komposition funktioniert recht gut, mit moderaten Differenzen.")
else:
    print("\nBewertung: Die Approximation mit Komposition zeigt bei einigen Werten größere Differenzen.")

print(f"\nVerwendete Anzahl von Kompositionen: {NUM_COMPOSITIONS}")
print("Eine höhere Anzahl von Kompositionen kann die Genauigkeit weiter verbessern, erhöht aber auch die Multiplikationstiefe und das Rauschen.")

#Plot für die Approximation des Polynoms 
# zur visuellen Bestätigung
def plot_approximation_composed(poly_coeffs, num_compositions, context, test_values_for_plot):
    x_for_plot = np.linspace(-1, 1, 400)
    y_sign = np.sign(x_for_plot)

    # Berechne die komponierte Polynom-Approximation für den Plot
    def composed_poly_eval(x_val, coeffs, num_comps):
        current_y = x_val
        for _ in range(num_comps):
            current_y = np.polyval(coeffs[::-1], current_y) # np.polyval erwartet höchste Grad zuerst
        return current_y

    y_poly_approx_composed = composed_poly_eval(x_for_plot, poly_coeffs, num_compositions)

    plt.figure(figsize=(10, 6))
    plt.plot(x_for_plot, y_sign, label='Signum-Funktion (True)', color='blue', linestyle='--', linewidth=2)
    plt.plot(x_for_plot, y_poly_approx_composed, label=f'Komponierte Polynom-Approximation (x{num_compositions})', color='red', linestyle='-', alpha=0.7)

    # Die Entschlüsselten Werte auf dem Plot visualisieren
    decrypted_points = []
    for val in test_values_for_plot:
        _, _, dec_approx = homomorphic_sgn_test_composed(val, context)
        decrypted_points.append(dec_approx)
    
    plt.scatter(test_values_for_plot, decrypted_points, color='green', marker='o', s=50, label='Entschlüsselte CKKS-Punkte')

    plt.title(f'Polynom-Approximation der Signum-Funktion mit {num_compositions} Komposition(en)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.ylim(-1.5, 1.5)
    plt.show()

plot_approximation_composed(poly_coeffs_single_step, NUM_COMPOSITIONS, context, test_values) 

###################
# Ergebnisse
###################
#
#  Testergebnisse (mit Polynom-Komposition): (7 mal)
#  Originalwert    Erwartetes Signum    Entschlüsselte Approx.      Differenz
#  ------------------------------------------------------------------------------
#  0.8000          1.0000               1.0000                       0.000
# -0.7300         -1.0000              -1.0000                       0.000
#  0.6000          1.0000               1.0000                       0.000
# -0.5000         -1.0000              -1.0000                       0.000
#  0.4000          1.0000               1.0000                       0.000
# -0.3000         -1.0000              -1.0000                       0.000
#  0.3000          1.0000               1.0000                       0.000
# -0.1000         -1.0000              -0.9659                       0.035
#  0.0500          1.0000               0.7144                       0.286
