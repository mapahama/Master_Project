# ============================================
# Tschebyscheff-Approximation der Signumfunktion
# ============================================

import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Tschebyscheff-Approximation der Signumfunktion
CHEBYSHEV_DEGREE = 15
x_vals = np.linspace(-1, 1, 2000)
sign_vals = np.sign(x_vals)

# Polynomfit im Tschebyscheff-System
cheb_poly = np.polynomial.chebyshev.Chebyshev.fit(x_vals, sign_vals, CHEBYSHEV_DEGREE)
monomial_coeffs = cheb_poly.convert(kind=np.polynomial.Polynomial).coef

# 2. Visualisierung der Approximation
plt.figure(figsize=(10, 6))
plt.plot(x_vals, sign_vals, "--", label="sign(x)")
plt.plot(x_vals, cheb_poly(x_vals), label=f"Chebyshev Approx (deg={CHEBYSHEV_DEGREE})")
plt.legend(); plt.title("Tschebyscheff-Approximation der Signumfunktion"); plt.grid(True)
plt.ylim(-1.5, 1.5); plt.show()

# 3. CKKS-Kontext initialisieren
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

# 4. Vergleich: festgelegter Wert vs. deterministische Testwerte
reference = 0.5
test_values = np.linspace(-1, 1, 100)
enc_reference = ts.ckks_vector(context, [reference])

results = []
for val in test_values:
    # Schritt 1: Verschlüsselung des Testwerts
    enc_val = ts.ckks_vector(context, [val])

    # Schritt 2: Subtraktion im Ciphertext = Differenz val - reference
    enc_diff = enc_val - enc_reference

    # Schritt 3: Anwendung des Signum-Approximationspolynoms auf die Differenz
    enc_sign_approx = enc_diff.polyval(monomial_coeffs)

    # Schritt 4: Entschlüsselung (zur Evaluation)
    decrypted_result = enc_sign_approx.decrypt()[0]

    # Schritt 5: Auswertung – war das Vorzeichen korrekt?
    predicted = decrypted_result > 0      # erfolgt hier nur zur Evaluierung!
    expected = val > reference
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
