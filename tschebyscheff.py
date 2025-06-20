# .\venv311\Scripts\activate

## Tschebyscheff  Genauigkeit 80% - Vergleich von 20 werten
import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. Tschebyscheff-Approximation vorbereiten
# =============================================================================
CHEBYSHEV_DEGREE = 15
x_fit = np.linspace(-1, 1, 2000)
y_fit = np.sign(x_fit)

# Tschebyscheff-Polynom fitten und in Standardform konvertieren
chebyshev_poly = np.polynomial.chebyshev.Chebyshev.fit(x_fit, y_fit, deg=CHEBYSHEV_DEGREE)
power_coeffs = chebyshev_poly.convert(kind=np.polynomial.Polynomial).coef

# Genauigkeit berechnen
approximated_values = chebyshev_poly(x_fit)
errors = np.abs(y_fit - approximated_values)
max_error = np.max(errors)
avg_error = np.mean(errors)

print("\n--- Genauigkeit der Tschebyscheff-Approximation ---")
#print(f"Maximaler absoluter Fehler: {max_error:.4f}")
print(f"Durchschnittlicher Fehler: {avg_error:.4f}")

# =============================================================================
# 2. Plot: Tschebyscheff-Polynom vs. Signum-Funktion
# =============================================================================
plt.figure(figsize=(10, 6))
plt.plot(x_fit, y_fit, label="sign(x)", color="black", linestyle="--")
plt.plot(x_fit, approximated_values, label=f"Tschebyscheff (Grad {CHEBYSHEV_DEGREE})", color="green")
plt.title("Approximation der Signumfunktion mit Tschebyscheff-Polynom")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.ylim(-1.5, 1.5)
plt.show()

# =============================================================================
# 3. CKKS-Kontext vorbereiten
# =============================================================================
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# =============================================================================
# 4. Homomorpher Vergleich: 1 Wert gegen 20 andere
# =============================================================================
val_fixed = 0.5
comparison_values = np.random.uniform(-1, 1, 20)
enc_fixed = ts.ckks_vector(context, [val_fixed])

results = []
for val in comparison_values:
    enc_other = ts.ckks_vector(context, [val])
    enc_diff = enc_other - enc_fixed
    enc_sign = enc_diff.polyval(power_coeffs)
    dec_sign = enc_sign.decrypt()[0]
    
    predicted = dec_sign > 0
    expected = val > val_fixed
    correct = predicted == expected
    
    results.append({
        "Vergleichswert": round(val, 4),
        "sign(x)": round(dec_sign, 4),
        "Erwartet (val > 0.5)": expected,
        "Vorhersage": predicted,
        "Korrekt?": correct
    })

# =============================================================================
# 5. Ergebnis-Auswertung
# =============================================================================
correct_count = sum(r["Korrekt?"] for r in results)
total = len(results)
accuracy = (correct_count / total) * 100

df = pd.DataFrame(results)
print("\n--- Vergleichsergebnisse ---")
print(df.to_string(index=False))
print(f"\nAnzahl korrekter Vergleiche: {correct_count} von {total}")
print(f"--> Genauigkeit: {accuracy:.2f}%")