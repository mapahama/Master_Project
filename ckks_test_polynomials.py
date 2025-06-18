############
# Test – Comparing values using Approximation of the sign function (via Polynomials, without comparison operators)
##############
import numpy as np

# Vergleichswerte
a_values = [5.2, 2.0, 1.5, 1.0, -3.0]
b = 1.5  # Referenzwert

# Signum-Approximation: |x| ≈ x * sign(x)
# Polynome //aus den Papers
def g(x):
    return -1.3266 * x**3 + 2.0766 * x

def f(x):
    return -0.5 * x**3 + 1.5 * x

# Verschachtelte Approximation der Signum-Funktion
def approx_sign(x):
    return f(f(g(g(x))))

# Vergleich durchführen
for a in a_values:
    diff = a - b
    sign_approx = approx_sign(diff)

    # Schwellenwerte für Interpretation
    if sign_approx > 0.5:
        relation = "a > b"
    elif sign_approx < -0.5:
        relation = "a < b"
    else:
        relation = "a ≈ b"

    print(f"a = {a:4.2f}, b = {b:4.2f} → a - b = {diff:5.2f}, sign ≈ {sign_approx:6.3f} → {relation}")
