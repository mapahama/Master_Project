import tenseal as ts

# Schritt 1: Erstellen den CKKS-Kontext
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 20, 40]  # drei Stufen - nach zwei Multiplikationen muss reskaliert werden
)
context.global_scale = 2**20
context.generate_galois_keys()

# Schritt 2: Verschlüsseln zwei Vektoren
v1 = [2.5]
v2 = [4.0]
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

# Schritt 3: Multiplikation – erzeugt automatisch großes Delta
product = enc_v1 * enc_v2  # erwartet: [10.0]
print("Nach Multiplikation (verschlüsselt):", product)

# Schritt 4: Weiterrechnen (nochmal multiplizieren)
enc_const = ts.ckks_vector(context, [0.5])
product_scaled = product * enc_const  # sollte automatisch wieder skalieren

# Schritt 5: Entschlüsselung und Testausgabe
result = product_scaled.decrypt()
print("Nach zweiter Multiplikation & Rescaling (entschlüsselt):", result)