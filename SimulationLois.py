import numpy as np

# Fonction pour générer une loi uniforme [a, b]
def loi_uniforme(a, b, n):
    U = np.random.uniform(0, 1, n)
    return a + (b - a) * U

# Fonction pour générer une loi exponentielle de paramètre λ
def loi_exponentielle(lambda_exp, n):
    U = np.random.uniform(0, 1, n)
    return -1 / lambda_exp * np.log(1 - U)

# Fonction pour générer une loi de Cauchy de paramètre c
def loi_cauchy(c, n):
    U = np.random.uniform(0, 1, n)
    return c * np.tan(np.pi * (U - 0.5))

# Fonction pour générer une loi de Bernoulli de paramètre p
def loi_bernoulli(p, n):
    U = np.random.uniform(0, 1, n)
    return (U < p).astype(int)

# Fonction pour générer une loi normale N(0, 1) avec méthode de Box-Muller
def loi_normale(n):
    U1 = np.random.uniform(0, 1, n // 2)
    U2 = np.random.uniform(0, 1, n // 2)
    X1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return np.concatenate((X1, X2))



