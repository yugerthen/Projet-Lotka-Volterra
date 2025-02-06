import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Charger les données
data = pd.read_csv('populations_lapins_renards.csv')
rabbits = data['lapin'].astype(float).values
foxes = data['renard'].astype(float).values

# Lotka-Volterra
def lotka_volterra(y, t, alpha, beta, gamma, delta):
    proi, preda = y
    return [
        alpha * proi - beta * proi * preda,
        delta * proi * preda - gamma * preda
    ]

# Simulation : Euler
def simulate(y0, t, para):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + np.array(lotka_volterra(y[i-1], t[i-1], *para)) * dt
        y[i] = np.maximum(y[i], 0)  # Pas de valeurs négatives
    return y

# Fonction pour l'optimisation
def mse(para):
    y0 = [rabbits[0], foxes[0]]
    t = np.linspace(0, len(rabbits) - 1, len(rabbits))
    y_sim = simulate(y0, t, para)
    return np.mean((rabbits - y_sim[:, 0])**2 + (foxes - y_sim[:, 1])**2)

# Optimisation
initial_guess = [0.1, 0.01, 0.1, 0.01]
bounds = [(0.05, 0.5), (0.005, 0.1), (0.1, 0.8), (0.005, 0.05)]
result = minimize(mse, initial_guess, bounds=bounds)
alpha, beta, gamma, delta = result.x

# Simulation avec les paramètres trouvés
t = np.linspace(0, len(rabbits) - 1, len(rabbits))
y0 = [rabbits[0], foxes[0]]
y_sim = simulate(y0, t, [alpha, beta, gamma, delta])

# Tracé des résultats
plt.plot(rabbits, 'b-', label="Lapins (réels)")
plt.plot(foxes, 'r-', label="Renards (réels)")
plt.plot(y_sim[:, 0], 'b--', label="Lapins (après simulation)")
plt.plot(y_sim[:, 1], 'r--', label="Renards (après simulation)")
plt.legend()
plt.xlabel("Temps")
plt.ylabel("Population")
plt.grid()
plt.show()

print(f"Résultats optimaux : alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}, delta={delta:.4f}")
