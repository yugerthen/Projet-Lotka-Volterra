import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Charger les données
data = pd.read_csv('populations_lapins_renards.csv')
rabbits = data['lapin'].astype(float).values
foxes = data['renard'].astype(float).values

# Modèle de Lotka-Volterra
def lotka_volterra(y, t, alpha, beta, gamma, delta):
    prey, predator = y
    return [
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator
    ]

# Simulation avec la méthode d'Euler
def simulate(y0, t, params):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + np.array(lotka_volterra(y[i-1], t[i-1], *params)) * dt
        y[i] = np.maximum(y[i], 0)  # Pas de populations négatives
    return y

# Fonction pour calculer l'erreur quadratique moyenne (MSE)
def mse(params):
    y0 = [rabbits[0], foxes[0]]
    t = np.linspace(0, len(rabbits) - 1, len(rabbits))
    y_sim = simulate(y0, t, params)
    return np.mean((rabbits - y_sim[:, 0])**2 + (foxes - y_sim[:, 1])**2)

# Optimisation des paramètres
initial_guess = [0.1, 0.01, 0.1, 0.01]  # alpha, beta, gamma, delta
bounds = [(0.05, 0.5), (0.005, 0.1), (0.1, 0.8), (0.005, 0.05)]
result = minimize(mse, initial_guess, bounds=bounds)
alpha, beta, gamma, delta = result.x

# Simulation avec les paramètres optimaux
t = np.linspace(0, len(rabbits) - 1, len(rabbits))
y0 = [rabbits[0], foxes[0]]
y_sim = simulate(y0, t, [alpha, beta, gamma, delta])

# Tracé des résultats
plt.figure(figsize=(10, 5))
plt.plot(rabbits, 'b-', label="Lapins (réels)")
plt.plot(foxes, 'r-', label="Renards (réels)")
plt.plot(y_sim[:, 0], 'b--', label="Lapins (simulés)")
plt.plot(y_sim[:, 1], 'r--', label="Renards (simulés)")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.title("Modèle Lotka-Volterra : Ajustement des paramètres")
plt.legend()
plt.grid()
plt.show()

# Affichage des résultats optimaux
print(f"Paramètres optimaux : alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}, delta={delta:.4f}")
