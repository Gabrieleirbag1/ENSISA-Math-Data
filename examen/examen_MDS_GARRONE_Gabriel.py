# -*- coding: utf-8 -*-
"""
@author: JDION
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

### EXERCICE 1

def heaviside(x):
    return tf.where(x >= 0, 1.0, 0.0)

# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(4, input_dim=2, activation = heaviside)) #à compléter
modele.add(Dense(1, activation = heaviside)) #à compléter

# Couche 1
poids = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]) #à compléter
biais = np.array([0.0, 1.0, 0.0, 1.0]) #à compléter
param = [poids,biais]
modele.layers[0].set_weights(param)
# j'ai du rajouter un .T sinon les dimensions n'allaient pas

# Couche 2
poids = np.array([[1.0], [1.0], [1.0], [1.0]]) #à compléter
biais = np.array([-4.0]) #à compléter
param = [poids,biais]
modele.layers[1].set_weights(param)

# Affichage
x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 2, 100)
x_grid, y_grid = np.meshgrid(x, y)
xy_grid = np.c_[x_grid.ravel(), y_grid.ravel()]

# Prédiction des valeurs sur la grille
z_grid = modele.predict(xy_grid).reshape(x_grid.shape)

# Visualisation des résultats en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Sortie du réseau sur [-1, 2]^2')
plt.show()



### EXERCICE 3 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"

cols = [
    "Frequency", 
    "Angle_Attack", 
    "Chord_Length",
    "Free_Stream_Velocity",
    "Suction_Side_Displacement_Thickness",
    "Scaled_Sound_Pressure",
]

df = pd.read_csv(url, sep="\t", header=None, names=cols)
X = df.drop("Scaled_Sound_Pressure", axis=1)
y = df["Scaled_Sound_Pressure"]

#question 1 - Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## valeur de la fréquende ce vibration du profil d'aile num 1 

#qesution 2 
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y)

print("Coefficients de la régression linéaire :", lin_reg.coef_)

#question 3
profile_1 = np.array([[500, 0, 0.305, 71.3, 0.00219]])
profile_1_scaled = scaler.transform(profile_1)
predicted_value = lin_reg.predict(profile_1_scaled)
print(f"\nQ3)\nValeur prédite de Scaled_Sound_Pressure pour le profil d'aile numéro 1 : {predicted_value[0]:.2f}")
print("On constate que cette valeur est très proche de la valeur réelle, indiquant que le modèle de régression linéaire sans régularisation fonctionne bien pour cette entrée spécifique.")

#question 4 - ajout de la régularisation Lasso
#a)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_scaled, y)
print("\nQ4 a)\nCoefficients de la régression Lasso :", lasso_reg.coef_)

#b) 
alphas = np.logspace(-4, 1, 100)
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=cols[i])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Chemin de régularisation Lasso')
plt.legend()
plt.grid(True)
plt.show()

print("\nQ4 b)\nUn ordre de grandeur pertinent pour alpha semble être entre 10^-2 et 10^-1 car c'est dans cette plage que les coefficients des variables inutiles s'annulent tandis que celui de la variable principale reste significatif.")

#c)
idx_max = np.argmax(np.abs(lasso_reg.coef_))
print(f"\nQ4 c)\nLa variable explicative ayant le plus d'effet est {cols[idx_max]} avec un coefficient de {lasso_reg.coef_[idx_max]:.2f}")