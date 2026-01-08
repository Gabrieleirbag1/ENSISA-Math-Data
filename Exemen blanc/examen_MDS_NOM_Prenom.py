# -*- coding: utf-8 -*-
"""
@author: JDION

Requirements:
  - Python 3.8+
  - tensorflow
  - numpy
  - matplotlib

Run:
  python examen_MDS_NOM_Prenom.py

This script builds a tiny network (Heaviside activations) and visualizes its output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Use tensorflow.keras to avoid conflicts with a separate keras package
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

### EXERCICE 1

def heaviside(x):
    # Retourne 1.0 si x>=0 sinon 0.0
    return tf.where(x >= 0, 1.0, 0.0)

# Enregistrer la fonction d'activation personnalisée correctement
get_custom_objects().update({'heaviside': heaviside})

def build_model():
    """Construit le réseau avec poids fixés et renvoie l'objet modèle."""
    modele = Sequential([
        tf.keras.Input(shape=(2,)),
        Dense(2, activation=heaviside),  # (2) fonctions H, et (2) variables
        Dense(1, activation=heaviside)   # (1) seul point rouge
    ])

    # Couche 1 : kernel shape = (input_dim, units) = (2, 2)
    coeff = np.array([[-1.0, 3.0],
                      [ 2.0, 1.0]])  # valeurs pour x et y
    biais = np.array([0.0, 0.0])      # pas de biais pour H1 et H2
    poids = [coeff, biais]
    modele.layers[0].set_weights(poids)

    # Couche 2 : kernel shape = (units_prev, units) = (2,1)
    coeff = np.array([[1.0],
                      [1.0]])  # valeurs en sortie de H1 et H2
    biais = np.array([-2.0])
    poids = [coeff, biais]
    modele.layers[1].set_weights(poids)

    return modele


### EXERCICE 3 

def read_data():
    # Use path relative to this file so the script works when run from any CWD
    housing_path = os.path.join(os.path.dirname(__file__), 'housing.csv')
    try:
        housing_data = np.genfromtxt(housing_path, delimiter=',', skip_header=1)
        return housing_data
    except Exception as e:
        print(f"Could not read housing data at {housing_path}: {e}")
        return None


def main():
    modele = build_model()

    # Affichage
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
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
    ax.set_title('Sortie du réseau sur [-5, 5]^2')
    plt.show()

    # Chargement des données (Exercice 3)
    housing_data = read_data()
    if housing_data is None:
        print("housing_data not loaded — ensure 'housing.csv' is located next to this script.")
    else:
        print(f"housing_data loaded, shape = {housing_data.shape}")


if __name__ == '__main__':
    main()

