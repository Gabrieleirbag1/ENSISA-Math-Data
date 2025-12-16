import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(file_path = os.path.join(os.path.dirname(__file__), "titanic2.csv")):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if reader.line_num == 1:
                continue
            data.append([int(value) for value in row])
    return data

data = load_data()
data = np.array(data)
x = data[:, 0]
y = data[:, 1]

scaler = StandardScaler()
x_reshaped = x.reshape(-1, 1)
x_normalized = scaler.fit_transform(x_reshaped)

model = LogisticRegression(max_iter=1000)
model.fit(x_normalized, y)
plt.scatter(x, y, color='blue', label='Données', alpha=0.5)

x_range = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
x_range_normalized = scaler.transform(x_range)
y_prob = model.predict_proba(x_range_normalized)[:, 1]

plt.plot(x_range, y_prob, color='red', label='Probabilité de survie (degré 4)', linewidth=2)
plt.axhline(y=0.5, color='green', linestyle='--', label='Frontière de décision (p=0.5)')

plt.xlabel('Âge')
plt.ylabel('Survie / Probabilité')
plt.title('Régression logistique - Survie sur le Titanic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Affichage du modèle et de la fonction d'erreur
print("\nModèle de régression logistique:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

