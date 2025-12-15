import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

with open(os.path.join(os.path.dirname(__file__), "data.csv"), 'r') as file:
    data = file.readlines()
    data = [list(map(float, line.strip().split(','))) for line in data]
    x = np.array(data[0])
    y = np.array(data[1])

plt.scatter(x, y, color='blue', label='Données')

poly = PolynomialFeatures(degree=4, include_bias=False)
x_reshaped = x.reshape(-1, 1)
x_poly = poly.fit_transform(x_reshaped)

model = LinearRegression()
model.fit(x_poly, y)

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

plt.plot(x_range, y_pred, color='red', label='Ajustement polynomial degré 4')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("Coefficients du modèle polynomial :", model.coef_)
print("Intercept :", model.intercept_)
print("Erreur quadratique moyenne (MSE) :", mean_squared_error(y, model.predict(x_poly)))