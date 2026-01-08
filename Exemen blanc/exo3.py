from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
import matplotlib.pyplot as plt

def read_data():
    housing_data = np.genfromtxt('housing.csv', delimiter=',')
    return housing_data

housing_data = read_data()

X, y = housing_data[:, :-1], housing_data[:, -1]

# Q1
print("\n QUESTION 1 \n")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
valeur_q1 = X_scaled[0, 0]
print("Revenu médian :", valeur_q1)

# Q2
print("\n QUESTION 2 \n")
reg = LinearRegression()
reg.fit(X_scaled, y)

coefs_str = ', '.join([f'{coef:.4f}' for coef in reg.coef_])
print("Intercept :", reg.intercept_)
print("Coefficients de régression linéaire :", coefs_str)

# Q3
print("\n QUESTION 3 \n")
x_g1 = X_scaled[0, :].reshape(1, -1)
prediction = reg.predict(x_g1)
vrai_prix = y[0]
print("Prédiction pour le premier échantillon :", prediction[0])
print("Valeur réelle pour le premier échantillon :", vrai_prix)

# Q4 a
print("\n QUESTION 4 \n")
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_scaled, y)


# Q4 b
n_lambda = 100
lambdas = np.logspace(-4, 2, n_lambda)
coefs = []
for lmbda in lambdas:
    lasso = Lasso(alpha=lmbda)
    lasso.fit(X_scaled, y)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(lambdas, coefs[:, i], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Lasso Paths')
plt.legend()
plt.show()

# Q4 c
importance = np.abs(lasso_reg.coef_)
most_important_feature = np.argmax(importance) + 1  # +1 pour l'indexation à partir de 1
print(f"La variable explicative ayant le plus d'effet sur la variable expliquée est la feature {most_important_feature} avec un coefficient de {lasso_reg.coef_[most_important_feature - 1]:.4f}.")