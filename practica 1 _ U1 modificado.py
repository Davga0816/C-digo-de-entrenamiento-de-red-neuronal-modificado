import numpy as np
import pandas as pd

class PerceptronLoanApproval(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Lectura del conjunto de datos con encabezados
df = pd.read_csv('C:/Users/isaac mendoza/Desktop/datosss.csv')  # Ruta a tu archivo CSV

# Preprocesamiento de datos
# Codificar el historial crediticio como valores numéricos
df['HistorialCrediticio'] = df['HistorialCrediticio'].map({'Malo': 0, 'Regular': 1, 'Bueno': 2})

# Extracción de características y etiquetas
X = df[['Edad', 'IngresosAnuales', 'CantidadDeDeudas', 'HistorialCrediticio']].values
y = df['Etiqueta'].values

# Creación y entrenamiento del perceptrón
perceptron = PerceptronLoanApproval(eta=0.1, n_iter=100)
perceptron.fit(X, y)

# Hacer una predicción para una nueva solicitud (ejemplo)
nueva_solicitud = np.array([55, 120, 0, 2])  # Nueva solicitud con cuatro características, historial crediticio "Bueno" codificado como 2
prediccion = perceptron.predict(nueva_solicitud)

# Imprimir el resultado de la predicción
if prediccion == 1:
    print("Se aprueba la solicitud de préstamo.")
else:
    print("Se rechaza la solicitud de préstamo.")
