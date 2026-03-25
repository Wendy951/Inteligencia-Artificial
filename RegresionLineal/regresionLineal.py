import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# PASO 1: CREAR DATOS (Simulación)
# X: Días de lluvia por mes (entre 0 y 20 días)
# y: Ventas del mes (relación: Ventas = -2 * lluvia + 26 + ruido)

np.random.seed(42)

X = np.random.randint(0, 20, size=(50, 1))  # Días de lluvia
y = (-2 * X).squeeze() + 26 + np.random.randn(50) * 2  # Ventas en unidades

print("Primeras 5 muestras de datos:")
print(f"Días de lluvia: {X[:5].flatten()}")
print(f"Ventas (unidades): {y[:5]}")


# PASO 2: DIVIDIR DATOS (Entrenamiento vs Prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDatos para entrenar (Train): {len(X_train)} meses")
print(f"Datos para validar (Test): {len(X_test)} meses")


# PASO 3: ENTRENAR EL MODELO
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("\n Modelo entrenado con éxito.")
print(f"   Pendiente aprendida (impacto por día de lluvia): {modelo.coef_[0]:.2f} unidades")
print(f"   Ventas base (intercepto): {modelo.intercept_:.2f} unidades")


# PASO 4: HACER PREDICCIONES
y_pred = modelo.predict(X_test)

print("\n Comparación (Realidad vs Predicción) en meses de prueba:")
for i in range(5):
    print(f"Mes {i+1}: Real={y_test[i]:.2f} uds | Predicho={y_pred[i]:.2f} uds")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Métricas de rendimiento:")
print(f"   Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"   R² (Qué tanto explica el modelo): {r2:.2f} (1.0 es perfecto)")


# PASO 5: VISUALIZAR RESULTADOS
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Datos Entrenamiento')
plt.scatter(X_test, y_test, color='blue', edgecolors='black', label='Datos Prueba (Realidad)')

X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)
plt.plot(X_linea, y_linea, color='red', linewidth=2, label='Línea de Predicción (Modelo)')

plt.title(' Regresión Lineal Simple: Ventas vs Días de Lluvia')
plt.xlabel('Días de lluvia en el mes')
plt.ylabel('Ventas (unidades)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# PASO 6: PRUEBA FINAL
dias_lluvia_julio = np.array([[6]])  # Julio tendrá 6 días de lluvia
ventas_estimadas = modelo.predict(dias_lluvia_julio)

print(f"\n Si en julio llueve 6 días...")
print(f"   El modelo estima ventas de: {ventas_estimadas[0]:.2f} unidades")