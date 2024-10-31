import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Datos específicos de Pong
data = {
    'EpisodiosHastaConverger': [
        500, 750, 500, 
        500, 750, 500, 
        250, 250, 250, 
        500, 750, 1500, 
        4250, 4750, 5000, 
        7000, 8000, 9750, 
        650, 750, 650, 
        10000, 11750, 
        5800, 6250, 
        7000, 5000
    ],
    'PuntuacionMedia': [
        11.07, 14.63, 17.7, 
        17.67, 18.22, 18.47, 
        17.6, 18.8, 17.6, 
        -21.0, -21.0, 18.2, 
        1089.66, 1129.00, 1375.67, 
        31.2, 176.93, 96.0, 
        17.5, 19.77, 19.47, 
        88.9, 78.9, 
        1197.5, 1188.33, 
        6240.0, 4053.33
    ]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Ajuste del modelo de regresión lineal
X = df[['EpisodiosHastaConverger']]
y = df['PuntuacionMedia']
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Calcular R^2 y la correlación
r2 = r2_score(y, y_pred)
correlation = df['EpisodiosHastaConverger'].corr(df['PuntuacionMedia'])

# Gráfico de dispersión con línea de regresión
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='EpisodiosHastaConverger', y='PuntuacionMedia', color='blue', s=50, alpha=0.7, edgecolor='k')
plt.plot(df['EpisodiosHastaConverger'], y_pred, color='red', linestyle='--', linewidth=2, label=f'Regresión Lineal (R² = {r2:.2f}, Correlación = {correlation:.2f})')

# Configuración del título y etiquetas
plt.title('Relación entre Episodios hasta Converger y Puntuación Media para *Pong*', fontsize=15)
plt.xlabel('Episodios hasta Converger', fontsize=12)
plt.ylabel('Puntuación Media', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
