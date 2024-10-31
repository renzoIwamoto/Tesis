import numpy as np

# Lista de puntajes (Scores) proporcionados
scores = [
    4925.0, 4725.0, 4725.0, 450.0, 5225.0, 4675.0, 700.0, 4125.0, 1400.0, 1200.0, 
    4725.0, 4650.0, 725.0, 750.0, 975.0, 800.0, 675.0, 1275.0, 4700.0, 4450.0, 
    4700.0, 4800.0, 4625.0, 325.0, 4575.0, 4775.0, 4575.0, 775.0, 550.0, 4650.0, 
    350.0, 4825.0
]

# Calcular la media
mean_score = np.mean(scores)

# Calcular la desviación estándar
std_dev = np.std(scores)

# Calcular el valor mínimo y máximo
min_score = np.min(scores)
max_score = np.max(scores)

# Mostrar los resultados
print(f"Media: {mean_score}")
print(f"Desviación estándar: {std_dev}")
print(f"Valor mínimo: {min_score}")
print(f"Valor máximo: {max_score}")
