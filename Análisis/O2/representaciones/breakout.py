import matplotlib.pyplot as plt
import numpy as np

# Datos actualizados de transferencia de representaciones a Breakout
puntuacion_entrenado = {
    'Modelo Base': [188.9],
    'Pong': [176.93],
    'Breakout': [95.17],
    'Frogger': [85.97],
    'Q*bert': [161.90],
    'Alien': [114.53],
    'Mario Bros': [48.03]
}

desviacion_entrenado = {
    'Modelo Base': [113.13], 
    'Pong': [165.63],
    'Breakout': [115.86],
    'Frogger': [134.44],
    'Q*bert': [147.33],
    'Alien': [137.68],
    'Mario Bros': [14.14]
}

# Juegos en el orden deseado
juegos = list(puntuacion_entrenado.keys())

# Colores para cada juego
colores = {
    'Modelo Base': 'lightsteelblue',
    'Pong': 'skyblue',
    'Breakout': 'lightgreen',
    'Frogger': 'orange',
    'Q*bert': 'violet',
    'Alien': 'lightcoral',
    'Mario Bros': 'lightpink'
}

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(14, 8))

# Posiciones en el eje x para cada juego
x_positions = np.arange(len(juegos))

# Graficar cada juego con su respectiva puntuación y desviación estándar
for i, juego in enumerate(juegos):
    ax.bar(x_positions[i], puntuacion_entrenado[juego], width=0.6, yerr=desviacion_entrenado[juego], 
           capsize=5, label=juego, color=colores[juego], edgecolor='black', linewidth=1)
    # Etiquetas de valores encima de cada barra
    ax.text(x_positions[i], puntuacion_entrenado[juego][0] + desviacion_entrenado[juego][0] * 0.1, 
            f'{puntuacion_entrenado[juego][0]:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir línea en la barra del Modelo Base en el valor 89.29
modelo_base_x = x_positions[0]  # Primera posición para el Modelo Base
ax.hlines(89.29, modelo_base_x - 0.3, modelo_base_x + 0.3, color='red', linestyle='--', linewidth=2)
ax.text(modelo_base_x, 89.29 + 5, '89.29', color='red', ha='center', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Breakout', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
