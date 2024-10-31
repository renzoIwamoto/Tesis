import matplotlib.pyplot as plt
import numpy as np

# Datos del modelo base para Frogger
puntuacion_base = [263.26]
desviacion_base = [101.26]

# Datos actualizados de transferencia de representaciones a Frogger
puntuacion_entrenado = {
    'Pong': [131.8],
    'Breakout': [107.78],
    'Frogger': [106.33],
    'Q*bert': [117.83],
    'Alien': [100.53],
    'Mario Bros': [87.1]
}

desviacion_entrenado = {
    'Pong': [50.03],
    'Breakout': [30.99],
    'Frogger': [64.06],
    'Q*bert': [51.03],
    'Alien': [39.54],
    'Mario Bros': [37.75]
}

# Juegos en el orden deseado
juegos = ['Modelo Base', 'Pong', 'Breakout', 'Frogger', 'Q*bert', 'Alien', 'Mario Bros']

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

# Graficar el modelo base con su respectiva puntuación y desviación estándar
ax.bar(x_positions[0], puntuacion_base[0], width=0.6, yerr=desviacion_base[0], capsize=5,
       label='Modelo Base', color=colores['Modelo Base'], edgecolor='black', linewidth=1.5)
ax.text(x_positions[0], puntuacion_base[0] + desviacion_base[0] * 0.1, f'{puntuacion_base[0]:.2f}', 
        ha='center', va='bottom', fontsize=10)

# Añadir la línea horizontal en la barra del modelo base
ax.plot([x_positions[0] - 0.3, x_positions[0] + 0.3], [156.16, 156.16], color='red', linestyle='--', linewidth=1.5)
ax.text(x_positions[0], 156.16 + 5, '156.16', color='red', ha='center', fontsize=10)

# Graficar las barras para cada juego con sus desviaciones estándar
for i, juego in enumerate(juegos[1:]):
    ax.bar(x_positions[i + 1], puntuacion_entrenado[juego][0], width=0.6, yerr=desviacion_entrenado[juego][0], 
           capsize=5, label=juego, color=colores[juego], edgecolor='black', linewidth=1)
    ax.text(x_positions[i + 1], puntuacion_entrenado[juego][0] + desviacion_entrenado[juego][0] * 0.1, 
            f'{puntuacion_entrenado[juego][0]:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Frogger', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
