import matplotlib.pyplot as plt
import numpy as np

# Datos del modelo base para Pong
puntuacion_base = [19.8]
desviacion_base = [1.19]

# Datos actualizados de transferencia de representaciones a Pong
puntuacion_entrenado = {
    'Pong': [19.77],
    'Breakout': [19.83],
    'Frogger': [18.7],
    'Q*bert': [19.13],
    'Alien': [19.20],
    'Mario Bros': [18.83]
}

desviacion_entrenado = {
    'Pong': [1.33],
    'Breakout': [1.07],
    'Frogger': [1.70],
    'Q*bert': [1.59],
    'Alien': [1.56],
    'Mario Bros': [2.03]
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

# Añadir línea de referencia y etiqueta para el valor 19.62
ax.plot([x_positions[0] - 0.3, x_positions[0] + 0.3], [19.62, 19.62], color='red', linestyle='--', linewidth=1.5)
ax.text(x_positions[0], 19.62 - 0.5, '19.62', color='red', ha='center', fontsize=10, fontweight='bold')

# Graficar los juegos con sus respectivas puntuaciones y desviaciones estándar
for i, juego in enumerate(juegos[1:]):
    ax.bar(x_positions[i + 1], puntuacion_entrenado[juego][0], width=0.6, yerr=desviacion_entrenado[juego][0], 
           capsize=5, label=juego, color=colores[juego], edgecolor='black', linewidth=1)
    ax.text(x_positions[i + 1], puntuacion_entrenado[juego][0] + desviacion_entrenado[juego][0] * 0.1, 
            f'{puntuacion_entrenado[juego][0]:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Pong', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
