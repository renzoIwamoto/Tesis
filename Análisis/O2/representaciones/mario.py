import matplotlib.pyplot as plt
import numpy as np

# Datos del modelo base para Mario Bros
puntuacion_base = [12506.07]
desviacion_base = [4625.65]

# Datos actualizados de transferencia de representaciones a Mario Bros
puntuacion_entrenado = {
    'Pong': [6240.0],
    'Breakout': [4075.0],
    'Frogger': [4967.74],
    'Q*bert': [5173.33],
    'Alien': [4566.67],
    'Mario Bros': [3974.19]
}

desviacion_entrenado = {
    'Pong': [3615.67],
    'Breakout': [1803.98],
    'Frogger': [3254.89],
    'Q*bert': [2889.86],
    'Alien': [2165.93],
    'Mario Bros': [1418.36]
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

# Graficar cada juego con su respectiva puntuación y desviación estándar
ax.bar(x_positions[0], puntuacion_base[0], width=0.6, yerr=desviacion_base[0], capsize=5,
       label='Modelo Base', color=colores['Modelo Base'], edgecolor='black', linewidth=1.5)
ax.text(x_positions[0], puntuacion_base[0] + desviacion_base[0] * 0.1, f'{puntuacion_base[0]:.2f}', 
        ha='center', va='bottom', fontsize=10)

# Añadir línea y etiqueta para el valor de referencia
ax.plot([x_positions[0] - 0.3, x_positions[0] + 0.3], [3587.10, 3587.10], color='red', linestyle='--', linewidth=1.5)
ax.text(x_positions[0], 3587.10 + 100, '3587.10', color='red', ha='center', fontsize=10, fontweight='bold')

for i, juego in enumerate(juegos[1:]):
    ax.bar(x_positions[i + 1], puntuacion_entrenado[juego][0], width=0.6, yerr=desviacion_entrenado[juego][0], 
           capsize=5, label=juego, color=colores[juego], edgecolor='black', linewidth=1)
    ax.text(x_positions[i + 1], puntuacion_entrenado[juego][0] + desviacion_entrenado[juego][0] * 0.1, 
            f'{puntuacion_entrenado[juego][0]:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Mario Bros', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
