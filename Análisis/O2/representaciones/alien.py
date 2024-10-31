import matplotlib.pyplot as plt
import numpy as np

# Datos actualizados de transferencia de representaciones a Alien
puntuacion_entrenado = {
    'Modelo Base': [1784.67],
    'Pong': [1375.67],
    'Breakout': [1196.33],
    'Frogger': [1247.03],
    'Q*bert': [1244.17],
    'Alien': [1125.66],
    'Mario Bros': [1161.33]
}

desviacion_entrenado = {
    'Modelo Base': [554.97], 
    'Pong': [540.30],
    'Breakout': [328.44],
    'Frogger': [449.35],
    'Q*bert': [158.07],
    'Alien': [335.08],
    'Mario Bros': [707.66]
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

# Añadir línea indicadora en "Modelo Base" para el valor 931
modelo_base_index = x_positions[juegos.index('Modelo Base')]
ax.hlines(931, modelo_base_index - 0.3, modelo_base_index + 0.3, colors='red', linestyles='--', linewidth=2)
ax.text(modelo_base_index, 931, '931', ha='center', va='bottom', color='red', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Alien', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
