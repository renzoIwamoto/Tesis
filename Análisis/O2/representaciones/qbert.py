import matplotlib.pyplot as plt
import numpy as np

# Datos del modelo base para Q*bert
puntuacion_base = [3957.5]
desviacion_base = [1555.85]

# Datos actualizados de transferencia de representaciones a Q*bert
puntuacion_entrenado = {
    'Pong': [1208.33],
    'Breakout': [3533.33],
    'Frogger': [1184.17],
    'Q*bert': [1148.33],
    'Alien': [1351.67],
    'Mario Bros': [1115.0]
}

desviacion_entrenado = {
    'Pong': [317.10],
    'Breakout': [1726.99],
    'Frogger': [162.98],
    'Q*bert': [328.44],
    'Alien': [249.24],
    'Mario Bros': [319.08]
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

# Graficar el modelo base
ax.bar(x_positions[0], puntuacion_base[0], width=0.6, yerr=desviacion_base[0], capsize=5,
       label='Modelo Base', color=colores['Modelo Base'], edgecolor='black', linewidth=1.5)
ax.text(x_positions[0], puntuacion_base[0] + desviacion_base[0] * 0.1, f'{puntuacion_base[0]:.2f}', 
        ha='center', va='bottom', fontsize=10)

# Agregar la línea de referencia en 1215 en la barra del modelo base
ax.plot([x_positions[0] - 0.3, x_positions[0] + 0.3], [1215, 1215], color='red', linestyle='--', linewidth=1.5)
ax.text(x_positions[0], 1215 + 12, '1215.0', color='red', ha='center', fontsize=10, fontweight='bold')

# Graficar cada juego con su respectiva puntuación y desviación estándar
for i, juego in enumerate(juegos[1:]):
    ax.bar(x_positions[i + 1], puntuacion_entrenado[juego][0], width=0.6, yerr=desviacion_entrenado[juego][0], 
           capsize=5, label=juego, color=colores[juego], edgecolor='black', linewidth=1)
    ax.text(x_positions[i + 1], puntuacion_entrenado[juego][0] + desviacion_entrenado[juego][0] * 0.1, 
            f'{puntuacion_entrenado[juego][0]:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir detalles al gráfico
ax.set_title('Evaluación de Transferencia de Representaciones a Q*bert', fontsize=18, fontweight='bold')
ax.set_ylabel('Puntuación Media', fontsize=14)
ax.set_xticks(x_positions)
ax.set_xticklabels(juegos, rotation=15, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Juegos", fontsize=10, title_fontsize='12')

# Aplicar cuadrícula
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico con un ajuste de diseño
plt.tight_layout()
plt.show()
