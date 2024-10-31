import matplotlib.pyplot as plt
import numpy as np

# Datos de los juegos y modelos
juegos = ['Pong', 'Breakout', 'Frogger', 'Q*bert', 'Alien']
modelos = ['Modelo 1', 'Modelo 2', 'Modelo 3']

# Datos de puntuación media - Aprendizaje curricular (Tabla 20)
puntuacion_media_1 = {
    'Pong': [17.60, 18.8, 17.60],
    'Breakout': [84.4, 45.2, 19.2],
    'Frogger': [172.0, 173.0, 134.4],
    'Q*bert': [4825.0, 3940.0, 4710.0],
    'Alien': [252.0, 1025.33, 376.33]
}

# Datos de puntuación media - Modelo base (Tabla 21)
puntuacion_media_2 = {
    'Pong': [-21.0, -21.0, 18.2],
    'Breakout': [20.4, 30.4, 25.0],
    'Frogger': [53.8, 162.2, 123.4],
    'Q*bert': [1090.0, 1447.5, 1233.33],
    'Alien': [988.33, 783.0, 900.0]
}

# Datos de episodios hasta converger - Aprendizaje curricular (Tabla 20)
episodios_1 = {
    'Pong': [250, 250, 250],
    'Breakout': [0, 8000, 0],  # No converge
    'Frogger': [3500, 4500, 5000],
    'Q*bert': [1000, 750, 1000],
    'Alien': [0, 6500, 5000]  # No converge en algunos casos
}

# Datos de episodios hasta converger - Modelo base (Tabla 21)
episodios_2 = {
    'Pong': [500, 750, 1500],
    'Breakout': [15000, 11000, 10000],
    'Frogger': [0, 15000, 20000],  # No converge
    'Q*bert': [15000, 10000, 7500],
    'Alien': [0, 0, 0]  # No converge en ninguno
}

# Datos de desviación estándar - Aprendizaje curricular (Tabla 20)
desviacion_estandar_1 = {
    'Pong': [0.49, 1.17, 1.02],
    'Breakout': [56.21, 17.88, 14.54],
    'Frogger': [87.24, 89.52, 70.49],
    'Q*bert': [251.50, 1707.75, 408.23],
    'Alien': [120.76, 508.88, 147.61]
}

# Datos de desviación estándar - Modelo base (Tabla 21)
desviacion_estandar_2 = {
    'Pong': [0.0, 0.0, 1.33],
    'Breakout': [12.00, 6.92, 13.39],
    'Frogger': [35.05, 36.05, 39.36],
    'Q*bert': [207.73, 1003.31, 208.20],
    'Alien': [177.73, 291.62, 239.18]
}

# Función para graficar comparativos
def plot_comparative(data_1, data_2, ylabel, title, color_1, color_2):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    for i, juego in enumerate(juegos):
        ax = axs[i // 3, i % 3]
        ax.bar(np.arange(3) - 0.2, data_1[juego], width=0.4, label='Aprendizaje curricular', color=color_1)
        ax.bar(np.arange(3) + 0.2, data_2[juego], width=0.4, label='Modelo base', color=color_2)
        ax.set_title(juego, fontsize=12)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(modelos)
        ax.set_ylabel(ylabel)
        ax.legend()  # Colocar la leyenda dentro de cada gráfico
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Graficar puntuaciones con colores diferentes
fig1 = plot_comparative(puntuacion_media_1, puntuacion_media_2, 'Puntuación Media', 'Comparación de Puntuación Media', 'skyblue', 'salmon')

# Graficar episodios hasta converger con colores diferentes
fig2 = plot_comparative(episodios_1, episodios_2, 'Episodios hasta Converger', 'Comparación de Episodios hasta Converger', 'lightgreen', 'orange')

# Graficar desviación estándar con colores diferentes
fig3 = plot_comparative(desviacion_estandar_1, desviacion_estandar_2, 'Desviación Estándar', 'Comparación de Desviación Estándar', 'lightcoral', 'deepskyblue')

plt.show()
