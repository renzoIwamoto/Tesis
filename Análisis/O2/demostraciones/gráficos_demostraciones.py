import matplotlib.pyplot as plt
import numpy as np

# Datos de los juegos y modelos
juegos = ['Pong', 'Breakout', 'Frogger', 'Q*bert', 'Alien', 'Mario Bros']
modelos = ['Modelo 1', 'Modelo 2', 'Modelo 3']

# Datos de puntuación media
puntuacion_media_1 = { 
    'Pong': [11.07, 14.63, 17.7],
    'Breakout': [75.07, 33.17, 24.93],
    'Frogger': [40.1, 91.73, 136.43],
    'Q*bert': [1035.83, 546.67, 2687.5],
    'Alien': [252.0, 1025.33, 376.33],
    'Mario Bros': [3093.33, 4080.0, 3493.33]
}

puntuacion_media_2 = {
    'Pong': [19.5, 19.27, 19.8],
    'Breakout': [188.9, 166.03, 170.03],
    'Frogger': [263.26, 247.97, 254.83],
    'Q*bert': [1342.5, 1641.67, 3957.5],
    'Alien': [1784.67, 1317.67, 1023.67],
    'Mario Bros': [9466.67, 12506.07, 8613.33]
}

# Datos de episodios hasta converger
episodios_1 = {
    'Pong': [500, 750, 500],
    'Breakout': [6000, 5000, 6000],
    'Frogger': [3000, 2000, 2500],
    'Q*bert': [0, 0, 0],  # No converge
    'Alien': [0, 6500, 5000],
    'Mario Bros': [5000, 0, 0]  # No converge
}

episodios_2 = {
    'Pong': [1200, 1000, 1500],
    'Breakout': [15000, 19000, 8500],
    'Frogger': [10000, 12500, 11000],
    'Q*bert': [5000, 12000, 0],  # No converge
    'Alien': [11000, 0, 6500],  # No converge en modelo 2
    'Mario Bros': [10000, 0, 10000]  # No converge en modelo 2
}

# Datos de desviación estándar
desviacion_estandar_1 = {
    'Pong': [5.53, 3.36, 2.00],
    'Breakout': [112.43, 53.38, 6.71],
    'Frogger': [47.25, 36.24, 80.27],
    'Q*bert': [1150.93, 144.87, 1854.86],
    'Alien': [120.76, 508.88, 147.61],
    'Mario Bros': [1542.06, 1769.07, 1482.10]
}

desviacion_estandar_2 = {
    'Pong': [1.43, 0.99, 1.19],
    'Breakout': [113.13, 152.17, 138.33],
    'Frogger': [101.26, 69.79, 100.79],
    'Q*bert': [646.89, 290.64, 1555.85],
    'Alien': [554.97, 381.77, 168.37],
    'Mario Bros': [4084.88, 4625.65, 4119.20]
}

# Función para graficar comparativos
def plot_comparative(data_1, data_2, ylabel, title, color_1, color_2):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    for i, juego in enumerate(juegos):
        ax = axs[i // 3, i % 3]
        ax.bar(np.arange(3) - 0.2, data_1[juego], width=0.4, label='Aprendizaje por demostraciones', color=color_1)
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
