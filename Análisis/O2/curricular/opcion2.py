import matplotlib.pyplot as plt
import numpy as np

# Datos de los juegos y modelos (sin Mario Bros)
juegos = ['Pong', 'Breakout', 'Frogger', 'Q*bert', 'Alien']
modelos = ['Entrenamiento 1', 'Entrenamiento 2', 'Entrenamiento 3']

# Puntuación media y desviación estándar para aprendizaje curricular (Tabla 20)
puntuacion_1 = {
    'Pong': [17.60, 18.8, 17.60],
    'Breakout': [84.4, 45.2, 19.2],
    'Frogger': [172.0, 173.0, 134.4],
    'Q*bert': [4825.0, 3940.0, 4710.0],
    'Alien': [904.0, 786.0, 762.0]
}
desviacion_1 = {
    'Pong': [0.49, 1.17, 1.02],
    'Breakout': [56.21, 17.88, 14.54],
    'Frogger': [87.24, 89.52, 70.49],
    'Q*bert': [251.50, 1707.75, 408.23],
    'Alien': [243.03, 263.03, 234.55]
}

# Puntuación media y desviación estándar para modelo base (Tabla 21)
puntuacion_2 = {
    'Pong': [-21.0, -21.0, 18.2],
    'Breakout': [20.4, 30.4, 25.0],
    'Frogger': [53.8, 162.2, 123.4],
    'Q*bert': [1090.0, 1447.5, 1233.33],
    'Alien': [988.33, 783.0, 900.0]
}
desviacion_2 = {
    'Pong': [0.0, 0.0, 1.33],
    'Breakout': [12.00, 6.92, 13.39],
    'Frogger': [35.05, 36.05, 39.36],
    'Q*bert': [207.73, 1003.31, 208.20],
    'Alien': [177.73, 291.62, 239.18]
}

# Función para graficar las barras con barras de error
def plot_error_bars_comparative(puntuacion_1, puntuacion_2, desviacion_1, desviacion_2, title):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    for i, juego in enumerate(juegos):
        ax = axs[i // 3, i % 3]
        
        # Datos de aprendizaje curricular y modelo base
        p1 = puntuacion_1[juego]
        p2 = puntuacion_2[juego]
        d1 = desviacion_1[juego]
        d2 = desviacion_2[juego]
        
        # Posiciones de las barras
        x = np.arange(len(modelos))
        
        # Ancho de las barras
        width = 0.35

        # Crear las barras con barras de error
        ax.bar(x - width/2, p1, width, yerr=d1, label='Aprendizaje curricular', capsize=5, color='skyblue')
        ax.bar(x + width/2, p2, width, yerr=d2, label='Modelo base', capsize=5, color='salmon')
        
        ax.set_title(juego, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(modelos)
        ax.set_ylabel('Puntuación Media')
        ax.legend()

    # Ocultar la sexta celda (que no será usada)
    axs[1, 2].axis('off')

    # Ajustar el layout para agregar más espacio entre las filas
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)  # Incrementar h_pad para más espacio entre filas
    return fig

# Graficar barras con barras de error para los 5 juegos
plot_error_bars_comparative(puntuacion_1, puntuacion_2, desviacion_1, desviacion_2, 'Comparación de Puntuación Media con Desviación Estándar por Juego')
plt.show()
