
# Tesis: Evaluación de Técnicas de Transferencia de Aprendizaje en Aprendizaje por Refuerzo Profundo

## Descripción del Proyecto
Este repositorio contiene el código fuente para la tesis **"Evaluación de Técnicas de Transferencia de Aprendizaje en Aprendizaje por Refuerzo Profundo usando Juegos de Atari"**. El proyecto explora la efectividad de diferentes técnicas de transferencia de aprendizaje utilizando juegos de Atari como entorno de experimentación, con implementaciones en PyTorch.

## Estructura del Proyecto
El código principal se organiza en las siguientes carpetas:

- **transferencia/**: Carpeta principal que contiene el código del experimento de transferencia de aprendizaje.
  - **representaciones/**: Incluye scripts para la representación de estados de los juegos y la aplicación de las técnicas de transferencia.
  - **models/**: Contiene los modelos preentrenados utilizados para la transferencia de aprendizaje.
  - **results/**: Directorio donde se guardan los resultados de los experimentos, como gráficos y métricas.
  - **utils/**: Funciones auxiliares y utilidades para manejo de datos y configuraciones.

### Archivos Clave:
- **main.py**: Script principal para ejecutar los experimentos.
- **config.py**: Archivo de configuración para ajustar los parámetros de los experimentos.
- **utils.py**: Funciones auxiliares como carga de modelos y procesamiento de datos.

## Requisitos Previos
Este proyecto requiere las siguientes dependencias:

- Python 3.8+
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib

Instala las dependencias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
