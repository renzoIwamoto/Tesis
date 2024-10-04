# Tesis: Evaluación de Técnicas de Transferencia de Aprendizaje en Aprendizaje por Refuerzo Profundo

## Descripción del Proyecto
Este repositorio contiene el código fuente para la tesis **"Evaluación de Técnicas de Transferencia de Aprendizaje en Aprendizaje por Refuerzo Profundo usando Juegos de Atari"**. El proyecto explora la efectividad de la transferencia de representaciones, aprendizaje curricular y aprendizaje por demostraciones utilizando juegos de Atari como entorno de experimentación, con implementaciones en PyTorch.

## Estructura del Proyecto
El código principal se organiza en las siguientes carpetas:

- **modelo base**: Contiene la carpeta con el modelo base implementado
  - **DQN**: Contiene el código de evaluación, pruebas, resultados y scripts para el modelo base.
    - `modelo-base_pytorch.py`: Código principal para entrenar el modelo base en diferentes entornos de juegos de Atari.
    - `eval.py`: Código para evaluar modelos preentrenados. 
- **transferencia/**: Carpeta principal que contiene el código del experimento de transferencia de aprendizaje.
  - **representaciones/**: Incluye scripts, resultados de entrenamiento para cada uno de los juegos usando transferencia de representaciones.
    - `representation_transfer.py`: Script para realizar transferencias de representaciones entre diferentes juegos.
  - **demostraciones**: Incluye scripts, resultados de entrenamiento para cada uno de los juegos usando aprendizaje por demostraciones
    - `demostraciones.py`: Script para transferencias demostrativas entre diferentes juegos.
  - **curricular**: Incluye scripts, resultados de entrenamiento para cada uno de los juegos usando aprendizaje curricular
    - `curricular.py`: Script para realizar transferencias de aprendizaje curricular.

   Para cada uno de los scripts de transferencia de aprendizaje y el modelo base, se generó una carpeta de resultados de entrenamiento donde se puede encontrar el log de entrenamiento, los gráficos de entrenamiento y los hiperparámetros utilizados

## Requisitos Previos
Para ejecutar este proyecto, necesitarás las siguientes dependencias:

- Python 3.8+
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib (para visualización)
  
