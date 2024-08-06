import os

# Definir el directorio base
BASE_FOLDER = '/data/riwamoto'
# Definir el nombre del archivo de prueba
test_file_path = os.path.join(BASE_FOLDER, 'test_file.txt')

# Crear el archivo de prueba y escribir algo en él
try:
    with open(test_file_path, 'w') as file:
        file.write("Este es un archivo de prueba.")
    print(f"Archivo de prueba creado exitosamente en {test_file_path}")
except Exception as e:
    print(f"No se pudo crear el archivo de prueba: {e}")
