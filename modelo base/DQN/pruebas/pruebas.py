import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Lista de juegos a probar
games = [
    
    'SpaceInvadersDeterministic-v4',
    'BoxingDeterministic-v4',
    'IceHockeyDeterministic-v4',
    'QbertDeterministic-v4',
    'BreakoutDeterministic-v4',
    'PongDeterministic-v4'
]

# Función para probar ale.lives() en diferentes juegos
def test_ale_lives(games):
    for game in games:
        try:
            env = gym.make(game)
            env.reset()
            lives = env.ale.lives()
            print(f"Game: {game}, Initial lives: {lives}")
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, done, _, _ = env.step(action)
                current_lives = env.ale.lives()
                print(f"Game: {game}, Current lives: {current_lives}")
            env.close()
        except AttributeError as e:
            print(f"Game: {game} does not support ale.lives(). Error: {e}")
        except Exception as e:
            print(f"An error occurred with game: {game}. Error: {e}")

# Ejecutar la función
test_ale_lives(games)

# Lista de combinaciones de dificultad y modo a comparar
combinations = [
    {'difficulty': 0, 'mode': 0},
    {'difficulty': 1, 'mode': 0},
    {'difficulty': 0, 'mode': 4},
    {'difficulty': 1, 'mode': 4},
    {'difficulty': 0, 'mode': 8},
    {'difficulty': 1, 'mode': 8},
]

def run_env(env_name, difficulty, mode, num_episodes=1):
    env = gym.make(env_name, difficulty=difficulty, mode=mode, render_mode="rgb_array")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frames = []
        
        while not done:
            action = env.action_space.sample()  # Acción aleatoria
            next_state, reward, terminated, truncated, _ = env.step(action)
            frames.append(next_state)
            total_reward += reward
            done = terminated or truncated
            
        # Mostrar la recompensa total por episodio
        print(f"Env: {env_name}, Difficulty: {difficulty}, Mode: {mode}, Episode: {episode+1}, Total Reward: {total_reward}")
        env.close()
        
        # Mostrar algunas imágenes del juego
        plt.figure(figsize=(10, 5))
        for i, frame in enumerate(frames[:4]):  # Mostrar los primeros 4 frames
            plt.subplot(1, 4, i+1)
            plt.imshow(frame)
            plt.title(f"Frame {i+1}")
            plt.axis('off')
        plt.suptitle(f"{env_name} - Difficulty {difficulty} - Mode {mode}")
        plt.show()

# Ejemplo de uso
env_name = 'ALE/Breakout-v5'
for combo in combinations:
    run_env(env_name, difficulty=combo['difficulty'], mode=combo['mode'], num_episodes=1)

def preprocess_frame(frame):
    # Mostrar el frame original
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title("Original Frame")
    plt.axis('off')
    
    # Convertir a escala de grises
    gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(np.uint8)
    
    # Redimensionar a 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Mostrar el frame procesado
    plt.subplot(1, 2, 2)
    plt.imshow(resized, cmap='gray')
    plt.title("Processed Frame")
    plt.axis('off')
    plt.show()
    
    # Normalizar
    return resized / 255.0

# Crear el entorno de BreakoutDeterministic-v4
env = gym.make('BreakoutDeterministic-v4', render_mode='rgb_array')

# Reiniciar el entorno para obtener el primer frame
state, _ = env.reset()

# Realizar un paso en el entorno para obtener un nuevo frame
action = env.action_space.sample()  # Selecciona una acción aleatoria
next_state, _, _, _, _ = env.step(action)

# Procesar y mostrar el frame
processed_frame = preprocess_frame(next_state)

# Cerrar el entorno
env.close()


# Crear los entornos
env_no_skip = gym.make('BreakoutNoFrameskip-v4', render_mode="rgb_array")
env_skip = gym.make('BreakoutDeterministic-v4', render_mode="rgb_array")

def check_frame_skip(env, num_steps=4):
    state, _ = env.reset()
    plt.figure(figsize=(10, 5))
    
    for i in range(num_steps):
        # Seleccionar acción aleatoria
        action = env.action_space.sample()
        
        # Ejecutar la acción en el entorno
        next_state, reward, done, truncated, info = env.step(action)
        
        # Imprimir el shape del estado
        print(f"Step {i+1}, State shape: {next_state.shape}")
        
        # Mostrar la imagen del estado
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(next_state)
        plt.title(f'Step {i+1}')
        plt.axis('off')
        
        if done:
            break
    
    plt.show()

print("BreakoutNoFrameskip-v4 (Sin Frame Skip)")
check_frame_skip(env_no_skip, num_steps=4)

print("BreakoutDeterministic-v4 (Con Frame Skip)")
check_frame_skip(env_skip, num_steps=4)

env_no_skip.close()
env_skip.close()
