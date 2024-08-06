import numpy as np
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle
import os
from gymnasium.wrappers import RecordVideo
import datetime

# Configuración del entorno y parámetros
#ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'PacmanDeterministic-v4'
GAME_NAME = ENV_NAME.split('-')[0]
FRAME_STACK = 4                          # Número de frames apilados para representar el estado.
GAMMA = 0.99                             # Factor de descuento para las recompensas futuras
LEARNING_RATE = 0.00025                  # Tasa de aprendizaje para el optimizador.
MEMORY_SIZE = 250000                     # Tamaño de la memoria de experiencia.
BATCH_SIZE = 32
TRAINING_START = 50000                   # Número de pasos antes de comenzar el entrenamiento.
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EXPLORATION_STEPS = 250000 #1000000              # Número de pasos para disminuir epsilon.
UPDATE_TARGET_FREQUENCY = 10000          # Frecuencia para actualizar el modelo objetivo.
SAVE_FREQUENCY = 10000                  # Frecuencia para guardar el modelo.
EVALUATION_FREQUENCY = 50000             # Frecuencia para evaluar el agente.
NUM_EVALUATION_EPISODES = 10             # Número de episodios para la evaluación.
EPISODES = 5000                         # Número total de episodios para el entrenamiento.
TRAIN_FREQUENCY = 4  # Entrenar cada 4 steps

# Configuración de GPU
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

print(physical_devices)

# Crear la carpeta principal del juego
GAME_FOLDER = f'{GAME_NAME}_results'
os.makedirs(GAME_FOLDER, exist_ok=True)

# Crear subcarpetas
MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
REPLAYS_FOLDER = os.path.join(GAME_FOLDER, 'replays')
VIDEOS_FOLDER = os.path.join(GAME_FOLDER, 'videos')
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(REPLAYS_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape                # shape de estado 
        self.action_size = action_size                # numero de acciones posibles
        self.memory = deque(maxlen=MEMORY_SIZE)       # una cola para almacenar las experiencias 
        self.epsilon = INITIAL_EPSILON

        self.model = self.build_model()               # red principal para tomar desiciones
        self.target_model = self.build_model()        # copia de la red que se actualiza periodicamente durante el entrenamiento
        self.update_target_model()

        self.loss_history = []
        self.q_values_history = []

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='huber_loss')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # guarda la transición de experiencia en el experience replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:               # con una probabilidad igual a epsilon
            return random.randrange(self.action_size)      # retornar una acción aleatoria
        q_values = self.model.predict(state, verbose=0)    # calcular Q-values con modelo local
        self.q_values_history.append(np.max(q_values))     # guardar maximo Q-value
        return np.argmax(q_values[0])                      # retorna índice de máximo Q-value

    def replay(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)                   # desempaqueta el minibatch en 5 listas separadas

        states = np.array(states).reshape(BATCH_SIZE, *self.state_shape)                 # Convierte la lista de estados y próximos estados en matrices numpy y las reestructura según self.state_shape. Esto asegura que los datos tengan la forma correcta para ser introducidos en la red neuronal.
        next_states = np.array(next_states).reshape(BATCH_SIZE, *self.state_shape)

        targets = self.model.predict(states, verbose=0)                                  # Predice los valores Q actuales para los estados en el minibatch usando el modelo principal.
        next_q_values = self.target_model.predict(next_states, verbose=0)                # Predice los valores Q para los próximos estados usando el modelo objetivo.

        for i in range(BATCH_SIZE):
            if dones[i]:                                                                 # si el episodio ha terminado después de tomar la acción
                targets[i][actions[i]] = rewards[i]                                      # no hay siguiente estado
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])   # el episodio continua, por lo que se actualiza el valor Q objetivo según la ecuación de Bellman

        history = self.model.fit(states, targets, batch_size=BATCH_SIZE, verbose=0)      # Ajusta los pesos del modelo principal usando los estados y los valores Q objetivos calculados.
        self.loss_history.append(history.history['loss'][0])


    def update_epsilon(self, step):
        self.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (step * (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS))

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)
        self.update_target_model()

@tf.function
def preprocess_frame(frame):
    gray = tf.image.rgb_to_grayscale(frame)
    resized = tf.image.resize(gray, [84, 84])
    normalized = resized / 255.0
    return normalized[:,:,0]  # Asegúrate de que sea 2D

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=-1)  # Asegúrate de que sea 3D (84, 84, 4)
    return stacked, stacked_frames  # Devuelve también stacked_frames para actualizar el deque

def evaluate_agent(env, agent, num_episodes):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(scores)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')

    ax2.plot(avg_q_values)
    ax2.set_title(f'{game_name} - Average Q-values')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Avg Q-value')

    ax3.plot(losses)
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(GAME_FOLDER, f'training_progress_{game_name}_{timestamp}.png'))
    plt.close()

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    timestamp = get_timestamp()
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    state_shape = (84, 84, FRAME_STACK)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)
    stacked_frames = deque(maxlen=FRAME_STACK)
    
    scores = []
    total_steps = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        episode_steps = 0

        for time_step in range(10000):
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                reward = -10  # Reward negativo cuando el episodio termina
            
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            #if episode_steps % 100 == 0:
            #    print(f"Episode: {episode}, Step: {episode_steps}, Total Steps: {total_steps}")

            if len(agent.memory) >= BATCH_SIZE and total_steps % TRAIN_FREQUENCY == 0:
                agent.replay()
                agent.update_epsilon(total_steps)

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()

            if total_steps % SAVE_FREQUENCY == 0:
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_step_{total_steps}_{timestamp}'))
                with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}_step_{total_steps}_{timestamp}.pkl'), 'wb') as f:
                    pickle.dump(agent.memory, f)

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                print(f"Step: {total_steps}, Evaluation Score: {eval_score}")

            if done:
                break

        scores.append(episode_reward)
        print(f"Episode: {episode}, Score: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Steps: {episode_steps}")

        if episode % 10 == 0:
            plot_training_progress(scores, agent.q_values_history, agent.loss_history, GAME_NAME, timestamp)

    # Guardar el modelo final y el experience replay
    agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}'))
    with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}_final_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(agent.memory, f)

    # Grabar video del agente entrenado
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join(VIDEOS_FOLDER, f'video_{timestamp}'))
    state, _ = env.reset()
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    while not done:
        action = agent.act(np.expand_dims(state, axis=0))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state
    env.close()

if __name__ == "__main__":
    main()