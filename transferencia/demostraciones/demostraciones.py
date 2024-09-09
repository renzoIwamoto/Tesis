import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym
import os
import datetime
import logging
import argparse
import json
import matplotlib.pyplot as plt
import cv2

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_args():
    parser = argparse.ArgumentParser(description='Transferencia de aprendizaje en DQN')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno de destino')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    parser.add_argument('--base_model_game', type=str, required=True, help='Nombre del juego del modelo base')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Ruta al modelo preentrenado')
    return parser.parse_args()

args = get_args()

# Configuración del entorno y parámetros
ENV_NAME = args.env_name
BASE_MODEL_GAME = args.base_model_game  # Juego del que se carga el modelo base
GAME_NAME = ENV_NAME.split('-')[0].replace('/', '_')
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 100000
BATCH_SIZE = 256
TRAINING_START = 100000
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.05
EXPLORATION_STEPS = 1000000
UPDATE_TARGET_FREQUENCY = 1000
SAVE_FREQUENCY = 1000000
EVALUATION_FREQUENCY = 500000
NUM_EVALUATION_EPISODES = 5
EPISODES = 100000
TOTAL_STEPS_LIMIT = 10000000
TRAIN_FREQUENCY = 16
MAX_STEPS_EPISODE = 50000
NEGATIVE_REWARD = 0
DIFFICULTY = 0
DEVICE_ID = args.device
EXPERT_STEPS = 1000000  # Pasos para generar experiencias con el modelo experto

# Funciones para gráficos
def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp, local_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    window_size = min(20, len(scores))  # Usamos los últimos 20 episodios o menos si hay menos datos

    # Calcular promedios móviles
    smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    smoothed_avg_q_values = np.convolve(avg_q_values, np.ones(window_size)/window_size, mode='valid')
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

    # Gráfico de puntuaciones
    ax1.plot(range(len(smoothed_scores)), smoothed_scores, label='Average Score', color='blue')
    ax1.plot(range(len(scores)), scores, label='Episode Scores', color='gray', alpha=0.5)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()

    # Gráfico de valores Q promedio
    ax2.plot(range(len(smoothed_avg_q_values)), smoothed_avg_q_values, label='Average Q-value', color='green')
    ax2.plot(range(len(avg_q_values)), avg_q_values, label='Episode Q-values', color='gray', alpha=0.5)
    ax2.set_title(f'{game_name} - Average Q-values per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Q-value')
    ax2.legend()

    # Gráfico de pérdidas
    ax3.plot(range(len(smoothed_losses)), smoothed_losses, label='Smoothed Losses', color='red')
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(local_folder, f'training_progress_{game_name}_{timestamp}.png'))
    plt.close()

def plot_evaluation_scores(evaluation_scores, game_name, timestamp, local_folder):
    steps, scores = zip(*evaluation_scores) if evaluation_scores else ([], [])
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, marker='o')
    plt.title(f'{game_name} - Evaluation Scores')
    plt.xlabel('Total Steps')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    plt.savefig(os.path.join(local_folder, f'evaluation_scores_{game_name}_{timestamp}.png'))
    plt.close()

# Guardar hiperparámetros
def save_hyperparameters(timestamp):
    hyperparameters = {
        'ENV_NAME': ENV_NAME,
        'FRAME_STACK': FRAME_STACK,
        'GAMMA': GAMMA,
        'LEARNING_RATE': LEARNING_RATE,
        'MEMORY_SIZE': MEMORY_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'TRAINING_START': TRAINING_START,
        'INITIAL_EPSILON': INITIAL_EPSILON,
        'FINAL_EPSILON': FINAL_EPSILON,
        'EXPLORATION_STEPS': EXPLORATION_STEPS,
        'UPDATE_TARGET_FREQUENCY': UPDATE_TARGET_FREQUENCY,
        'SAVE_FREQUENCY': SAVE_FREQUENCY,
        'EVALUATION_FREQUENCY': EVALUATION_FREQUENCY,
        'NUM_EVALUATION_EPISODES': NUM_EVALUATION_EPISODES,
        'EPISODES': EPISODES,
        'TOTAL_STEPS_LIMIT': TOTAL_STEPS_LIMIT,
        'TRAIN_FREQUENCY': TRAIN_FREQUENCY,
        'MAX_STEPS_EPISODE': MAX_STEPS_EPISODE,
        'NEGATIVE_REWARD': NEGATIVE_REWARD,
        'OBSERVACION': ""
    }

# Agente DQN actualizado
class DQNAgent:
    def __init__(self, state_shape, action_size, device, trainable=True):
        self.state_shape = state_shape
        self.action_size = action_size
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.trainable = trainable

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.q_network = self.build_model().to(self.device)
        self.target_q_network = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.update_target_model()
        self.loss_history = []
        self.q_values_episode = []

    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, env):
        if random.random() <= self.epsilon:
            return env.action_space.sample()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE or not self.trainable:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.stack(states)).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(np.stack(next_states)).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_q_network(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, total_steps):
        if total_steps < EXPLORATION_STEPS and self.trainable:
            self.epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (total_steps / EXPLORATION_STEPS)
        else:
            self.epsilon = FINAL_EPSILON

    def load_pretrained_model(self, model_path):
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f'Modelo preentrenado cargado desde {model_path}')
        except Exception as e:
            logging.error(f'Error al cargar el modelo preentrenado: {e}')

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        logging.info(f"Modelo guardado en {path}")

# Preprocesamiento de imágenes
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

# Apilar frames
def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=0)
    return stacked, stacked_frames

# Función principal
def main():
    args = get_args()

    # Configuración de rutas
    timestamp = get_timestamp()
    game_name = args.env_name.replace('/', '_')
    local_folder = os.path.join('results', f'{game_name}_results_{timestamp}')
    os.makedirs(local_folder, exist_ok=True)
    models_folder = os.path.join(local_folder, 'models')
    videos_folder = os.path.join(local_folder, 'videos')
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

    save_hyperparameters(timestamp)

    # Crear entorno y agentes
    env = gym.make(args.env_name, render_mode="rgb_array", repeat_action_probability=0)
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n

    # Agente para generar demostraciones (preentrenado)
    pretrained_agent = DQNAgent(state_shape, action_size, DEVICE_ID, trainable=False)
    pretrained_agent.load_pretrained_model(args.pretrained_model)

    # Agente que se entrenará desde cero utilizando las demostraciones
    trained_agent = DQNAgent(state_shape, action_size, DEVICE_ID, trainable=True)
    
    # Fase 1: Generar experiencias con el agente preentrenado
    logging.info("Fase 1: Generando experiencias con el agente preentrenado...")
    total_steps = 0
    scores_fase_1 = []
    scores = []
    avg_q_values_per_episode = []
    losses = []
    evaluation_scores = []
    state, _ = env.reset(seed=np.random.randint(0, 100000))
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while total_steps < EXPERT_STEPS:
        episode_reward_fase_1 = 0  # Recompensa del episodio
        done = False

        while not done:
            action = pretrained_agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Guardar la experiencia en el agente que se va a entrenar
            trained_agent.remember(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()[0]  # Reinicio del entorno si el episodio ha terminado

            episode_reward_fase_1 += reward
            total_steps += 1

            # Si se han acumulado suficientes pasos, el agente empieza a entrenar
            if  total_steps % TRAIN_FREQUENCY == 0 and len(trained_agent.loss_history) > 0:
                trained_agent.replay()
                trained_agent.update_epsilon(total_steps)
                losses.append(trained_agent.loss_history[-1])

            # Actualizar el modelo objetivo cada cierto número de pasos
            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                trained_agent.update_target_model()

            # Si el episodio ha terminado, reiniciamos el entorno y hacemos la impresión
            if done:
                scores_fase_1.append(episode_reward_fase_1)
                logging.info(f"Fase 1 - Pasos totales: {total_steps}, Recompensa del episodio: {episode_reward_fase_1}")
                # Reiniciar el estado para el siguiente episodio
                state, _ = env.reset(seed=np.random.randint(0, 100000))
                state, stacked_frames = stack_frames(stacked_frames, state, True)

    logging.info(f"Experiencias generadas: {len(trained_agent.memory)}")

    # Fase 2: Entrenamiento del agente desde cero con las demostraciones
    logging.info("Fase 2: Entrenando la nueva red...")


    for episode in range(EPISODES):
        if total_steps >= TOTAL_STEPS_LIMIT:
            break

        state, _ = env.reset(seed=np.random.randint(0, 100000))
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        done = False

        for step in range(MAX_STEPS_EPISODE):
            action = trained_agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            trained_agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= TRAINING_START and total_steps % TRAIN_FREQUENCY == 0:
                trained_agent.replay()
                trained_agent.update_epsilon(total_steps)
                losses.append(trained_agent.loss_history[-1])

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                trained_agent.update_target_model()

            if done:
                break

        scores.append(episode_reward)
        avg_q_value = np.mean(trained_agent.q_values_episode) if trained_agent.q_values_episode else 0
        avg_q_values_per_episode.append(avg_q_value)

        if total_steps % SAVE_FREQUENCY == 0:
            model_save_path = os.path.join(models_folder, f'dqn_model_{game_name}_{timestamp}_step_{total_steps}.pth')
            trained_agent.save(model_save_path)

    # Graficar resultados
    plot_training_progress(scores, avg_q_values_per_episode, losses, game_name, timestamp, local_folder)
    plot_evaluation_scores(evaluation_scores, game_name, timestamp, local_folder)

    env.close()

if __name__ == "__main__":
    main()
#