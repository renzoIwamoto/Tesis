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
    parser = argparse.ArgumentParser(description='Entrenamiento de DQN con Aprendizaje Curricular')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno de Gym')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    parser.add_argument('--difficulty', type=int, default=0, help='Nivel de dificultad en el que se entrenará')
    parser.add_argument('--pretrained_model', type=str, help='Ruta al modelo base preentrenado')
    return parser.parse_args()

args = get_args()

# Configuración del entorno y parámetros
ENV_NAME = args.env_name
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
MIN_REWARD = float('inf')
MAX_REWARD = float('-inf')
DIFFICULTY = args.difficulty
DEVICE = args.device

print(f"Entrenamiento en {ENV_NAME} con dificultad {DIFFICULTY}")

# Funciones para gráficos
def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp, local_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    window_size = min(20, len(scores))  # Usamos los últimos 20 episodios o menos si hay menos datos

    # Calcular promedios móviles
    smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    smoothed_avg_q_values = np.convolve(avg_q_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')

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
def save_hyperparameters(hyperparameters, timestamp, local_folder):
    with open(os.path.join(local_folder, f'hyperparameters_{timestamp}.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

# Preprocesamiento de imágenes
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=0)
    return stacked, stacked_frames

class DQNAgent:
    def __init__(self, state_shape, action_size, device_id=0):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.q_network = self.build_model().to(self.device)
        self.target_q_network = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.loss_history = []
        self.q_values_episode = []

    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            self.q_values_episode.append(torch.max(q_values).item())
            return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

    def update_epsilon(self, step):
        self.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (step * (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS))

    def save(self, filename):
        try:
            torch.save(self.q_network.state_dict(), filename)
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {e}")

    def load(self, model_path):
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f'Modelo preentrenado cargado desde {model_path}')
        except Exception as e:
            logging.error(f'Error al cargar el modelo preentrenado: {e}')

# Entrenamiento en el nivel de dificultad pasado por argumento
def main():
    timestamp = get_timestamp()

    # Definir las carpetas para guardar los modelos y resultados
    BASE_FOLDER = '/data/riwamoto'
    GAME_FOLDER = os.path.join(BASE_FOLDER, f'{GAME_NAME}_results')
    LOCAL_FOLDER = os.path.join(GAME_FOLDER, f'local_results_{GAME_NAME}_{timestamp}')
    MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
    VIDEOS_FOLDER = os.path.join(LOCAL_FOLDER, 'videos')
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(LOCAL_FOLDER, exist_ok=True)

    # Guardar los hiperparámetros
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
        'DIFFICULTY': DIFFICULTY
    }

    save_hyperparameters(hyperparameters, timestamp, LOCAL_FOLDER)

    # Crear el entorno en la dificultad especificada
    env = gym.make(ENV_NAME, difficulty=DIFFICULTY, render_mode="rgb_array", repeat_action_probability=0)
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size, DEVICE)

    # Cargar el modelo preentrenado si se proporciona
    if args.pretrained_model:
        logging.info(f"Cargando el modelo base desde {args.pretrained_model}")
        agent.load(args.pretrained_model)

    scores = []
    total_steps = 0
    avg_q_values_per_episode = []
    losses = []
    evaluation_scores = []

    # Entrenamiento en la dificultad especificada
    logging.info(f"Entrenando en dificultad {DIFFICULTY}")
    for episode in range(EPISODES):
        if total_steps >= TOTAL_STEPS_LIMIT:
            break
        state, _ = env.reset(seed=np.random.randint(0, 100000))
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        for step in range(MAX_STEPS_EPISODE):
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            if total_steps >= TRAINING_START and total_steps % TRAIN_FREQUENCY == 0:
                agent.replay()
                agent.update_epsilon(total_steps)
                losses.append(agent.loss_history[-1])
            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()
            if total_steps % SAVE_FREQUENCY == 0:
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_difficulty_{DIFFICULTY}.pth'))
            if done:
                break
        avg_q_value = np.mean(agent.q_values_episode)
        avg_q_values_per_episode.append(avg_q_value)
        scores.append(episode_reward)
        logging.info(f"Ep: {episode}, Reward: {episode_reward}, Difficulty: {DIFFICULTY}, Total Steps: {total_steps}")

    # Guardar el modelo final
    agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}_difficulty_{DIFFICULTY}.pth'))

    # Graficar progreso
    plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp, LOCAL_FOLDER)
    plot_evaluation_scores(evaluation_scores, GAME_NAME, timestamp, LOCAL_FOLDER)

    logging.info(f"Entrenamiento finalizado en dificultad {DIFFICULTY}")
    env.close()

if __name__ == "__main__":
    main()