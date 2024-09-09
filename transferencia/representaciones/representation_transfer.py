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
import pickle
import psutil
import gc
import cv2
import matplotlib.pyplot as plt
import logging
from gymnasium.wrappers import RecordVideo
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Transferencia de aprendizaje en DQN')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno de destino')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    parser.add_argument('--base_model_game', type=str, required=True, help='Nombre del juego del modelo base')
    parser.add_argument('--base_model_path', type=str, required=True, help='Ruta del modelo preentrenado')
    parser.add_argument('--freeze_conv_layers', action='store_true', help='Congelar capas convolucionales')
    return parser.parse_args()

args = get_args()

# Configuración del entorno y parámetros
ENV_NAME = args.env_name
BASE_MODEL_GAME = args.base_model_game  # Juego del que se carga el modelo base
BASE_MODEL_PATH = args.base_model_path  # Ruta del modelo preentrenado
GAME_NAME = ENV_NAME.split('-')[0].replace('/', '_')
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 100000
BATCH_SIZE = 256
TRAINING_START = 100000
INITIAL_EPSILON = 0.05
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
DEVICE = args.device

print(ENV_NAME)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_FOLDER = '/data/riwamoto'
GAME_FOLDER = os.path.join(BASE_FOLDER, f'{GAME_NAME}_transfer_results')
os.makedirs(GAME_FOLDER, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_FOLDER = os.path.join(SCRIPT_DIR, 'resultados_represetation')
LOCAL_FOLDER = os.path.join(RESULTADOS_FOLDER, f'local_results_{GAME_NAME}_resultados_represetation_transfer_{get_timestamp()}')
os.makedirs(LOCAL_FOLDER, exist_ok=True)

timestamp = get_timestamp()
log_filename = f"{GAME_NAME}_resultados_represetation_transfer_{timestamp}.log"
log_filepath = os.path.join(LOCAL_FOLDER, log_filename)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filepath),
                        logging.StreamHandler()
                    ])


class TransferDQNAgent:
    def __init__(self, state_shape, action_size, base_model_game, base_model_path, freeze_conv_layers, device_id=0):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON

        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)

        # Cargar el modelo base preentrenado
        self.q_network = self.build_model().to(self.device)
        self.load_base_model(base_model_path)

        # Opción para congelar capas convolucionales
        if freeze_conv_layers:
            for name, param in self.q_network.named_parameters():
                if 'conv' in name:
                    param.requires_grad = False

        # Reinicializar la penúltima capa lineal
        nn.init.xavier_uniform_(self.q_network[-3].weight)
        nn.init.zeros_(self.q_network[-3].bias)
        # Crear una nueva capa densa para el espacio de acciones del juego de destino
        self.q_network[-1] = nn.Linear(512, self.action_size).to(self.device)
        
        self.target_q_network = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q_network.parameters()), lr=LEARNING_RATE)

        self.loss_history = []
        self.q_values_history = []
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

    def load_base_model(self, model_path):
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f'Modelo preentrenado cargado desde {model_path}')
        except Exception as e:
            logging.error(f'Error al cargar el modelo preentrenado: {e}')

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


# Preprocesamiento de imágenes
def preprocess_frame(frame):
    gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(np.uint8)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=0)
    return stacked, stacked_frames

def evaluate_agent(env, agent, num_episodes):
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.00  # Establecer epsilon a 0 para la evaluación
    
    for _ in range(num_episodes):
        state, _ = env.reset(seed=np.random.randint(0, 100000))
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    
    agent.epsilon = original_epsilon  # Restaurar el valor original de epsilon
    
    # Calcular media y desviación estándar de las recompensas obtenidas
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return mean_reward, std_reward

def smooth_data(data, window_size=100):
    """Aplica un suavizado por promedio móvil a los datos."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    window_size = min(20, len(scores))
    
    smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    smoothed_avg_q_values = np.convolve(avg_q_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_losses = smooth_data(losses)

    min_scores = np.array([min(scores[i:i+window_size]) for i in range(len(smoothed_scores))])
    max_scores = np.array([max(scores[i:i+window_size]) for i in range(len(smoothed_scores))])
    
    min_q_values = np.array([min(avg_q_values[i:i+window_size]) for i in range(len(smoothed_avg_q_values))])
    max_q_values = np.array([max(avg_q_values[i:i+window_size]) for i in range(len(smoothed_avg_q_values))])

    # Gráfico de puntuaciones
    ax1.plot(range(len(smoothed_scores)), smoothed_scores, label='Average Score', color='blue')
    ax1.fill_between(range(len(smoothed_scores)), 
                     min_scores, 
                     max_scores, 
                     alpha=0.3, color='blue')
    ax1.plot(range(len(scores)), scores, label='Episode Scores', color='gray', alpha=0.5)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()

    ax1.axhline(max(scores), color='red', linestyle='--', label='Max Score')
    ax1.axhline(min(scores), color='green', linestyle='--', label='Min Score')

    ax2.plot(range(len(smoothed_avg_q_values)), smoothed_avg_q_values, label='Average Q-value', color='green')
    ax2.fill_between(range(len(smoothed_avg_q_values)), 
                     min_q_values, 
                     max_q_values, 
                     alpha=0.3, color='green')
    ax2.plot(range(len(avg_q_values)), avg_q_values, label='Episode Q-values', color='gray', alpha=0.5)
    ax2.set_title(f'{game_name} - Average Q-values per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Q-value')
    ax2.legend()

    ax3.plot(range(len(smoothed_losses)), smoothed_losses, label='Smoothed Losses', color='red')
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_FOLDER, f'training_progress_{game_name}_{timestamp}.png'))
    plt.close()

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
    
    with open(os.path.join(LOCAL_FOLDER, f'hyperparameters_{timestamp}.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

def plot_evaluation_scores(evaluation_scores, game_name, timestamp):
    steps, scores = zip(*evaluation_scores)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, marker='o')
    plt.title(f'{game_name} - Evaluation Scores')
    plt.xlabel('Total Steps')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    
    plt.savefig(os.path.join(LOCAL_FOLDER, f'evaluation_scores_{game_name}_{timestamp}.png'))
    plt.close()

def main():
    timestamp = get_timestamp()
    
    MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
    VIDEOS_FOLDER = os.path.join(LOCAL_FOLDER, 'videos')
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)

    save_hyperparameters(timestamp)

    env = gym.make(ENV_NAME, difficulty=DIFFICULTY, render_mode="rgb_array", repeat_action_probability=0)
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n

    agent = TransferDQNAgent(state_shape, action_size, BASE_MODEL_GAME, args.freeze_conv_layers, DEVICE)
    stacked_frames = deque(maxlen=FRAME_STACK)

    scores = []
    total_steps = 0
    avg_q_values_per_episode = []
    losses = []
    evaluation_scores = []

    for episode in range(EPISODES):
        if total_steps >= TOTAL_STEPS_LIMIT:
            break
        
        state, _ = env.reset(seed=np.random.randint(0, 100000))
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        episode_steps = 0
        agent.q_values_episode = []

        for time_step in range(MAX_STEPS_EPISODE):
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if total_steps >= TRAINING_START and total_steps % TRAIN_FREQUENCY == 0:
                agent.replay()
                agent.update_epsilon(total_steps)
                losses.append(agent.loss_history[-1])

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()

            if total_steps % SAVE_FREQUENCY == 0:
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}.pth'))

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score, deviation = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                evaluation_scores.append((total_steps, eval_score))
                logging.info(f"Step: {total_steps}, Evaluation Score: {eval_score}, Desviacion: {deviation}")

            if done:
                break

        avg_q_value = np.mean(agent.q_values_episode)
        avg_q_values_per_episode.append(avg_q_value)
        scores.append(episode_reward)
        memory_info = psutil.virtual_memory()
        logging.info(f"Ep.: {episode}, Score: {episode_reward}, e: {agent.epsilon:.2f}, Steps: {episode_steps}, Avg Q-val: {avg_q_value:.2f}, replay: {len(agent.memory)}, Mem Usage: {memory_info.percent}%")
        torch.cuda.empty_cache()

        if episode % 200 == 0:
            plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp)

    try:
        mean_reward, std_reward = evaluate_agent(env, agent, num_episodes=30)
        logging.info(f"Final Evaluation - Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp)
        plot_evaluation_scores(evaluation_scores, GAME_NAME, timestamp)  
        agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}.pth'))
    except Exception as e:
        logging.error(f"Error al guardar el modelo o la memoria de experiencia: {e}")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join(VIDEOS_FOLDER, f'video_{timestamp}'))
    state, _ = env.reset(seed=np.random.randint(0, 100000))
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    while not done:
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state

if __name__ == "__main__":
    main()
