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

# Configuración del entorno y parámetros IceHockey
#ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'IceHockeyDeterministic-v4'
GAME_NAME = ENV_NAME.split('-')[0]
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
BATCH_SIZE = 256
TRAINING_START = 50000
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.05
EXPLORATION_STEPS = 250000
UPDATE_TARGET_FREQUENCY = 5000
SAVE_FREQUENCY = 1000000
EVALUATION_FREQUENCY = 500000
NUM_EVALUATION_EPISODES = 5
EPISODES = 20000
TRAIN_FREQUENCY = 16
MAX_STEPS_EPISODE = 50000
NEGATIVE_REWARD = 0  # Nuevo parámetro para el reward negativo

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Crear la carpeta principal del juego para modelos y replays
BASE_FOLDER = '/data/riwamoto'
GAME_FOLDER = os.path.join(BASE_FOLDER, f'{GAME_NAME}_results')
os.makedirs(GAME_FOLDER, exist_ok=True)

# Carpeta local para logs, gráficos, videos e hiperparámetros dentro de DQN
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio del script actual (DQN)
RESULTADOS_FOLDER = os.path.join(SCRIPT_DIR, 'resultados')
LOCAL_FOLDER = os.path.join(RESULTADOS_FOLDER, f'local_results_{GAME_NAME}_{get_timestamp()}')
os.makedirs(LOCAL_FOLDER, exist_ok=True)

# Configuración del logging
timestamp = get_timestamp()
log_filename = f"{GAME_NAME}_training_{timestamp}.log"
log_filepath = os.path.join(LOCAL_FOLDER, log_filename)

# Configurar logging para consola y archivo
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filepath),
                        logging.StreamHandler()
                    ])

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network = self.build_model().to(self.device)
        self.target_q_network = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        self.loss_history = []
        self.q_values_history = []
        self.q_values_episode = []

    def build_model(self):
        model = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # Cuarta capa convolucional
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # Aplanado de la salida de la última capa convolucional
            nn.Flatten(),
            
            # Primera capa completamente conectada
            nn.Linear(7 * 7 * 128, 512),
            nn.ReLU(),
            
            # Segunda capa completamente conectada
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # Capa de salida
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
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_model()

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
    for _ in range(num_episodes):
        state, _ = env.reset()
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
    return np.mean(total_rewards)

def smooth_data(data, window_size=100):
    """Aplica un suavizado por promedio móvil a los datos."""
    if len(data) < window_size:
        return data  # Devuelve los datos sin suavizar si son insuficientes
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Suavizar datos para mejorar la visualización, sólo si es posible
    if len(scores) >= 100:
        smoothed_scores = smooth_data(scores)
    else:
        smoothed_scores = scores

    if len(avg_q_values) >= 100:
        smoothed_avg_q_values = smooth_data(avg_q_values)
    else:
        smoothed_avg_q_values = avg_q_values

    if len(losses) >= 100:
        smoothed_losses = smooth_data(losses)
    else:
        smoothed_losses = losses

    # Gráfico de puntuaciones
    ax1.plot(range(len(smoothed_scores)), smoothed_scores, label='Smoothed Scores', color='blue', alpha=0.8)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()

    # Gráfico de valores Q promedio
    ax2.plot(range(len(smoothed_avg_q_values)), smoothed_avg_q_values, label='Smoothed Avg Q-values', color='green', alpha=0.8)
    ax2.set_title(f'{game_name} - Average Q-values per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Q-value')
    ax2.legend()

    # Gráfico de pérdidas
    ax3.plot(range(len(smoothed_losses)), smoothed_losses, label='Smoothed Losses', color='red', alpha=0.8)
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
        'TRAIN_FREQUENCY': TRAIN_FREQUENCY,
        'MAX_STEPS_EPISODE': MAX_STEPS_EPISODE,
        'NEGATIVE_REWARD': NEGATIVE_REWARD  # Guardar el nuevo parámetro en los hiperparámetros
    }
    
    with open(os.path.join(LOCAL_FOLDER, f'hyperparameters_{timestamp}.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

def main():
    timestamp = get_timestamp()
    
    MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
    REPLAYS_FOLDER = os.path.join(GAME_FOLDER, 'replays')
    VIDEOS_FOLDER = os.path.join(LOCAL_FOLDER, 'videos')
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPLAYS_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)

    save_hyperparameters(timestamp)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)
    stacked_frames = deque(maxlen=FRAME_STACK)

    scores = []
    total_steps = 0
    avg_q_values_per_episode = []
    losses = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        episode_steps = 0
        agent.q_values_episode = []

        for time_step in range(MAX_STEPS_EPISODE):
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                reward += NEGATIVE_REWARD  # Añadir el reward negativo cuando se llega a done

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
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_{total_steps}.pth'))
                with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}_{total_steps}.pkl'), 'wb') as f:
                    pickle.dump(agent.memory, f)

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                logging.info(f"Step: {total_steps}, Evaluation Score: {eval_score}")

            if done:
                break

        avg_q_value = np.mean(agent.q_values_episode)
        avg_q_values_per_episode.append(avg_q_value)
        scores.append(episode_reward)
        memory_info = psutil.virtual_memory()
        logging.info(f"Ep.: {episode}, Score: {episode_reward}, e: {agent.epsilon:.2f}, Steps: {episode_steps}, Avg Q-val: {avg_q_value:.2f}, replay: {len(agent.memory)}, Mem Usage: {memory_info.percent}%")
        gc.collect()
        torch.cuda.empty_cache()

        if episode % 200 == 0:
            plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp)

    agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}.pth'))
    with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}_final_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(agent.memory, f)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join(VIDEOS_FOLDER, f'video_{timestamp}'))
    state, _ = env.reset()
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    while not done:
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state
    env.close()

if __name__ == "__main__":
    main()
