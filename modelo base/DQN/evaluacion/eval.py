import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import os
import argparse
import logging
from collections import deque
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='Evaluación de un modelo DQN preentrenado')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno para evaluación')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Ruta del modelo preentrenado')
    return parser.parse_args()

args = get_args()

# Configuración
ENV_NAME = args.env_name
GAME_NAME = ENV_NAME.split('-')[0].replace('/', '_')
DEVICE = args.device
PRETRAINED_MODEL_PATH = args.pretrained_model
FRAME_STACK = 4
NUM_EVALUATION_EPISODES = 10

# Configuración de GPU
device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clase del agente para evaluación
class DQNAgent:
    def __init__(self, state_shape, action_size, device):
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.q_network = self.build_model().to(self.device)
        self.epsilon = 0.0

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

    def load_model(self, model_path):
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f'Modelo preentrenado cargado desde {model_path}')
        except Exception as e:
            logging.error(f'Error al cargar el modelo preentrenado: {e}')
    
    def select_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            #self.q_values_episode.append(torch.max(q_values).item())
            return np.argmax(q_values.cpu().data.numpy())

# Preprocesamiento de frames
def preprocess_frame(frame):
    gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(np.uint8)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0


# Función para apilar frames
def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=0)
    return stacked, stacked_frames

# Función de evaluación
def evaluate_agent(env, agent, num_episodes):
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.00  # Establecer epsilon a 0 para la evaluación
    for episode in range(num_episodes):
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
        logging.info(f'Episodio {episode + 1}/{num_episodes} - Recompensa: {episode_reward}')

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logging.info(f'Recompensa media: {mean_reward}, Desviación estándar: {std_reward}')
    return mean_reward, std_reward

# Función principal
def main():
    # Crear el entorno
    env = gym.make(ENV_NAME, render_mode="rgb_array", repeat_action_probability=0.025)
    action_size = env.action_space.n
    state_shape = (FRAME_STACK, 84, 84)

    # Crear el agente y cargar el modelo preentrenado
    agent = DQNAgent(state_shape, action_size, device)
    agent.load_model(PRETRAINED_MODEL_PATH)

    # Evaluar el agente
    evaluate_agent(env, agent, 30)

if __name__ == "__main__":
    main()
