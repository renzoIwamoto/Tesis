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
import cv2  # Asegúrate de tener OpenCV instalado

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parámetros generales (asegúrate de que estos coincidan con los de tu modelo base)
MEMORY_SIZE = 1000000  # Tamaño del buffer de replay
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.00025
TRAINING_START = 1000000  # Empezar a actualizar las experiencias después de 1,000,000 de pasos
FRAME_STACK = 4
MAX_STEPS_EPISODE = 50000
EXPLORATION_STEPS = 1000000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
UPDATE_TARGET_FREQUENCY = 1000  # Frecuencia para actualizar la red target
SAVE_FREQUENCY = 10000  # Frecuencia para guardar el modelo
EVALUATION_FREQUENCY = 50000  # Frecuencia para evaluar el agente
NUM_EVALUATION_EPISODES = 10
EPISODES = 100000  # Número máximo de episodios
TOTAL_STEPS_LIMIT = 2000000  # Límite total de pasos
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argumentos
def get_args():
    parser = argparse.ArgumentParser(description='Aprendizaje por demostraciones con DQN')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno de Gym')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Ruta del modelo preentrenado')
    parser.add_argument('--local_folder', type=str, default='results', help='Carpeta para guardar resultados')
    return parser.parse_args()

# Agente DQN
class DQNAgent:
    def __init__(self, state_shape, action_size, device):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.device = device
        self.q_network = self.build_model().to(self.device)
        self.target_q_network = self.build_model().to(self.device)
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
        if len(self.memory) < BATCH_SIZE:
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
        if total_steps < EXPLORATION_STEPS:
            self.epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (total_steps / EXPLORATION_STEPS)
        else:
            self.epsilon = FINAL_EPSILON

    def load_pretrained_model(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.update_target_model()
        logging.info(f"Modelo preentrenado cargado desde {model_path}")

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

# Funciones utilitarias para gráficos y guardado
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_hyperparameters(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Hiperparámetros guardados en {path}")

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp, local_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Gráfico de puntuaciones
    ax1.plot(range(len(scores)), scores, label='Episode Scores', color='gray', alpha=0.5)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()

    # Agregar valores máximo y mínimo al gráfico de puntuaciones
    if scores:
        ax1.axhline(max(scores), color='red', linestyle='--', label='Max Score')
        ax1.axhline(min(scores), color='green', linestyle='--', label='Min Score')
        ax1.legend()

    # Gráfico de valores Q promedio
    ax2.plot(range(len(avg_q_values)), avg_q_values, label='Average Q-value', color='green')
    ax2.set_title(f'{game_name} - Average Q-values per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Q-value')
    ax2.legend()

    # Gráfico de pérdidas
    ax3.plot(range(len(losses)), losses, label='Loss', color='red')
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()

    plt.tight_layout()
    save_path = os.path.join(local_folder, f'training_progress_{game_name}_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Gráfico de progreso de entrenamiento guardado en {save_path}")

def plot_evaluation_scores(evaluation_scores, game_name, timestamp, local_folder):
    steps, scores = zip(*evaluation_scores) if evaluation_scores else ([], [])
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, marker='o')
    plt.title(f'{game_name} - Evaluation Scores')
    plt.xlabel('Total Steps')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    save_path = os.path.join(local_folder, f'evaluation_scores_{game_name}_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Gráfico de puntuaciones de evaluación guardado en {save_path}")

def record_best_run(env, agent, local_folder, game_name, timestamp, num_runs=10):
    best_reward = float('-inf')
    best_video_path = None

    for run in range(num_runs):
        current_video_folder = os.path.join(local_folder, f'video_run_{run}_{timestamp}')
        os.makedirs(current_video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, current_video_folder)

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

        env.close()

        # Comprobar si esta corrida es la mejor
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_video_path = current_video_folder

    # Guardar solo el mejor video
    if best_video_path:
        final_video_folder = os.path.join(local_folder, 'videos', f'best_video_{timestamp}')
        os.makedirs(os.path.dirname(final_video_folder), exist_ok=True)
        os.rename(best_video_path, final_video_folder)
        logging.info(f"Mejor video guardado con recompensa {best_reward} en {final_video_folder}")
    else:
        logging.warning("No se guardó ningún video porque no se realizaron corridas.")

# Evaluación del agente
def evaluate_agent(env, agent, num_episodes):
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No explorar durante la evaluación

    for _ in range(num_episodes):
        state, _ = env.reset(seed=np.random.randint(0, 100000))
        done = False
        episode_reward = 0
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not done:
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    agent.epsilon = original_epsilon  # Restaurar epsilon original
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

# Guardar hiperparámetros
def save_hyperparameters(timestamp, local_folder, config):
    hyperparameters_path = os.path.join(local_folder, f'hyperparameters_{timestamp}.json')
    save_hyperparameters(config, hyperparameters_path)

# Llenar el buffer de replay con el modelo preentrenado
def generate_experience(env, agent, num_steps):
    state, _ = env.reset(seed=np.random.randint(0, 100000))
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    for step in range(num_steps):
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        agent.remember(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
        if done:
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        if (step + 1) % 100000 == 0:
            logging.info(f"Generadas {step + 1} experiencias.")

# Función principal
def main():
    args = get_args()

    # Configuración de rutas
    timestamp = get_timestamp()
    game_name = args.env_name.replace('/', '_')
    local_folder = os.path.join(args.local_folder, f'{game_name}_results_{timestamp}')
    os.makedirs(local_folder, exist_ok=True)
    models_folder = os.path.join(local_folder, 'models')
    replays_folder = os.path.join(local_folder, 'replays')
    videos_folder = os.path.join(local_folder, 'videos')
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(replays_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

    # Guardar hiperparámetros
    config = {
        'ENV_NAME': args.env_name,
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
        'MAX_STEPS_EPISODE': MAX_STEPS_EPISODE
    }
    save_hyperparameters(timestamp, local_folder, config)

    # Crear entorno y agente
    env = gym.make(args.env_name, render_mode="rgb_array", repeat_action_probability=0)
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size, DEVICE)
    agent.load_pretrained_model(args.pretrained_model)

    # Fase 1: Generar experiencias con el modelo preentrenado
    logging.info("Fase 1: Generando experiencias con el modelo preentrenado...")
    generate_experience(env, agent, MEMORY_SIZE)
    logging.info(f"Experiencias generadas: {len(agent.memory)}")

    # Guardar el buffer de replay generado
    replay_path = os.path.join(replays_folder, f'experience_replay_{game_name}_{timestamp}.pkl')
    with open(replay_path, 'wb') as f:
        import pickle
        pickle.dump(agent.memory, f)
    logging.info(f"Buffer de replay guardado en {replay_path}")

    # Fase 2: Entrenar la nueva red con las experiencias generadas
    logging.info("Fase 2: Entrenando la nueva red...")
    scores = []
    avg_q_values_per_episode = []
    losses = []
    evaluation_scores = []
    total_steps = 0

    for episode in range(EPISODES):
        if total_steps >= TOTAL_STEPS_LIMIT:
            logging.info("Se alcanzó el límite total de pasos.")
            break

        state, _ = env.reset(seed=np.random.randint(0, 100000))
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        agent.q_values_episode = []
        done = False

        for step in range(MAX_STEPS_EPISODE):
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= TRAINING_START:
                agent.replay()

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()
                logging.info(f"Red target actualizada en el paso {total_steps}.")

            if total_steps % SAVE_FREQUENCY == 0:
                model_save_path = os.path.join(models_folder, f'dqn_model_{game_name}_{timestamp}_step_{total_steps}.pth')
                agent.save(model_save_path)

            if total_steps % EVALUATION_FREQUENCY == 0:
                mean_reward, std_reward = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                evaluation_scores.append((total_steps, mean_reward))
                logging.info(f"Evaluación en paso {total_steps}: Media={mean_reward}, Std={std_reward}")

            if done:
                break

        scores.append(episode_reward)
        avg_q_value = np.mean(agent.q_values_episode) if agent.q_values_episode else 0
        avg_q_values_per_episode.append(avg_q_value)
        memory_info = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)
        logging.info(f"Ep.: {episode}, Score: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Steps: {step}, Avg Q-val: {avg_q_value:.2f}, Replay: {len(agent.memory)}, Mem Usage: {memory_info:.2f} GB")

        # Actualizar epsilon
        agent.update_epsilon(total_steps)

        # Guardar gráficos cada 200 episodios
        if (episode + 1) % 200 == 0:
            plot_training_progress(scores, avg_q_values_per_episode, losses, game_name, timestamp, local_folder)

    # Evaluación final
    try:
        mean_reward, std_reward = evaluate_agent(env, agent, num_episodes=30)
        logging.info(f"Evaluación Final - Media de Recompensas: {mean_reward}, Desviación Estándar: {std_reward}")
        print(f"Evaluación Final - Media de Recompensas: {mean_reward}, Desviación Estándar: {std_reward}")
        plot_training_progress(scores, avg_q_values_per_episode, losses, game_name, timestamp, local_folder)
        plot_evaluation_scores(evaluation_scores, game_name, timestamp, local_folder)
        final_model_path = os.path.join(models_folder, f'dqn_model_{game_name}_final_{timestamp}.pth')
        agent.save(final_model_path)
    except Exception as e:
        logging.error(f"Error al guardar el modelo o la memoria de experiencia: {e}")

    # Grabación del mejor video
    try:
        env = gym.make(args.env_name, render_mode="rgb_array")
        best_video_path = os.path.join(local_folder, 'videos', f'best_video_{timestamp}')
        record_best_run(env, agent, local_folder, game_name, timestamp, num_runs=10)
    except Exception as e:
        logging.error(f"Error al grabar el mejor video: {e}")

    env.close()

if __name__ == "__main__":
    main()
