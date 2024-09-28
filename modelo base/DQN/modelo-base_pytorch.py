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
import psutil
import logging
from gymnasium.wrappers import RecordVideo
import json
import argparse
import  utils

### promediar entre 10 y 20 últimos episodios (agregar el valor máximo y mínimo) (desviación estándar con barras)
### 3 corridas por cada juego
### revisar seed del entorno para que no repita lo mismo
### en el entrenamiento también usar el seed aleatorio

def get_args():
    parser = argparse.ArgumentParser(description='Entrenamiento de DQN')
    parser.add_argument('--env_name', type=str, default='ALE/Frogger-v5', help='Nombre del entorno de Gym')
    parser.add_argument('--device', type=int, default=0, help='ID de la GPU a utilizar')
    return parser.parse_args()

args = get_args()
# Configuración del entorno y parámetros
ENV_NAME = args.env_name # Breakout - Qbert - ALE/MarioBros-v5 - Pong - Alien - Frogger
GAME_NAME = ENV_NAME.split('-')[0].replace('/', '_')  # Reemplazar '/' con '_'
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 100000
BATCH_SIZE = 256
TRAINING_START = 100000
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.05           # podría variar entre juegos
EXPLORATION_STEPS = 1000000
UPDATE_TARGET_FREQUENCY = 1000 # 1000, 5000, 2500
SAVE_FREQUENCY = 1000000
EVALUATION_FREQUENCY = 500000
NUM_EVALUATION_EPISODES = 5
EPISODES = 100000  # Límite de episodios
TOTAL_STEPS_LIMIT = 10000000  # Límite de pasos totales
TRAIN_FREQUENCY = 16
MAX_STEPS_EPISODE = 50000
NEGATIVE_REWARD = 0  # Nuevo parámetro para el reward negativo
MIN_REWARD = float('inf')
MAX_REWARD = float('-inf')
DIFFICULTY = 1
DEVICE=args.device

print(ENV_NAME)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Crear la carpeta principal del juego para modelos y replays
BASE_FOLDER = '/data/riwamoto'
GAME_FOLDER = os.path.join(BASE_FOLDER, f'{GAME_NAME}_results')
os.makedirs(GAME_FOLDER, exist_ok=True)

# Crear la carpeta local para logs, gráficos, videos e hiperparámetros dentro de DQN
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio del script actual (DQN)
RESULTADOS_FOLDER = os.path.join(SCRIPT_DIR, 'resultados')
LOCAL_FOLDER = os.path.join(RESULTADOS_FOLDER, f'local_results_{GAME_NAME}_{get_timestamp()}')
os.makedirs(LOCAL_FOLDER, exist_ok=True)  # Asegurarse de que la carpeta exista

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
        #self.optimizer = optim.RMSprop(self.q_network.parameters(), 
        #                       lr=0.00025, 
        #                       momentum=0.95, 
        #                       alpha=0.95, 
        #                       eps=0.01)


        self.loss_history = []
        self.q_values_history = []
        self.q_values_episode = []

# se puede probar batch normalization cada dos convolucionales y en la primera densa
# pruebas de sensibilidad: permitir ajuste de pesos cada cierta cantidad de épocas
# probar usar dropout
    def build_model(self):
        model = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # probar descongelar la última convolucional
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            # agregar una capa densa extra de 128
            nn.Linear(512, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        #global MIN_REWARD, MAX_REWARD
        #MIN_REWARD = min(MIN_REWARD, reward)
        #MAX_REWARD = max(MAX_REWARD, reward)
        
        # Normalizar el reward
        #if MAX_REWARD > MIN_REWARD and MAX_REWARD != 0:
        #    normalized_reward = (reward) / (MAX_REWARD)
        #else:
        #    normalized_reward = reward
        
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
        if not isinstance(model_path, str) or not os.path.isfile(model_path):
            logging.error(f"Ruta del modelo inválida o el archivo no existe: {model_path}")
            return
        try:
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f'Modelo preentrenado cargado desde {model_path}')
        except Exception as e:
            logging.error(f'Error al cargar el modelo preentrenado: {e}')


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
        'TOTAL_STEPS_LIMIT': TOTAL_STEPS_LIMIT,  # Guardar el nuevo parámetro en los hiperparámetros
        'TRAIN_FREQUENCY': TRAIN_FREQUENCY,
        'MAX_STEPS_EPISODE': MAX_STEPS_EPISODE,
        'NEGATIVE_REWARD': NEGATIVE_REWARD,  # Guardar el nuevo parámetro en los hiperparámetros
        'OBSERVACION': ""
    }
    
    with open(os.path.join(LOCAL_FOLDER, f'hyperparameters_{timestamp}.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

def record_best_run(env, agent, num_runs=10, LOCAL_FOLDER='/'):
    best_reward = float('-inf')
    best_video_path = None
    timestamp = get_timestamp()

    for run in range(num_runs):
        current_video_folder = os.path.join(LOCAL_FOLDER, f'video_run_{run}_{timestamp}')
        os.makedirs(current_video_folder, exist_ok=True)
        env = RecordVideo(env, current_video_folder)

        state, _ = env.reset(seed=np.random.randint(0, 100000))
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = utils.stack_frames(stacked_frames, state, True, FRAME_STACK)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = utils.stack_frames(stacked_frames, next_state, False, FRAME_STACK)
            state = next_state
            episode_reward += reward

        env.close()

        # Comprobar si esta corrida es la mejor
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_video_path = current_video_folder

    # Guardar solo el mejor video
    if best_video_path:
        final_video_folder = os.path.join(os.path.join(LOCAL_FOLDER, 'videos'), f'best_video_{timestamp}')
        os.rename(best_video_path, final_video_folder)
        logging.info(f"Best video saved with reward {best_reward} at {final_video_folder}")
    else:
        logging.warning("No video was saved because no runs were performed.")

def main():
    timestamp = get_timestamp()
    
    MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
    VIDEOS_FOLDER = os.path.join(LOCAL_FOLDER, 'videos')
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)

    save_hyperparameters(timestamp)

    env = gym.make(ENV_NAME, difficulty=DIFFICULTY,render_mode="rgb_array", repeat_action_probability=0)
    state_shape = (FRAME_STACK, 84, 84)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size, DEVICE)
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
        state, stacked_frames = utils.stack_frames(stacked_frames, state, True, FRAME_STACK)
        episode_reward = 0
        episode_steps = 0
        agent.q_values_episode = []

        # Inicializar el número de vidas
        #lives = lives = env.unwrapped.ale.lives()


        for time_step in range(MAX_STEPS_EPISODE):
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            #if done:
            #    reward += NEGATIVE_REWARD  # Añadir el reward negativo cuando se llega a done

                        # Verificar si se ha perdido una vida
            #current_lives = env.unwrapped.ale.lives()
            #if current_lives < lives:
            #    reward += -10  # Aplicar el reward negativo
            #    lives = current_lives  # Actualizar el número de vidas

            next_state, stacked_frames = utils.stack_frames(stacked_frames, next_state, False, FRAME_STACK)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            # Imprimir el total de steps cada 100,000 steps
            if total_steps % 100000 == 0:
                logging.info(f"Total steps: {total_steps}")

            if total_steps >= TRAINING_START and total_steps % TRAIN_FREQUENCY == 0:
                agent.replay()
                agent.update_epsilon(total_steps)
                losses.append(agent.loss_history[-1])

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()

            if total_steps % SAVE_FREQUENCY == 0:
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}.pth'))
                #with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}.pkl'), 'wb') as f:
                    #pickle.dump(agent.memory, f)

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score, deviation = utils.evaluate_agent(env, agent, NUM_EVALUATION_EPISODES, FRAME_STACK)
                evaluation_scores.append((total_steps, eval_score))  # Guarda el score con el número de pasos
                logging.info(f"Step: {total_steps}, Evaluation Score: {eval_score}, Desviacion: {deviation}")
                #torch.cuda.empty_cache()  # Limpiar la caché de la GPU

            if done:
                break

        avg_q_value = np.mean(agent.q_values_episode)
        avg_q_values_per_episode.append(avg_q_value)
        scores.append(episode_reward)
        memory_info = psutil.virtual_memory()
        logging.info(f"Ep.: {episode}, Score: {episode_reward}, e: {agent.epsilon:.2f}, Steps: {episode_steps}, Avg Q-val: {avg_q_value:.2f}, replay: {len(agent.memory)}, Mem Usage: {memory_info.percent}%")
        torch.cuda.empty_cache()

        if episode % 200 == 0:
            utils.plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp, LOCAL_FOLDER)
            #gc.collect()

    try:
        mean_reward, std_reward = utils.evaluate_agent(env, agent, num_episodes=30, FRAME_STACK=FRAME_STACK)
        logging.info(f"Final Evaluation - Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        print(f"Final Evaluation - Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        utils.plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp, LOCAL_FOLDER)
        utils.plot_evaluation_scores(evaluation_scores, GAME_NAME, timestamp, LOCAL_FOLDER)  
        agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}.pth'))
        #with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}_final_{timestamp}.pkl'), 'wb') as f:
        #    pickle.dump(agent.memory, f)
    except Exception as e:
        logging.error(f"Error al guardar el modelo o la memoria de experiencia: {e}")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join(VIDEOS_FOLDER, f'video_{timestamp}'))
    state, _ = env.reset(seed=np.random.randint(0, 100000))
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = utils.stack_frames(stacked_frames, state, True, FRAME_STACK)
    done = False
    while not done:
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = utils.stack_frames(stacked_frames, next_state, False, FRAME_STACK)
        state = next_state

     # Grabación del mejor video en 10 intentos
    record_best_run(env, agent, num_runs=10, LOCAL_FOLDER=LOCAL_FOLDER)

    env.close()

   

if __name__ == "__main__":
    main()
