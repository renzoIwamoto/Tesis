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
from gymnasium.wrappers import RecordVideo

# Configuración del entorno y parámetros
ENV_NAME = 'BreakoutDeterministic-v4'  # Cambiado a entorno determinista
GAME_NAME = ENV_NAME.split('-')[0]
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 200000
BATCH_SIZE = 128
TRAINING_START = 50000
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.05
EXPLORATION_STEPS = 500000
UPDATE_TARGET_FREQUENCY = 160
SAVE_FREQUENCY = 50000
EVALUATION_FREQUENCY = 100000
NUM_EVALUATION_EPISODES = 10
EPISODES = 15000
TRAIN_FREQUENCY = 16
MAX_STEPS_EPISODE = 50000

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Crear la carpeta principal del juego
BASE_FOLDER = '/data/riwamoto'
GAME_FOLDER = os.path.join(BASE_FOLDER, f'{GAME_NAME}_results')
os.makedirs(GAME_FOLDER, exist_ok=True)

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
    stacked = np.stack(stacked_frames, axis=0)  # Stack frames along the first dimension
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

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp, run_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(scores)
    ax1.set_title(f'{game_name} - Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')

    ax2.plot(avg_q_values)
    ax2.set_title(f'{game_name} - Average Q-values per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Q-value')

    ax3.plot(losses)
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, f'training_progress_{game_name}_{timestamp}.png'))
    plt.close()

def main():
    timestamp = get_timestamp()
    RUN_FOLDER = os.path.join(GAME_FOLDER, f'run_{timestamp}')
    os.makedirs(RUN_FOLDER, exist_ok=True)

    MODELS_FOLDER = os.path.join(RUN_FOLDER, 'models')
    REPLAYS_FOLDER = os.path.join(RUN_FOLDER, 'replays')
    VIDEOS_FOLDER = os.path.join(RUN_FOLDER, 'videos')
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPLAYS_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)

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
                agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}'))
                with open(os.path.join(REPLAYS_FOLDER, f'experience_replay_{GAME_NAME}.pkl'), 'wb') as f:
                    pickle.dump(agent.memory, f)

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                print(f"Step: {total_steps}, Evaluation Score: {eval_score}")

            if done:
                break

        avg_q_value = np.mean(agent.q_values_episode)
        avg_q_values_per_episode.append(avg_q_value)
        scores.append(episode_reward)
        memory_info = psutil.virtual_memory()
        print(f"Episode: {episode}, Score: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Steps: {episode_steps}, Avg Q-value: {avg_q_value:.2f}, Exp replay: {len(agent.memory)}, Memory Usage: {memory_info.percent}%")
        gc.collect()
        torch.cuda.empty_cache()

        if episode % 20 == 0:
            plot_training_progress(scores, avg_q_values_per_episode, losses, GAME_NAME, timestamp, RUN_FOLDER)
            print(f"Total: {memory_info.total / (1024 ** 3):.2f} GB")
            print(f"Available: {memory_info.available / (1024 ** 3):.2f} GB")
            print(f"Used: {memory_info.used / (1024 ** 3):.2f} GB")
            print(f"Free: {memory_info.free / (1024 ** 3):.2f} GB")
            print(f"Percentage: {memory_info.percent}%")

    agent.save(os.path.join(MODELS_FOLDER, f'dqn_model_{GAME_NAME}_final_{timestamp}'))
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
