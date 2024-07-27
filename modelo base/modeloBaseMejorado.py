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
import sum_tree

# Configuración del entorno y parámetros
ENV_NAME = 'BreakoutDeterministic-v4'
GAME_NAME = ENV_NAME.split('-')[0]
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 1000000
BATCH_SIZE = 32
TRAINING_START = 50000
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORATION_STEPS = 1000000
UPDATE_TARGET_FREQUENCY = 10000
SAVE_FREQUENCY = 100000
EVALUATION_FREQUENCY = 50000
NUM_EVALUATION_EPISODES = 10
EPISODES = 10000

# Parámetros para PER
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 0.01

# Configuración de GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Crear la carpeta principal del juego
GAME_FOLDER = f'{GAME_NAME}_results'
os.makedirs(GAME_FOLDER, exist_ok=True)

class DuelingDQN(keras.Model):
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.value = keras.layers.Dense(1)
        self.advantage = keras.layers.Dense(action_size)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = sum_tree.SumTree(MEMORY_SIZE)
        self.epsilon = INITIAL_EPSILON

        self.model = DuelingDQN(action_size)
        self.target_model = DuelingDQN(action_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.update_target_model()

        self.loss_history = []
        self.q_values_history = []

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.memory.add(max_priority, (state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(state)
        self.q_values_history.append(tf.reduce_mean(q_values).numpy())
        return tf.argmax(q_values[0]).numpy()

    def replay(self, beta):
        mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = np.array(states).reshape(BATCH_SIZE, *self.state_shape)
        next_states = np.array(next_states).reshape(BATCH_SIZE, *self.state_shape)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.model(next_states)
            next_target_q_values = self.target_model(next_states)

            max_actions = tf.argmax(next_q_values, axis=1)
            next_q_values = tf.gather(next_target_q_values, max_actions, batch_dims=1)

            targets = q_values.numpy()
            for i in range(BATCH_SIZE):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + GAMMA * next_q_values[i]

            loss = tf.reduce_mean(is_weights * tf.square(targets - q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss_history.append(loss.numpy())

        # Update priorities
        td_errors = tf.abs(targets - q_values).numpy()
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i])

    def update_epsilon(self, step):
        self.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (step * (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS))

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)
        self.update_target_model()

@tf.function
def preprocess_frame(frame):
    gray = tf.image.rgb_to_grayscale(frame)
    resized = tf.image.resize(gray, [84, 84])
    normalized = resized / 255.0
    return normalized[:,:,0]

def stack_frames(stacked_frames, frame, is_new_episode):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=-1)
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
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def plot_training_progress(scores, avg_q_values, losses, game_name):
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
    plt.savefig(os.path.join(GAME_FOLDER, f'training_progress_{game_name}.png'))
    plt.close()

def main():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    state_shape = (84, 84, FRAME_STACK)
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)
    stacked_frames = deque(maxlen=FRAME_STACK)
    
    scores = []
    total_steps = 0

    models_folder = os.path.join(GAME_FOLDER, 'models')
    replays_folder = os.path.join(GAME_FOLDER, 'replays')
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(replays_folder, exist_ok=True)

    beta = PER_BETA

    for episode in range(EPISODES):
        state, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0

        for time_step in range(10000):
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps > TRAINING_START:
                agent.replay(beta)
                agent.update_epsilon(total_steps)
                beta = min(1.0, beta + PER_BETA_INCREMENT)

            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model()

            if total_steps % SAVE_FREQUENCY == 0:
                agent.save(os.path.join(models_folder, f'dqn_model_{GAME_NAME}_step_{total_steps}.h5'))
                with open(os.path.join(replays_folder, f'experience_replay_{GAME_NAME}_step_{total_steps}.pkl'), 'wb') as f:
                    pickle.dump(agent.memory, f)

            if total_steps % EVALUATION_FREQUENCY == 0:
                eval_score = evaluate_agent(env, agent, NUM_EVALUATION_EPISODES)
                print(f"Step: {total_steps}, Evaluation Score: {eval_score}")

            if done:
                break

        scores.append(episode_reward)
        print(f"Episode: {episode}, Score: {episode_reward}, Epsilon: {agent.epsilon:.2f}")

        if episode % 10 == 0:
            plot_training_progress(scores, agent.q_values_history, agent.loss_history, GAME_NAME)

    # Guardar el modelo final y el experience replay
    agent.save(os.path.join(models_folder, f'dqn_model_{GAME_NAME}_final.h5'))
    with open(os.path.join(replays_folder, f'experience_replay_{GAME_NAME}_final.pkl'), 'wb') as f:
        pickle.dump(agent.memory, f)

    # Grabar video del agente entrenado
    video_folder = os.path.join(GAME_FOLDER, 'videos')
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, video_folder)
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