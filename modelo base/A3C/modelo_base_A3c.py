import numpy as np
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import threading
import multiprocessing
from collections import deque
import matplotlib.pyplot as plt
import os
import datetime
from gymnasium.wrappers import RecordVideo

# Configuración del entorno y parámetros
ENV_NAME = 'MsPacmanDeterministic-v4'
GAME_NAME = ENV_NAME.split('-')[0]
FRAME_STACK = 4
GAMMA = 0.99
LEARNING_RATE = 0.0001
NUM_WORKERS = multiprocessing.cpu_count()
UPDATE_GLOBAL_ITER = 20
GLOBAL_MAX_EPISODES = 5000
EVALUATION_FREQUENCY = 100
NUM_EVALUATION_EPISODES = 10

# Configuración de carpetas
GAME_FOLDER = f'{GAME_NAME}_A3C_results'
MODELS_FOLDER = os.path.join(GAME_FOLDER, 'models')
VIDEOS_FOLDER = os.path.join(GAME_FOLDER, 'videos')
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)

# Configuración de GPU
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

print(physical_devices)
print("Entorno: " + ENV_NAME)

class ActorCriticModel(keras.Model):
    def __init__(self, state_shape, action_size):
        super(ActorCriticModel, self).__init__()
        self.shared = keras.Sequential([
            keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_shape),
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu')
        ])
        self.actor = keras.layers.Dense(action_size, activation='softmax')
        self.critic = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.shared(inputs)
        return self.actor(x), self.critic(x)

class A3CAgent(threading.Thread):
    def __init__(self, global_model, opt, result_queue, id):
        super(A3CAgent, self).__init__()
        self.id = id
        self.env = gym.make(ENV_NAME)
        self.state_shape = (84, 84, FRAME_STACK)
        self.action_size = self.env.action_space.n
        self.model = ActorCriticModel(self.state_shape, self.action_size)
        self.global_model = global_model
        self.opt = opt
        self.result_queue = result_queue
        self.global_max_episodes = GLOBAL_MAX_EPISODES

    def run(self):
        total_step = 1
        while self.global_max_episodes > 0:
            state, _ = self.env.reset()
            stacked_frames = deque(maxlen=FRAME_STACK)
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            done = False
            episode_reward = 0
            self.ep_loss = 0

            while not done:
                logits, _ = self.model(np.expand_dims(state, axis=0))
                action = np.random.choice(self.action_size, p=logits.numpy()[0])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                
                if done:
                    reward = -1
                episode_reward += reward
                self.train_model(state, action, reward, next_state, done)
                state = next_state
                total_step += 1

                if total_step % UPDATE_GLOBAL_ITER == 0:
                    self.update_global()

            self.result_queue.put(episode_reward)
            self.global_max_episodes -= 1

    def train_model(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            probs, value = self.model(np.expand_dims(state, axis=0))
            _, next_value = self.model(np.expand_dims(next_state, axis=0))

            target = reward + GAMMA * next_value * (1 - int(done))
            advantage = target - value

            action_onehot = tf.one_hot(action, self.action_size)
            log_prob = tf.math.log(tf.reduce_sum(action_onehot * probs, axis=1))
            actor_loss = -log_prob * tf.stop_gradient(advantage)
            critic_loss = tf.square(advantage)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))
        self.ep_loss += total_loss

    def update_global(self):
        self.model.set_weights(self.global_model.get_weights())

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

def evaluate_agent(env, model, num_episodes):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        episode_reward = 0
        while not done:
            logits, _ = model(np.expand_dims(state, axis=0))
            action = np.argmax(logits.numpy()[0])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def plot_training_progress(scores, game_name, timestamp):
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title(f'{game_name} - A3C Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(GAME_FOLDER, f'A3C_{game_name}_progress_{timestamp}.png'))
    plt.close()

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    timestamp = get_timestamp()
    global_model = ActorCriticModel((84, 84, FRAME_STACK), gym.make(ENV_NAME).action_space.n)
    global_model(tf.random.normal((1, 84, 84, FRAME_STACK)))
    global_model.summary()

    opt = tf.optimizers.Adam(LEARNING_RATE)
    result_queue = multiprocessing.Queue()

    workers = [A3CAgent(global_model, opt, result_queue, i) for i in range(NUM_WORKERS)]
    for worker in workers:
        worker.start()

    moving_average_rewards = []
    scores = []
    evaluation_scores = []

    while len(scores) < GLOBAL_MAX_EPISODES:
        episode_reward = result_queue.get()
        if episode_reward is not None:
            scores.append(episode_reward)
            moving_average_rewards.append(np.mean(scores[-100:]))
            print(f"Episode {len(scores)}: Reward = {episode_reward}, Moving Average = {moving_average_rewards[-1]}")

            if len(scores) % EVALUATION_FREQUENCY == 0:
                eval_score = evaluate_agent(gym.make(ENV_NAME), global_model, NUM_EVALUATION_EPISODES)
                evaluation_scores.append(eval_score)
                print(f"Evaluation at episode {len(scores)}: Average Score = {eval_score}")

                # Save model
                global_model.save_weights(os.path.join(MODELS_FOLDER, f'A3C_{GAME_NAME}_model_episode_{len(scores)}_{timestamp}.h5'))

            if len(scores) % 100 == 0:
                plot_training_progress(scores, GAME_NAME, timestamp)

    [worker.join() for worker in workers]

    # Final plots
    plot_training_progress(scores, GAME_NAME, timestamp)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(scores), EVALUATION_FREQUENCY), evaluation_scores)
    plt.title(f'{GAME_NAME} - A3C Evaluation Scores')
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Score')
    plt.savefig(os.path.join(GAME_FOLDER, f'A3C_{GAME_NAME}_evaluation_{timestamp}.png'))
    plt.close()

    # Save final model
    global_model.save_weights(os.path.join(MODELS_FOLDER, f'A3C_{GAME_NAME}_model_final_{timestamp}.h5'))

    # Record video of trained agent
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join(VIDEOS_FOLDER, f'A3C_{GAME_NAME}_video_{timestamp}'))
    state, _ = env.reset()
    stacked_frames = deque(maxlen=FRAME_STACK)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    while not done:
        logits, _ = global_model(np.expand_dims(state, axis=0))
        action = np.argmax(logits.numpy()[0])
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state
    env.close()

if __name__ == "__main__":
    main()