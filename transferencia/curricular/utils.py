from collections import deque
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_frame(frame):
    gray = (0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]).astype(np.uint8)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, frame, is_new_episode, FRAME_STACK):
    frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([frame] * FRAME_STACK, maxlen=FRAME_STACK)
    else:
        stacked_frames.append(frame)
    stacked = np.stack(stacked_frames, axis=0)
    return stacked, stacked_frames

def evaluate_agent(env, agent, num_episodes, FRAME_STACK):
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.00  # Establecer epsilon a 0 para la evaluación
    
    for _ in range(num_episodes):
        state, _ = env.reset(seed=np.random.randint(0, 100000))
        stacked_frames = deque(maxlen=FRAME_STACK)
        state, stacked_frames = stack_frames(stacked_frames, state, True, FRAME_STACK)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, FRAME_STACK)
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
        return data  # Devuelve los datos sin suavizar si son insuficientes
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_progress(scores, avg_q_values, losses, game_name, timestamp, LOCAL_FOLDER):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    window_size = min(20, len(scores))  # Usamos los últimos 20 episodios o menos si hay menos datos
    
    # Calcular promedios móviles
    smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    smoothed_avg_q_values = np.convolve(avg_q_values, np.ones(window_size)/window_size, mode='valid')
    smoothed_losses = smooth_data(losses)  # Mantenemos el suavizado original para las pérdidas

    # Calcular valores mínimos y máximos para el área sombreada
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

    # Agregar valores máximo y mínimo al gráfico de puntuaciones
    ax1.axhline(max(scores), color='red', linestyle='--', label='Max Score')
    ax1.axhline(min(scores), color='green', linestyle='--', label='Min Score')

    # Gráfico de valores Q promedio
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

    # Gráfico de pérdidas
    ax3.plot(range(len(smoothed_losses)), smoothed_losses, label='Smoothed Losses', color='red')
    ax3.set_title(f'{game_name} - Loss')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_FOLDER, f'training_progress_{game_name}_{timestamp}.png'))
    plt.close()

def plot_evaluation_scores(evaluation_scores, game_name, timestamp, LOCAL_FOLDER):
    steps, scores = zip(*evaluation_scores)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, marker='o')
    plt.title(f'{game_name} - Evaluation Scores')
    plt.xlabel('Total Steps')
    plt.ylabel('Evaluation Score')
    plt.grid(True)
    
    plt.savefig(os.path.join(LOCAL_FOLDER, f'evaluation_scores_{game_name}_{timestamp}.png'))
    plt.close()