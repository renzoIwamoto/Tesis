import gymnasium as gym
import matplotlib.pyplot as plt

# Crear los entornos
env_no_skip = gym.make('BreakoutNoFrameskip-v4', render_mode="rgb_array")
env_skip = gym.make('BreakoutDeterministic-v4', render_mode="rgb_array")

def check_frame_skip(env, num_steps=4):
    state, _ = env.reset()
    plt.figure(figsize=(10, 5))
    
    for i in range(num_steps):
        # Seleccionar acción aleatoria
        action = env.action_space.sample()
        
        # Ejecutar la acción en el entorno
        next_state, reward, done, truncated, info = env.step(action)
        
        # Imprimir el shape del estado
        print(f"Step {i+1}, State shape: {next_state.shape}")
        
        # Mostrar la imagen del estado
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(next_state)
        plt.title(f'Step {i+1}')
        plt.axis('off')
        
        if done:
            break
    
    plt.show()

print("BreakoutNoFrameskip-v4 (Sin Frame Skip)")
check_frame_skip(env_no_skip, num_steps=4)

print("BreakoutDeterministic-v4 (Con Frame Skip)")
check_frame_skip(env_skip, num_steps=4)

env_no_skip.close()
env_skip.close()
