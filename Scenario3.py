import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

def train(fourRoomsObj, Q, alpha, gamma, epsilon_start, epsilon_decay, min_epsilon, episodes=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    rewards = []
    steps = []
    epsilons = []
    
    for episode in range(episodes):
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        total_reward = 0
        step_count = 0
        epsilon = max(min_epsilon, epsilon_start * (epsilon_decay ** episode))
        epsilons.append(epsilon)
        
        while not fourRoomsObj.isTerminal():
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state[0]-1, state[1]-1, k])
            
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            # Special reward function for ordered collection
            if gridType > 0:
                reward = -1 if isTerminal and packagesRemaining > 0 else 1
            else:
                reward = -0.01
            
            total_reward += reward
            step_count += 1
            next_state = newPos
            next_k = packagesRemaining
            max_next_Q = 0 if isTerminal else np.max(Q[next_state[0]-1, next_state[1]-1, next_k])
            Q[state[0]-1, state[1]-1, k, action] += alpha * (reward + gamma * max_next_Q - Q[state[0]-1, state[1]-1, k, action])
            state, k = next_state, next_k
        
        rewards.append(total_reward)
        steps.append(step_count)
    
    return rewards, steps, epsilons

def evaluate_policy(fourRoomsObj, Q, episodes=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    total_rewards = []
    total_steps = []
    
    for _ in range(episodes):
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        episode_reward = 0
        steps = 0
        
        while not fourRoomsObj.isTerminal():
            action = np.argmax(Q[state[0]-1, state[1]-1, k])
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            if gridType > 0:
                reward = -1 if isTerminal and packagesRemaining > 0 else 1
            else:
                reward = -0.01
            episode_reward += reward
            steps += 1
            state, k = newPos, packagesRemaining
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    return np.mean(total_rewards), np.mean(total_steps)

def visualize_policy(Q, save_path="policy_scenario3.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_symbols = ['↑', '↓', '←', '→']
    
    for k in range(4):
        policy = np.zeros((11, 11), dtype=object)
        value = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                best_action = np.argmax(Q[i, j, k])
                policy[i, j] = action_symbols[best_action]
                value[i, j] = np.max(Q[i, j, k])
                
        axes[k].imshow(value, cmap='viridis')
        for i in range(11):
            for j in range(11):
                axes[k].text(j, i, policy[i, j], ha='center', va='center', color='white')
        axes[k].set_title(f"Packages Remaining: {k}")
        axes[k].set_xticks(np.arange(11))
        axes[k].set_yticks(np.arange(11))
        axes[k].set_xticklabels(np.arange(1, 12))
        axes[k].set_yticklabels(np.arange(1, 12))
        axes[k].grid(False)
    
    plt.suptitle("Learned Policy for Scenario 3")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def show_final_path(fourRoomsObj, Q, save_path):
    """
    Shows and saves the final path taken by the agent using the learned policy.
    
    Args:
        fourRoomsObj: The environment object
        Q: The trained Q-table
        save_path: Path to save the visualization
    
    Returns:
        int: Number of steps taken
    """
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    steps = 0
    
    while not fourRoomsObj.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, k])
        _, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
        state, k = newPos, packagesRemaining
        steps += 1
    
    fourRoomsObj.showPath(-1, savefig=save_path)
    return steps

def moving_average(data, window_size):
    """
    Calculate the moving average of the data.
    
    Args:
        data: List of values
        window_size: Size of the moving window
    
    Returns:
        List of moving averages
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards, steps, epsilons, window_size=50, save_dir=""):
    """
    Plot learning curves for rewards, steps, and epsilon decay.
    
    Args:
        rewards: List of rewards per episode
        steps: List of steps per episode
        epsilons: List of epsilon values per episode
        window_size: Window size for moving average
        save_dir: Directory to save the plots
    """
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= window_size:
        rewards_smooth = moving_average(rewards, window_size)
        plt.plot(range(window_size-1, len(rewards)), rewards_smooth, 
                 color='blue', label=f'Moving Avg (window={window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}reward_learning_curve.png")
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps, alpha=0.3, color='green', label='Raw')
    if len(steps) >= window_size:
        steps_smooth = moving_average(steps, window_size)
        plt.plot(range(window_size-1, len(steps)), steps_smooth, 
                 color='green', label=f'Moving Avg (window={window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}steps_learning_curve.png")
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}epsilon_decay.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 3: Ordered Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-window_size', type=int, default=50, help='Window size for moving average')
    parser.add_argument('-show_progress', action='store_true', help='Show progress bar during training')
    parser.add_argument('-output_dir', type=str, default='', help='Directory to save output files')
    args = parser.parse_args()
    
    # Ensure output directory exists and ends with a slash for consistency
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not output_dir.endswith('/'):
            output_dir += '/'
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize FourRooms environment for Scenario 3
    fourRoomsObj = FourRooms('rgb', stochastic=args.stochastic)
    Q = np.zeros((11, 11, 4, 4))  # State space: x: 1-11, y: 1-11, k: 0-3, actions: 0-3
    
    alpha, gamma = args.alpha, args.gamma
    
    # Train the Q-learning agent
    rewards, steps, epsilons = train(fourRoomsObj, Q, alpha, gamma, 
                                    epsilon_start=1.0, epsilon_decay=0.995, 
                                    min_epsilon=0.01, episodes=args.episodes, seed=args.seed)
    
    # Generate visualizations for learning curves and policy
    plot_learning_curves(rewards, steps, epsilons, window_size=args.window_size, save_dir=output_dir)
    visualize_policy(Q, save_path=f"{output_dir}policy_scenario3.png")
    
    # Demonstrate the final learned path
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    while not fourRoomsObj.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, k])
        _, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
        state, k = newPos, packagesRemaining
    fourRoomsObj.showPath(-1, savefig=f"{output_dir}path_scenario3.png")
    
    # Evaluate the learned policy
    avg_reward, avg_steps = evaluate_policy(fourRoomsObj, Q, episodes=100, seed=args.seed)
    print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

if __name__ == "__main__":
    main()