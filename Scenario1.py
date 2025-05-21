import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random

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
            reward = 1 if gridType > 0 else -0.01
            
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

def evaluate_policy(fourRoomsObj, Q, num_episodes=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    total_rewards = []
    total_steps = []
    
    for _ in range(num_episodes):
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        episode_reward = 0
        steps = 0
        
        while not fourRoomsObj.isTerminal():
            action = np.argmax(Q[state[0]-1, state[1]-1, k])
            gridType, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
            reward = 1 if gridType > 0 else -0.01
            episode_reward += reward
            steps += 1
            state, k = newPos, packagesRemaining
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    return np.mean(total_rewards), np.mean(total_steps)

def visualize_policy(Q, title="Learned Policy", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    action_symbols = ['↑', '↓', '←', '→']
    
    for k in range(2):
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
        axes[k].set_title(f"Package Remaining: {k}")
        axes[k].set_xticks(np.arange(11))
        axes[k].set_yticks(np.arange(11))
        axes[k].set_xticklabels(np.arange(1, 12))
        axes[k].set_yticklabels(np.arange(1, 12))
        axes[k].grid(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def show_final_path(fourRoomsObj, Q, save_path=None):
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    steps = 0
    
    while not fourRoomsObj.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, k])
        _, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
        state, k = newPos, packagesRemaining
        steps += 1
    
    print(f"Path completed in {steps} steps")
    fourRoomsObj.showPath(-1, savefig=save_path)

def plot_learning_curves(rewards1, steps1, rewards2, steps2, save_dir=''):
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards1, label='High Exploration: ε=1.0, decay=0.995')
    plt.plot(rewards2, label='Moderate Exploration: ε=0.5, decay=0.99')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward Learning Curves')
    plt.legend()
    plt.savefig(f'{save_dir}reward_learning_curves.png')
    plt.close()
    
    # Plot steps per episode
    plt.figure(figsize=(10, 5))
    plt.plot(steps1, label='High Exploration: ε=1.0, decay=0.995')
    plt.plot(steps2, label='Moderate Exploration: ε=0.5, decay=0.99')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode Learning Curves')
    plt.legend()
    plt.savefig(f'{save_dir}steps_learning_curves.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 1: Simple Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-output_dir', type=str, default='', help='Directory to save output files')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Ensure output directory ends with a slash if provided
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'

    fourRoomsObj = FourRooms('simple', stochastic=args.stochastic)
    Q1 = np.zeros((11, 11, 2, 4))  # x: 1-11, y: 1-11, k: 0-1, actions: 0-3
    Q2 = np.zeros((11, 11, 2, 4))
    
    # Strategy 1: High exploration
    rewards1, steps1, epsilons1 = train(fourRoomsObj, Q1, args.alpha, args.gamma, 
                                        epsilon_start=1.0, epsilon_decay=0.995, 
                                        min_epsilon=0.01, episodes=args.episodes, 
                                        seed=args.seed)
    
    # Strategy 2: Moderate exploration
    rewards2, steps2, epsilons2 = train(fourRoomsObj, Q2, args.alpha, args.gamma, 
                                        epsilon_start=0.5, epsilon_decay=0.99, 
                                        min_epsilon=0.01, episodes=args.episodes, 
                                        seed=args.seed)
    
    # Evaluate both policies
    avg_reward1, avg_steps1 = evaluate_policy(fourRoomsObj, Q1, seed=args.seed)
    avg_reward2, avg_steps2 = evaluate_policy(fourRoomsObj, Q2, seed=args.seed)
    
    print(f"High Exploration - Avg Reward: {avg_reward1:.2f}, Avg Steps: {avg_steps1:.2f}")
    print(f"Moderate Exploration - Avg Reward: {avg_reward2:.2f}, Avg Steps: {avg_steps2:.2f}")
    
    # Visualize policies
    visualize_policy(Q1, "High Exploration Policy", f'{output_dir}policy_high_exploration.png')
    visualize_policy(Q2, "Moderate Exploration Policy", f'{output_dir}policy_moderate_exploration.png')
    
    # Plot learning curves
    plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, output_dir)
    
    # Show final paths
    print("Generating path visualization for high exploration strategy...")
    show_final_path(fourRoomsObj, Q1, f'{output_dir}path_high_exploration.png')
    print("Generating path visualization for moderate exploration strategy...")
    show_final_path(fourRoomsObj, Q2, f'{output_dir}path_moderate_exploration.png')

if __name__ == "__main__":
    main()