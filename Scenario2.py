import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

class MultiPackageAgent:
    """Q-learning agent for multiple package collection scenario."""
    
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, 
                 epsilon_decay=0.995, min_epsilon=0.01, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.Q = np.zeros((11, 11, 4, 4))  # State space: (x, y, packages_remaining, action) - FIXED: was 5, should be 4
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_epsilon(self, episode):
        return max(self.min_epsilon, self.epsilon_start * (self.epsilon_decay ** episode))

    def choose_action(self, state, packages_remaining, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        return np.argmax(self.Q[state[0]-1, state[1]-1, packages_remaining])

    def train(self, episodes=1000, show_progress=True):
        rewards = []
        steps = []
        epsilons = []
        
        iterator = tqdm(range(episodes), desc="Training Scenario 2", disable=not show_progress)
        
        for episode in iterator:
            self.env.newEpoch()
            state = self.env.getPosition()
            k = self.env.getPackagesRemaining()
            total_reward = 0
            step_count = 0
            epsilon = self.get_epsilon(episode)
            epsilons.append(epsilon)
            
            # Add step limit to prevent infinite loops
            max_steps = 500
            
            while not self.env.isTerminal() and step_count < max_steps:
                action = self.choose_action(state, k, epsilon)
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                reward = 10 if grid_type > 0 else -0.01  # Increased reward for packages
                
                # Q-learning update
                current_q = self.Q[state[0]-1, state[1]-1, k, action]
                next_max_q = 0 if is_terminal else np.max(self.Q[new_pos[0]-1, new_pos[1]-1, new_packages])
                self.Q[state[0]-1, state[1]-1, k, action] += self.alpha * (
                    reward + self.gamma * next_max_q - current_q
                )
                
                total_reward += reward
                step_count += 1
                state, k = new_pos, new_packages
            
            rewards.append(total_reward)
            steps.append(step_count)
            
            if show_progress:
                iterator.set_postfix({
                    'ε': f"{epsilon:.3f}",
                    'Reward': total_reward,
                    'Steps': step_count,
                    'Avg Reward': f"{np.mean(rewards[-100:]):.2f}"
                })
        
        return rewards, steps, epsilons

    def evaluate(self, episodes=50):  # Reduced for safety
        total_rewards = []
        total_steps = []
        
        for _ in range(episodes):
            self.env.newEpoch()
            state = self.env.getPosition()
            k = self.env.getPackagesRemaining()
            episode_reward = 0
            steps = 0
            max_steps = 200  # Safety limit
            
            while not self.env.isTerminal() and steps < max_steps:
                action = np.argmax(self.Q[state[0]-1, state[1]-1, k])
                grid_type, new_pos, new_packages, _ = self.env.takeAction(action)
                reward = 10 if grid_type > 0 else -0.01
                episode_reward += reward
                steps += 1
                state, k = new_pos, new_packages
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        return np.mean(total_rewards), np.mean(total_steps)

def visualize_policy(agent, save_path="policy_scenario2.png"):
    Q = agent.Q
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_symbols = ['↑', '↓', '←', '→']
    
    for k in range(4):
        policy_grid = np.zeros((11, 11), dtype=object)
        value_grid = np.zeros((11, 11))
        
        for i in range(11):
            for j in range(11):
                best_action = np.argmax(Q[i, j, k])
                policy_grid[i, j] = action_symbols[best_action]
                value_grid[i, j] = np.max(Q[i, j, k])
        
        im = axes[k].imshow(value_grid, cmap='viridis')
        axes[k].set_title(f"Packages Remaining: {k}")
        
        for i in range(11):
            for j in range(11):
                axes[k].text(j, i, policy_grid[i, j], 
                          ha='center', va='center', 
                          color='white', fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
    
    plt.suptitle("Learned Policy for Scenario 2: Multiple Package Collection")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards, steps, epsilons, window_size=50, save_dir=''):
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) > window_size:
        smooth = moving_average(rewards, window_size)
        plt.plot(range(window_size-1, len(rewards)), smooth, label='Smoothed', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Learning Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}reward_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    plt.plot(steps, alpha=0.3, label='Raw')
    if len(steps) > window_size:
        smooth = moving_average(steps, window_size)
        plt.plot(range(window_size-1, len(steps)), smooth, label='Smoothed', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Efficiency Learning Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}steps_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}epsilon_decay.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 2: Multiple Packages')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic transitions')
    parser.add_argument('-episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('-show_progress', action='store_true', help='Show training progress')
    parser.add_argument('-window_size', type=int, default=50, help='Smoothing window size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = FourRooms('multi', stochastic=args.stochastic)

    print("\n=== Initializing Multi-Package Agent ===")
    agent = MultiPackageAgent(
        env,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        seed=args.seed
    )

    print("\n=== Training Started ===")
    rewards, steps, epsilons = agent.train(args.episodes, args.show_progress)

    print("\n=== Training Completed ===")
    print(f"Final Epsilon: {epsilons[-1]:.4f}")
    print(f"Average Reward (Last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Average Steps (Last 100): {np.mean(steps[-100:]):.2f}")

    print("\n=== Generating Visualizations ===")
    try:
        plot_learning_curves(rewards, steps, epsilons, 
                            window_size=args.window_size, 
                            save_dir=args.output_dir)
        visualize_policy(agent, f"{args.output_dir}policy_scenario2.png")
    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n=== Evaluating Policy ===")
    try:
        avg_reward, avg_steps = agent.evaluate()
        print(f"Evaluation Results (50 episodes):")
        print(f"  - Average Reward: {avg_reward:.2f}")
        print(f"  - Average Steps: {avg_steps:.2f}")
    except Exception as e:
        print(f"Evaluation error: {e}")

    print("\n=== Generating Final Path ===")
    try:
        env.newEpoch()
        state = env.getPosition()
        k = env.getPackagesRemaining()
        steps = 0
        max_steps = 200
        
        while not env.isTerminal() and steps < max_steps:
            action = np.argmax(agent.Q[state[0]-1, state[1]-1, k])
            _, new_pos, new_packages, _ = env.takeAction(action)
            state, k = new_pos, new_packages
            steps += 1
        
        env.showPath(-1, savefig=f"{args.output_dir}path_scenario2.png")
        print(f"Final path completed in {steps} steps")
    except Exception as e:
        print(f"Path generation error: {e}")

    print("\n=== Scenario 2 Completed Successfully ===")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()