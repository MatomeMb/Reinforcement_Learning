import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

class OrderedPackageAgent:
    """Q-learning agent for ordered package collection scenario (R->G->B)."""
    
    def __init__(self, env, alpha=0.2, gamma=0.95, epsilon_start=1.0, 
                 epsilon_decay=0.9995, min_epsilon=0.05, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.Q = np.zeros((11, 11, 4, 4))  # (x, y, package_state, action)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_epsilon(self, episode):
        return max(self.min_epsilon, self.epsilon_start * (self.epsilon_decay ** episode))

    def choose_action(self, state, package_state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        return np.argmax(self.Q[state[0]-1, state[1]-1, package_state])

    def train(self, episodes=1000, show_progress=True):
        rewards = []
        steps = []
        epsilons = []
        successes = 0
        
        iterator = tqdm(range(episodes), desc="Training Scenario 3", disable=not show_progress)
        
        for episode in iterator:
            self.env.newEpoch()
            state = self.env.getPosition()
            k = self.env.getPackagesRemaining()
            total_reward = 0
            step_count = 0
            epsilon = self.get_epsilon(episode)
            epsilons.append(epsilon)
            
            max_steps = 500  # Increased step limit
            
            while not self.env.isTerminal() and step_count < max_steps:
                action = self.choose_action(state, k, epsilon)
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                
                # IMPROVED reward structure for ordered collection
                reward = -0.01  # Small step penalty
                
                if grid_type > 0:  # Found a package
                    if new_packages < k:  # Successfully collected
                        if k == 3 and grid_type == 1:      # Red first
                            reward = 100
                        elif k == 2 and grid_type == 2:    # Green second  
                            reward = 200
                        elif k == 1 and grid_type == 3:    # Blue third
                            reward = 300
                        else:  # Wrong order
                            reward = -500
                    else:  # Tried wrong package
                        reward = -100
                
                # Completion bonus
                if is_terminal and new_packages == 0:
                    reward += 500
                elif is_terminal and new_packages > 0:
                    reward = -300  # Failed to complete
                
                # Q-learning update
                current_q = self.Q[state[0]-1, state[1]-1, k, action]
                next_max_q = 0 if is_terminal else np.max(self.Q[new_pos[0]-1, new_pos[1]-1, new_packages])
                self.Q[state[0]-1, state[1]-1, k, action] += self.alpha * (
                    reward + self.gamma * next_max_q - current_q
                )
                
                total_reward += reward
                step_count += 1
                state, k = new_pos, new_packages
            
            if k == 0:
                successes += 1
            rewards.append(total_reward)
            steps.append(step_count)
            
            if show_progress:
                iterator.set_postfix({
                    'epsilon': f"{epsilon:.3f}",
                    'Reward': f"{total_reward:.1f}",
                    'Steps': step_count,
                    'Success': f"{(successes/(episode+1))*100:.1f}%"
                })
        
        print(f"Training completed with {successes}/{episodes} successful episodes ({successes/episodes*100:.1f}%)")
        return rewards, steps, epsilons

    def evaluate(self, episodes=20):
        total_rewards = []
        total_steps = []
        successes = 0
        
        for _ in range(episodes):
            self.env.newEpoch()
            state = self.env.getPosition()
            k = self.env.getPackagesRemaining()
            episode_reward = 0
            steps = 0
            max_steps = 200
            
            while not self.env.isTerminal() and steps < max_steps:
                action = np.argmax(self.Q[state[0]-1, state[1]-1, k])
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                
                # Same reward structure as training
                reward = -0.01
                if grid_type > 0:
                    if new_packages < k:
                        if k == 3 and grid_type == 1:
                            reward = 100
                        elif k == 2 and grid_type == 2:
                            reward = 200
                        elif k == 1 and grid_type == 3:
                            reward = 300
                        else:
                            reward = -500
                    else:
                        reward = -100
                        
                if is_terminal and new_packages == 0:
                    reward += 500
                elif is_terminal and new_packages > 0:
                    reward = -300
                    
                episode_reward += reward
                steps += 1
                state, k = new_pos, new_packages
            
            if k == 0:
                successes += 1
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        return np.mean(total_rewards), np.mean(total_steps), successes/episodes

def visualize_policy(agent, save_path="policy_scenario3.png"):
    Q = agent.Q
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_symbols = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    states = ['All Collected', 'Blue Remaining', 'Green+Blue Remaining', 'All Remaining (Start)']
    
    for k in range(4):
        policy = np.zeros((11, 11), dtype=object)
        values = np.zeros((11, 11))
        
        for i in range(11):
            for j in range(11):
                best_action = np.argmax(Q[i, j, k])
                policy[i, j] = action_symbols[best_action]
                values[i, j] = np.max(Q[i, j, k])
        
        im = axes[k].imshow(values, cmap='viridis')
        axes[k].set_title(f"{states[k]}")
        
        for i in range(11):
            for j in range(11):
                axes[k].text(j, i, policy[i, j], 
                           ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=6)
        
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
    
    plt.suptitle("Learned Policy for Ordered Package Collection (R->G->B)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards, steps, epsilons, window_size=50, save_dir=""):
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw', color='blue')
    if len(rewards) > window_size:
        smooth = moving_average(rewards, window_size)
        plt.plot(range(window_size-1, len(rewards)), smooth, 
                label='Smoothed', linewidth=2, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Learning Curve - Scenario 3 (Ordered Collection)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}reward_curve_scenario3.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    plt.plot(steps, alpha=0.3, label='Raw', color='blue')
    if len(steps) > window_size:
        smooth = moving_average(steps, window_size)
        plt.plot(range(window_size-1, len(steps)), smooth, 
                label='Smoothed', linewidth=2, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Efficiency Learning Curve - Scenario 3 (Ordered Collection)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}steps_curve_scenario3.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, color='purple')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay - Scenario 3')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}epsilon_decay_scenario3.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 3: Ordered Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-episodes', type=int, default=1200, help='Training episodes')  # Increased
    parser.add_argument('-alpha', type=float, default=0.2, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('-show_progress', action='store_true', help='Show training progress')
    parser.add_argument('-window_size', type=int, default=50, help='Smoothing window size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = FourRooms('rgb', stochastic=args.stochastic)

    print("\n=== Initializing Ordered Package Agent ===")
    print("Task: Collect packages in order Red -> Green -> Blue")
    agent = OrderedPackageAgent(
        env,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_decay=0.9995,  # Slower decay for complex task
        min_epsilon=0.05,
        seed=args.seed
    )

    print("\n=== Training Started ===")
    rewards, steps, epsilons = agent.train(args.episodes, args.show_progress)

    print("\n=== Training Results ===")
    print(f"Final Exploration Rate: {epsilons[-1]:.4f}")
    print(f"Average Reward (Last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Average Steps (Last 100): {np.mean(steps[-100:]):.2f}")

    print("\n=== Generating Visualizations ===")
    try:
        plot_learning_curves(rewards, steps, epsilons, 
                            window_size=args.window_size, 
                            save_dir=args.output_dir)
        visualize_policy(agent, f"{args.output_dir}policy_scenario3.png")
        print("Visualizations saved successfully")
    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n=== Policy Evaluation ===")
    try:
        avg_reward, avg_steps, success_rate = agent.evaluate(episodes=30)
        print(f"Evaluation Results (30 episodes):")
        print(f"  - Average Reward: {avg_reward:.2f}")
        print(f"  - Average Steps: {avg_steps:.2f}")
        print(f"  - Success Rate: {success_rate*100:.1f}%")
    except Exception as e:
        print(f"Evaluation error: {e}")

    print("\n=== Final Path Demonstration ===")
    try:
        env.newEpoch()
        state = env.getPosition()
        k = env.getPackagesRemaining()
        path_steps = 0
        max_steps = 200
        
        print(f"Starting position: {state}, Packages remaining: {k}")
        
        while not env.isTerminal() and path_steps < max_steps:
            action = np.argmax(agent.Q[state[0]-1, state[1]-1, k])
            grid_type, new_pos, new_packages, _ = env.takeAction(action)
            
            if grid_type > 0:
                colors = {1: 'RED', 2: 'GREEN', 3: 'BLUE'}
                print(f"Step {path_steps}: Found {colors.get(grid_type, 'UNKNOWN')} package at {new_pos}")
            
            state, k = new_pos, new_packages
            path_steps += 1
        
        env.showPath(-1, savefig=f"{args.output_dir}path_scenario3.png")
        status = "SUCCESS" if k == 0 else f"FAILED ({k} packages remaining)"
        print(f"\n{status}: Path completed in {path_steps} steps")
        
    except Exception as e:
        print(f"Path demonstration error: {e}")

    print("\n=== Scenario 3 Complete ===")
    print(f"Output files saved to: {args.output_dir}")
    print("Key files generated:")
    print("  - reward_curve_scenario3.png")
    print("  - steps_curve_scenario3.png")
    print("  - epsilon_decay_scenario3.png")
    print("  - policy_scenario3.png")
    print("  - path_scenario3.png")

if __name__ == "__main__":
    main()