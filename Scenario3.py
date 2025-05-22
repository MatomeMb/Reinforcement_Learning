import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

class OrderedPackageAgent:
    """Q-learning agent for ordered package collection scenario (R→G→B)."""
    
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, 
                 epsilon_decay=0.998, min_epsilon=0.05, seed=None):
        """
        Initialize Q-learning agent for ordered collection.
        
        Args:
            env: FourRooms environment object
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon_start: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum exploration rate
            seed: Random seed for reproducibility
        """
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
        """Calculate exponentially decaying exploration rate."""
        return max(self.min_epsilon, self.epsilon_start * (self.epsilon_decay ** episode))

    def choose_action(self, state, package_state, epsilon):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(4)  # Random action
        return np.argmax(self.Q[state[0]-1, state[1]-1, package_state])

    def train(self, episodes=2000, show_progress=True):
        """Train the agent with ordered collection constraints."""
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
            
            while not self.env.isTerminal():
                action = self.choose_action(state, k, epsilon)
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                
                # Custom reward structure for ordered collection
                reward = 10 if grid_type > 0 and not (is_terminal and new_packages > 0) else -10 if is_terminal else -0.01
                
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
                    'ε': f"{epsilon:.3f}",
                    'Reward': total_reward,
                    'Steps': step_count,
                    'Success': f"{(successes/(episode+1))*100:.1f}%"
                })
        
        print(f"Training completed with {successes}/{episodes} successful episodes ({successes/episodes*100:.1f}%)")
        return rewards, steps, epsilons

    def evaluate(self, episodes=100):
        """Evaluate the learned policy's performance."""
        total_rewards = []
        total_steps = []
        successes = 0
        
        for _ in range(episodes):
            self.env.newEpoch()
            state = self.env.getPosition()
            k = self.env.getPackagesRemaining()
            episode_reward = 0
            steps = 0
            
            while not self.env.isTerminal():
                action = np.argmax(self.Q[state[0]-1, state[1]-1, k])
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                reward = 10 if grid_type > 0 and not (is_terminal and new_packages > 0) else -10 if is_terminal else -0.01
                episode_reward += reward
                steps += 1
                state, k = new_pos, new_packages
            
            if k == 0:
                successes += 1
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        return np.mean(total_rewards), np.mean(total_steps), successes/episodes

def visualize_policy(agent, save_path="policy_scenario3.png"):
    """Visualize the learned policy with action directions."""
    Q = agent.Q
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_symbols = ['↑', '↓', '←', '→']
    states = ['All Collected', 'Blue Remaining', 'Green+Blue Remaining', 'All Remaining']
    
    for k in range(4):
        policy = np.zeros((11, 11), dtype=object)
        values = np.zeros((11, 11))
        
        for i in range(11):
            for j in range(11):
                best_action = np.argmax(Q[i, j, k])
                policy[i, j] = action_symbols[best_action]
                values[i, j] = np.max(Q[i, j, k])
        
        im = axes[k].imshow(values, cmap='viridis')
        axes[k].set_title(f"{states[k]} (k={k})")
        
        # Add action markers
        for i in range(11):
            for j in range(11):
                axes[k].text(j, i, policy[i, j], 
                           ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=8)
        
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
    
    plt.suptitle("Learned Policy for Ordered Package Collection (R→G→B)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def moving_average(data, window_size=50):
    """Calculate smoothed moving average of data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards, steps, epsilons, window_size=50, save_dir=""):
    """Generate training progress visualizations."""
    # Reward curve
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
    
    # Steps curve
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
    
    # Epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}epsilon_decay.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 3: Ordered Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-episodes', type=int, default=2000, help='Training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('-show_progress', action='store_true', help='Show training progress')
    parser.add_argument('-window_size', type=int, default=50, help='Smoothing window size')
    args = parser.parse_args()

    # Initialize environment and agent
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = FourRooms('rgb', stochastic=args.stochastic)

    print("\n=== Initializing Ordered Package Agent ===")
    agent = OrderedPackageAgent(
        env,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_decay=0.998,
        min_epsilon=0.05,
        seed=args.seed
    )

    # Training phase
    print("\n=== Training Started ===")
    rewards, steps, epsilons = agent.train(args.episodes, args.show_progress)

    # Post-training analysis
    print("\n=== Training Results ===")
    print(f"Final Exploration Rate: {epsilons[-1]:.4f}")
    print(f"Average Reward (Last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Average Steps (Last 100): {np.mean(steps[-100:]):.2f}")

    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    plot_learning_curves(rewards, steps, epsilons, 
                        window_size=args.window_size, 
                        save_dir=args.output_dir)
    visualize_policy(agent, f"{args.output_dir}policy_scenario3.png")

    # Policy evaluation
    print("\n=== Policy Evaluation ===")
    avg_reward, avg_steps, success_rate = agent.evaluate()
    print(f"Evaluation Results (100 episodes):")
    print(f"  - Average Reward: {avg_reward:.2f}")
    print(f"  - Average Steps: {avg_steps:.2f}")
    print(f"  - Success Rate: {success_rate*100:.1f}%")

    # Final path demonstration
    print("\n=== Final Path Demonstration ===")
    env.newEpoch()
    state = env.getPosition()
    k = env.getPackagesRemaining()
    path_steps = 0
    
    while not env.isTerminal():
        action = np.argmax(agent.Q[state[0]-1, state[1]-1, k])
        _, new_pos, new_packages, _ = env.takeAction(action)
        state, k = new_pos, new_packages
        path_steps += 1
    
    env.showPath(-1, savefig=f"{args.output_dir}path_scenario3.png")
    status = "SUCCESS" if k == 0 else f"FAILED ({k} remaining)"
    print(f"{status}: Path completed in {path_steps} steps")

    print("\n=== Scenario 3 Complete ===")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()