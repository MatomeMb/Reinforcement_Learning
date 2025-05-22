import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

class QLearningAgent:
    """Q-learning agent for the FourRooms environment."""
    
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, 
                 epsilon_decay=0.995, min_epsilon=0.01, seed=None):
        """
        Initialize Q-learning agent.
        
        Args:
            env: FourRooms environment object
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon_start: Initial exploration rate (0-1)
            epsilon_decay: Epsilon decay rate per episode (0-1)
            min_epsilon: Minimum exploration rate (0-1)
            seed: Random seed for reproducibility
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.seed = seed
        self.Q = np.zeros((11, 11, 2, 4))  # State-action value table
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_epsilon(self, episode):
        """Calculate exponential decay epsilon."""
        return max(self.min_epsilon, self.epsilon_start * (self.epsilon_decay ** episode))

    def choose_action(self, state, packages_remaining, epsilon):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(4)  # Random action
        return np.argmax(self.Q[state[0]-1, state[1]-1, packages_remaining])

    def train(self, episodes=1000, show_progress=True):
        """
        Train the agent through Q-learning.
        
        Returns:
            tuple: (episode_rewards, episode_steps, epsilons) for analysis
        """
        rewards = []
        steps = []
        epsilons = []
        
        iterator = tqdm(range(episodes), desc="Training", disable=not show_progress)
        
        for episode in iterator:
            self.env.newEpoch()
            state = self.env.getPosition()
            packages_remaining = self.env.getPackagesRemaining()
            total_reward = 0
            step_count = 0
            epsilon = self.get_epsilon(episode)
            epsilons.append(epsilon)
            
            while not self.env.isTerminal():
                action = self.choose_action(state, packages_remaining, epsilon)
                
                # Take action and observe next state
                grid_type, new_pos, new_packages, is_terminal = self.env.takeAction(action)
                reward = 1 if grid_type > 0 else -0.01  # +1 for package, -0.01 otherwise
                
                # Q-learning update
                current_q = self.Q[state[0]-1, state[1]-1, packages_remaining, action]
                next_max_q = 0 if is_terminal else np.max(self.Q[new_pos[0]-1, new_pos[1]-1, new_packages])
                new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
                self.Q[state[0]-1, state[1]-1, packages_remaining, action] = new_q
                
                total_reward += reward
                step_count += 1
                state, packages_remaining = new_pos, new_packages
            
            rewards.append(total_reward)
            steps.append(step_count)
            
            # Update progress bar
            if show_progress:
                iterator.set_postfix({
                    'ε': f"{epsilon:.3f}",
                    'Reward': total_reward,
                    'Steps': step_count,
                    'Avg Reward': f"{np.mean(rewards[-100:]):.2f}"
                })
        
        return rewards, steps, epsilons

    def evaluate(self, episodes=100):
        """Evaluate trained policy over multiple episodes."""
        total_rewards = []
        total_steps = []
        
        for _ in range(episodes):
            self.env.newEpoch()
            state = self.env.getPosition()
            packages_remaining = self.env.getPackagesRemaining()
            episode_reward = 0
            step_count = 0
            
            while not self.env.isTerminal():
                action = np.argmax(self.Q[state[0]-1, state[1]-1, packages_remaining])
                grid_type, new_pos, new_packages, _ = self.env.takeAction(action)
                reward = 1 if grid_type > 0 else -0.01
                
                episode_reward += reward
                step_count += 1
                state, packages_remaining = new_pos, new_packages
            
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
        
        return np.mean(total_rewards), np.mean(total_steps)

def visualize_policy(Q, title="Learned Policy", save_path=None):
    """Visualize the policy using heatmaps and action arrows."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    action_symbols = ['↑', '↓', '←', '→']
    
    for k in range(2):
        policy_grid = np.zeros((11, 11), dtype=object)
        value_grid = np.zeros((11, 11))
        
        for i in range(11):
            for j in range(11):
                best_action = np.argmax(Q[i, j, k])
                policy_grid[i, j] = action_symbols[best_action]
                value_grid[i, j] = np.max(Q[i, j, k])
        
        im = axs[k].imshow(value_grid, cmap='viridis')
        axs[k].set_title(f"Packages Remaining: {k}")
        axs[k].set_xticks(np.arange(11))
        axs[k].set_yticks(np.arange(11))
        
        # Add action arrows
        for i in range(11):
            for j in range(11):
                axs[k].text(j, i, policy_grid[i, j], 
                          ha='center', va='center', 
                          color='white', fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=axs[k], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def show_final_path(env, Q, save_path=None):
    """Demonstrate agent's final path using greedy policy."""
    env.newEpoch()
    state = env.getPosition()
    packages_remaining = env.getPackagesRemaining()
    steps = 0
    
    while not env.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, packages_remaining])
        _, new_pos, new_packages, _ = env.takeAction(action)
        state, packages_remaining = new_pos, new_packages
        steps += 1
    
    env.showPath(-1, savefig=save_path)
    return steps

def moving_average(data, window_size=50):
    """Smooth data using moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, save_dir='', window_size=50):
    """Plot comparison of learning metrics between two strategies."""
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(rewards1, window_size), label='High Exploration (Smoothed)', linewidth=2)
    plt.plot(moving_average(rewards2, window_size), label='Moderate Exploration (Smoothed)', linewidth=2)
    plt.plot(rewards1, alpha=0.2, label='High Exploration (Raw)')
    plt.plot(rewards2, alpha=0.2, label='Moderate Exploration (Raw)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}reward_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(steps1, window_size), label='High Exploration (Smoothed)', linewidth=2)
    plt.plot(moving_average(steps2, window_size), label='Moderate Exploration (Smoothed)', linewidth=2)
    plt.plot(steps1, alpha=0.2, label='High Exploration (Raw)')
    plt.plot(steps2, alpha=0.2, label='Moderate Exploration (Raw)')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Efficiency Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}steps_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons1, label='High Exploration Strategy')
    plt.plot(epsilons2, label='Moderate Exploration Strategy')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Exploration Rate Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()

def sensitivity_analysis(env, param_name, param_values, fixed_params, episodes=500, seed=None, output_dir=''):
    """Analyze parameter sensitivity using multiple training runs."""
    results = {'rewards': [], 'steps': [], 'avg_rewards': [], 'avg_steps': []}
    
    for value in param_values:
        print(f"Testing {param_name}={value}...")
        params = fixed_params.copy()
        params[param_name] = value
        
        # Create and train agent
        agent = QLearningAgent(env, **params, seed=seed)
        rewards, steps, _ = agent.train(episodes, show_progress=False)
        avg_reward, avg_steps = agent.evaluate()
        
        # Store results
        results['rewards'].append(rewards)
        results['steps'].append(steps)
        results['avg_rewards'].append(avg_reward)
        results['avg_steps'].append(avg_steps)
        
        # Plot learning curve for this parameter value
        plt.figure()
        plt.plot(rewards, alpha=0.3)
        plt.plot(moving_average(rewards, 50), linewidth=2)
        plt.title(f"{param_name}={value} Learning Curve")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.savefig(f"{output_dir}{param_name}_{value}_learning.png")
        plt.close()
    
    # Plot parameter sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, results['avg_rewards'], 'o-')
    plt.title(f"Parameter Sensitivity: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig(f"{output_dir}{param_name}_sensitivity.png")
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Q-Learning for FourRooms Scenario 1")
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic transitions')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-output', type=str, default='results/', help='Output directory')
    parser.add_argument('-sensitivity', action='store_true', help='Run sensitivity analysis')
    args = parser.parse_args()

    # Setup environment and output
    os.makedirs(args.output, exist_ok=True)
    env = FourRooms('simple', stochastic=args.stochastic)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Train two exploration strategies
    print("\n=== Training High Exploration Strategy ===")
    agent_high = QLearningAgent(
        env, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01, seed=args.seed
    )
    rewards_high, steps_high, epsilons_high = agent_high.train(args.episodes)

    print("\n=== Training Moderate Exploration Strategy ===")
    agent_mod = QLearningAgent(
        env, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=0.5, epsilon_decay=0.99, min_epsilon=0.01, seed=args.seed
    )
    rewards_mod, steps_mod, epsilons_mod = agent_mod.train(args.episodes)

    # Evaluate policies
    avg_reward_high, avg_steps_high = agent_high.evaluate()
    avg_reward_mod, avg_steps_mod = agent_mod.evaluate()
    print(f"\nHigh Exploration: Avg Reward = {avg_reward_high:.2f}, Avg Steps = {avg_steps_high:.2f}")
    print(f"Moderate Exploration: Avg Reward = {avg_reward_mod:.2f}, Avg Steps = {avg_steps_mod:.2f}")

    # Generate visualizations
    visualize_policy(agent_high.Q, "High Exploration Policy", f"{args.output}policy_high.png")
    visualize_policy(agent_mod.Q, "Moderate Exploration Policy", f"{args.output}policy_mod.png")
    plot_learning_curves(rewards_high, steps_high, epsilons_high,
                        rewards_mod, steps_mod, epsilons_mod, args.output)
    
    # Demonstrate final paths
    show_final_path(env, agent_high.Q, f"{args.output}path_high.png")
    show_final_path(env, agent_mod.Q, f"{args.output}path_mod.png")

    # Sensitivity analysis
    if args.sensitivity:
        print("\n=== Running Sensitivity Analysis ===")
        fixed_params = {
            'alpha': args.alpha,
            'gamma': args.gamma,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01
        }
        sensitivity_analysis(env, 'alpha', [0.01, 0.05, 0.1, 0.2], fixed_params, output_dir=args.output)
        sensitivity_analysis(env, 'gamma', [0.8, 0.9, 0.95, 0.99], fixed_params, output_dir=args.output)

    print("\n=== Training Complete ===")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()