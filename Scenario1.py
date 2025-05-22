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
            
            max_steps = 200  # Prevent infinite loops
            
            while not self.env.isTerminal() and step_count < max_steps:
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
                    'Reward': f"{total_reward:.2f}",
                    'Steps': step_count,
                    'Avg Reward': f"{np.mean(rewards[-100:]):.2f}"
                })
        
        return rewards, steps, epsilons

    def evaluate(self, episodes=50):  # REDUCED from 100 to prevent memory issues
        """Evaluate trained policy over multiple episodes."""
        total_rewards = []
        total_steps = []
        
        for _ in range(episodes):
            self.env.newEpoch()
            state = self.env.getPosition()
            packages_remaining = self.env.getPackagesRemaining()
            episode_reward = 0
            step_count = 0
            max_steps = 100  # Safety limit
            
            while not self.env.isTerminal() and step_count < max_steps:
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
    else:
        plt.show()

def show_final_path(env, Q, save_path=None):
    """Demonstrate agent's final path using greedy policy."""
    env.newEpoch()
    state = env.getPosition()
    packages_remaining = env.getPackagesRemaining()
    steps = 0
    max_steps = 100  # Safety limit
    
    while not env.isTerminal() and steps < max_steps:
        action = np.argmax(Q[state[0]-1, state[1]-1, packages_remaining])
        _, new_pos, new_packages, _ = env.takeAction(action)
        state, packages_remaining = new_pos, new_packages
        steps += 1
    
    env.showPath(-1, savefig=save_path)
    return steps

def moving_average(data, window_size=50):
    """Smooth data using moving average."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, save_dir='', window_size=50):
    """Plot comparison of learning metrics between two strategies."""
    # Plot rewards
    plt.figure(figsize=(12, 6))
    smooth1 = moving_average(rewards1, window_size)
    smooth2 = moving_average(rewards2, window_size)
    
    plt.plot(rewards1, alpha=0.2, color='blue', label='High Exploration (Raw)')
    plt.plot(rewards2, alpha=0.2, color='red', label='Moderate Exploration (Raw)')
    
    if len(smooth1) > 0:
        plt.plot(range(window_size-1, len(rewards1)), smooth1, 
                color='blue', linewidth=2, label='High Exploration (Smoothed)')
    if len(smooth2) > 0:
        plt.plot(range(window_size-1, len(rewards2)), smooth2, 
                color='red', linewidth=2, label='Moderate Exploration (Smoothed)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}reward_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    smooth1_steps = moving_average(steps1, window_size)
    smooth2_steps = moving_average(steps2, window_size)
    
    plt.plot(steps1, alpha=0.2, color='blue', label='High Exploration (Raw)')
    plt.plot(steps2, alpha=0.2, color='red', label='Moderate Exploration (Raw)')
    
    if len(smooth1_steps) > 0:
        plt.plot(range(window_size-1, len(steps1)), smooth1_steps, 
                color='blue', linewidth=2, label='High Exploration (Smoothed)')
    if len(smooth2_steps) > 0:
        plt.plot(range(window_size-1, len(steps2)), smooth2_steps, 
                color='red', linewidth=2, label='Moderate Exploration (Smoothed)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}steps_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons1, label='High Exploration Strategy', color='blue')
    plt.plot(epsilons2, label='Moderate Exploration Strategy', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Exploration Rate Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()

def sensitivity_analysis(env, param_name, param_values, fixed_params, episodes=200, seed=None, output_dir=''):
    """SIMPLIFIED sensitivity analysis to prevent memory issues."""
    results = {'avg_rewards': [], 'avg_steps': []}
    
    print(f"\n=== {param_name.upper()} Sensitivity Analysis ===")
    
    for value in param_values:
        print(f"Testing {param_name}={value}...")
        params = fixed_params.copy()
        params[param_name] = value
        
        # Create and train agent with fewer episodes
        agent = QLearningAgent(env, **params, seed=seed)
        rewards, steps, _ = agent.train(episodes, show_progress=False)
        
        # Simple evaluation
        final_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        final_steps = np.mean(steps[-20:]) if len(steps) >= 20 else np.mean(steps)
        
        results['avg_rewards'].append(final_reward)
        results['avg_steps'].append(final_steps)
        
        print(f"  Final Performance: Reward={final_reward:.2f}, Steps={final_steps:.1f}")
    
    # Plot parameter sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, results['avg_rewards'], 'o-', linewidth=2, markersize=8)
    plt.title(f"Parameter Sensitivity: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}sensitivity_{param_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Q-Learning for FourRooms Scenario 1")
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic transitions')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    parser.add_argument('-episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('-show_progress', action='store_true', help='Show training progress')
    parser.add_argument('-sensitivity_analysis', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('-window_size', type=int, default=50, help='Moving average window size')
    args = parser.parse_args()

    # Setup environment and output
    os.makedirs(args.output_dir, exist_ok=True)
    env = FourRooms('simple', stochastic=args.stochastic)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Train two exploration strategies
    print("\n=== Training High Exploration Strategy ===")
    agent_high = QLearningAgent(
        env, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01, seed=args.seed
    )
    rewards_high, steps_high, epsilons_high = agent_high.train(args.episodes, show_progress=args.show_progress)

    print("\n=== Training Moderate Exploration Strategy ===")
    agent_mod = QLearningAgent(
        env, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=0.5, epsilon_decay=0.99, min_epsilon=0.01, seed=args.seed+1  # Different seed
    )
    rewards_mod, steps_mod, epsilons_mod = agent_mod.train(args.episodes, show_progress=args.show_progress)

    # Evaluate policies
    print("\n=== Evaluating Policies ===")
    avg_reward_high, avg_steps_high = agent_high.evaluate(episodes=30)  # Reduced
    avg_reward_mod, avg_steps_mod = agent_mod.evaluate(episodes=30)     # Reduced
    
    print(f"High Exploration: Avg Reward = {avg_reward_high:.2f}, Avg Steps = {avg_steps_high:.2f}")
    print(f"Moderate Exploration: Avg Reward = {avg_reward_mod:.2f}, Avg Steps = {avg_steps_mod:.2f}")

    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    visualize_policy(agent_high.Q, "High Exploration Policy", f"{args.output_dir}policy_high_exploration.png")
    visualize_policy(agent_mod.Q, "Moderate Exploration Policy", f"{args.output_dir}policy_moderate_exploration.png")
    
    plot_learning_curves(rewards_high, steps_high, epsilons_high,
                        rewards_mod, steps_mod, epsilons_mod, 
                        args.output_dir, args.window_size)
    
    # Demonstrate final paths
    print("\n=== Generating Final Paths ===")
    steps_high = show_final_path(env, agent_high.Q, f"{args.output_dir}path_high_exploration.png")
    steps_mod = show_final_path(env, agent_mod.Q, f"{args.output_dir}path_moderate_exploration.png")
    
    print(f"High Exploration final path: {steps_high} steps")
    print(f"Moderate Exploration final path: {steps_mod} steps")

    # Sensitivity analysis (optional and simplified)
    if args.sensitivity_analysis:
        print("\n=== Running Sensitivity Analysis ===")
        fixed_params = {
            'alpha': args.alpha,
            'gamma': args.gamma,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01
        }
        
        try:
            sensitivity_analysis(env, 'alpha', [0.05, 0.1, 0.2, 0.3], 
                               fixed_params, episodes=200, seed=args.seed, output_dir=args.output_dir)
            sensitivity_analysis(env, 'gamma', [0.8, 0.9, 0.95, 0.99], 
                               fixed_params, episodes=200, seed=args.seed, output_dir=args.output_dir)
        except Exception as e:
            print(f"Sensitivity analysis error (non-critical): {e}")

    print(f"\n=== Scenario 1 Complete ===")
    print(f"Results saved to: {args.output_dir}")
    print(f"Key files generated:")
    print(f"  - reward_learning_curves.png")
    print(f"  - steps_learning_curves.png") 
    print(f"  - epsilon_decay.png")
    print(f"  - policy_high_exploration.png")
    print(f"  - policy_moderate_exploration.png")
    print(f"  - path_high_exploration.png")
    print(f"  - path_moderate_exploration.png")

if __name__ == "__main__":
    main()