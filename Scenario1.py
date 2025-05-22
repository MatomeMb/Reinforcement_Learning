import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

def train(fourRoomsObj, Q, alpha, gamma, epsilon_start, epsilon_decay, min_epsilon, episodes=1000, seed=None, show_progress=True):
    """
    Train the Q-learning agent for simple package collection.
    
    Args:
        fourRoomsObj: FourRooms environment object
        Q: Q-table to be updated
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_decay: Decay rate for exploration
        min_epsilon: Minimum exploration rate
        episodes: Number of training episodes
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bar
    
    Returns:
        tuple: (rewards, steps, epsilons) for each episode
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    rewards = []
    steps = []
    epsilons = []
    
    # Create progress bar if requested
    episode_iterator = tqdm(range(episodes), desc="Training", disable=not show_progress)
    
    for episode in episode_iterator:
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        total_reward = 0
        step_count = 0
        epsilon = max(min_epsilon, epsilon_start * (epsilon_decay ** episode))
        epsilons.append(epsilon)
        
        while not fourRoomsObj.isTerminal():
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state[0]-1, state[1]-1, k])
            
            # Take action and observe result
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            reward = 1 if gridType > 0 else -0.01
            
            total_reward += reward
            step_count += 1
            
            # Q-learning update
            next_state = newPos
            next_k = packagesRemaining
            max_next_Q = 0 if isTerminal else np.max(Q[next_state[0]-1, next_state[1]-1, next_k])
            Q[state[0]-1, state[1]-1, k, action] += alpha * (reward + gamma * max_next_Q - Q[state[0]-1, state[1]-1, k, action])
            
            # Update state
            state, k = next_state, next_k
        
        rewards.append(total_reward)
        steps.append(step_count)
        
        # Update progress bar with current metrics
        if show_progress:
            episode_iterator.set_postfix({
                'ε': f"{epsilon:.3f}",
                'reward': f"{total_reward:.2f}",
                'steps': step_count,
                'avg_reward': f"{np.mean(rewards[-100:]):.2f}" if len(rewards) > 0 else "N/A"
            })
    
    return rewards, steps, epsilons

def evaluate_policy(fourRoomsObj, Q, episodes=100, seed=None):
    """
    Evaluate the learned policy over multiple episodes.
    
    Args:
        fourRoomsObj: FourRooms environment object
        Q: Trained Q-table
        episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (average_reward, average_steps)
    """
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
            # Use greedy policy (no exploration)
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
    """
    Visualize the learned policy as a heatmap with action arrows.
    
    Args:
        Q: Trained Q-table
        title: Title for the plot
        save_path: Path to save the visualization
    """
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
                
        # Create heatmap
        im = axes[k].imshow(value, cmap='viridis', aspect='equal')
        
        # Add action arrows
        for i in range(11):
            for j in range(11):
                axes[k].text(j, i, policy[i, j], ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=8)
        
        axes[k].set_title(f"Package Remaining: {k}", fontsize=12)
        axes[k].set_xticks(np.arange(11))
        axes[k].set_yticks(np.arange(11))
        axes[k].set_xticklabels(np.arange(1, 12))
        axes[k].set_yticklabels(np.arange(1, 12))
        axes[k].grid(False)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[k], shrink=0.8)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def show_final_path(fourRoomsObj, Q, save_path=None):
    """
    Demonstrate the final path using the learned policy.
    
    Args:
        fourRoomsObj: FourRooms environment object
        Q: Trained Q-table
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
    """Calculate the moving average of the given data."""
    if window_size > len(data):
        window_size = len(data)
    if window_size <= 0:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, save_dir='', window_size=50):
    """
    Plot learning curves comparing two exploration strategies.
    
    Args:
        rewards1, steps1, epsilons1: Data from first strategy
        rewards2, steps2, epsilons2: Data from second strategy
        save_dir: Directory to save plots
        window_size: Window size for moving average
    """
    # Plot rewards (with moving average)
    plt.figure(figsize=(12, 6))
    if len(rewards1) > window_size:
        rewards1_smooth = moving_average(rewards1, window_size)
        rewards2_smooth = moving_average(rewards2, window_size)
        plt.plot(range(window_size-1, len(rewards1)), rewards1_smooth, 
                 label='High Exploration (Smoothed)', linewidth=2)
        plt.plot(range(window_size-1, len(rewards2)), rewards2_smooth, 
                 label='Moderate Exploration (Smoothed)', linewidth=2)
    plt.plot(rewards1, alpha=0.3, label='High Exploration: Raw', color='blue')
    plt.plot(rewards2, alpha=0.3, label='Moderate Exploration: Raw', color='orange')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Reward Learning Curves - Strategy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}reward_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps per episode (with moving average)
    plt.figure(figsize=(12, 6))
    if len(steps1) > window_size:
        steps1_smooth = moving_average(steps1, window_size)
        steps2_smooth = moving_average(steps2, window_size)
        plt.plot(range(window_size-1, len(steps1)), steps1_smooth, 
                 label='High Exploration (Smoothed)', linewidth=2)
        plt.plot(range(window_size-1, len(steps2)), steps2_smooth, 
                 label='Moderate Exploration (Smoothed)', linewidth=2)
    plt.plot(steps1, alpha=0.3, label='High Exploration: Raw', color='blue')
    plt.plot(steps2, alpha=0.3, label='Moderate Exploration: Raw', color='orange')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.title('Steps per Episode Learning Curves - Strategy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}steps_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons1, label='High Exploration: ε=1.0, decay=0.995', linewidth=2)
    plt.plot(epsilons2, label='Moderate Exploration: ε=0.5, decay=0.99', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Epsilon Value', fontsize=12)
    plt.title('Epsilon Decay Over Training - Strategy Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()

def sensitivity_analysis(fourRoomsObj, parameter_name, parameter_values, fixed_params, episodes=500, seed=None, output_dir=''):
    """
    Perform sensitivity analysis on a specific parameter.
    
    Args:
        fourRoomsObj: The environment object
        parameter_name: Name of the parameter to vary
        parameter_values: List of values to test for the parameter
        fixed_params: Dictionary of fixed parameters
        episodes: Number of episodes to train for each parameter value
        seed: Random seed for reproducibility
        output_dir: Directory to save output files
    
    Returns:
        Dictionary of results for each parameter value
    """
    results = {
        'rewards': [],
        'steps': [],
        'avg_rewards': [],
        'avg_steps': []
    }
    
    for value in parameter_values:
        print(f"Testing {parameter_name}={value}...")
        
        # Create a new Q-table
        Q = np.zeros((11, 11, 2, 4))
        
        # Set the parameter value
        params = fixed_params.copy()
        params[parameter_name] = value
        
        # Train with this parameter value
        rewards, steps, _ = train(
            fourRoomsObj, Q, 
            alpha=params['alpha'], 
            gamma=params['gamma'], 
            epsilon_start=params['epsilon_start'], 
            epsilon_decay=params['epsilon_decay'], 
            min_epsilon=params['min_epsilon'], 
            episodes=episodes, 
            seed=seed,
            show_progress=False
        )
        
        # Evaluate the policy
        avg_reward, avg_steps = evaluate_policy(fourRoomsObj, Q, seed=seed)
        
        # Store results
        results['rewards'].append(rewards)
        results['steps'].append(steps)
        results['avg_rewards'].append(avg_reward)
        results['avg_steps'].append(avg_steps)
        
        print(f"{parameter_name}={value} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
    
    # Plot average rewards for each parameter value
    plt.figure(figsize=(10, 6))
    plt.plot(parameter_values, results['avg_rewards'], 'o-', linewidth=2, markersize=8)
    plt.xlabel(parameter_name, fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'Effect of {parameter_name} on Average Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_reward.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot average steps for each parameter value
    plt.figure(figsize=(10, 6))
    plt.plot(parameter_values, results['avg_steps'], 'o-', linewidth=2, markersize=8)
    plt.xlabel(parameter_name, fontsize=12)
    plt.ylabel('Average Steps', fontsize=12)
    plt.title(f'Effect of {parameter_name} on Average Steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_steps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot learning curves for each parameter value
    plt.figure(figsize=(12, 6))
    for i, value in enumerate(parameter_values):
        if len(results['rewards'][i]) > 50:  # Apply smoothing if enough data
            rewards_smooth = moving_average(results['rewards'][i], 50)
            plt.plot(range(50-1, len(results['rewards'][i])), rewards_smooth, 
                     label=f'{parameter_name}={value}', linewidth=2)
        else:
            plt.plot(results['rewards'][i], label=f'{parameter_name}={value}', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(f'Learning Curves for Different {parameter_name} Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 1: Simple Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('-output_dir', type=str, default='', help='Directory to save output files')
    parser.add_argument('-window_size', type=int, default=50, help='Window size for moving average smoothing')
    parser.add_argument('-sensitivity_analysis', action='store_true', help='Perform hyperparameter sensitivity analysis')
    parser.add_argument('-show_progress', action='store_true', help='Show progress bar during training')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Ensure output directory exists
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not output_dir.endswith('/'):
            output_dir += '/'

    print("=" * 60)
    print("Q-LEARNING FOR SCENARIO 1: SIMPLE PACKAGE COLLECTION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Learning Rate (α): {args.alpha}")
    print(f"  - Discount Factor (γ): {args.gamma}")
    print(f"  - Stochastic Environment: {args.stochastic}")
    print(f"  - Random Seed: {args.seed}")
    print(f"  - Output Directory: {output_dir if output_dir else 'Current directory'}")
    print("-" * 60)

    fourRoomsObj = FourRooms('simple', stochastic=args.stochastic)
    Q1 = np.zeros((11, 11, 2, 4))  # x: 1-11, y: 1-11, k: 0-1, actions: 0-3
    Q2 = np.zeros((11, 11, 2, 4))
    
    # Strategy 1: High exploration
    print("Training Strategy 1: High Exploration (ε=1.0, decay=0.995)...")
    rewards1, steps1, epsilons1 = train(fourRoomsObj, Q1, args.alpha, args.gamma, 
                                   epsilon_start=1.0, epsilon_decay=0.995, 
                                   min_epsilon=0.01, episodes=args.episodes, 
                                   seed=args.seed, show_progress=args.show_progress)
    
    # Strategy 2: Moderate exploration
    print("Training Strategy 2: Moderate Exploration (ε=0.5, decay=0.99)...")
    rewards2, steps2, epsilons2 = train(fourRoomsObj, Q2, args.alpha, args.gamma, 
                                        epsilon_start=0.5, epsilon_decay=0.99, 
                                        min_epsilon=0.01, episodes=args.episodes, 
                                        seed=args.seed, show_progress=args.show_progress)
    
    # Evaluate both policies
    avg_reward1, avg_steps1 = evaluate_policy(fourRoomsObj, Q1, seed=args.seed)
    avg_reward2, avg_steps2 = evaluate_policy(fourRoomsObj, Q2, seed=args.seed)
    
    print("\n" + "=" * 60)
    print("EXPLORATION STRATEGY COMPARISON RESULTS:")
    print("=" * 60)
    print(f"High Exploration Strategy:")
    print(f"  - Average Reward: {avg_reward1:.2f}")
    print(f"  - Average Steps: {avg_steps1:.2f}")
    print(f"  - Final Training Reward: {np.mean(rewards1[-100:]):.2f}")
    print(f"Moderate Exploration Strategy:")
    print(f"  - Average Reward: {avg_reward2:.2f}")
    print(f"  - Average Steps: {avg_steps2:.2f}")
    print(f"  - Final Training Reward: {np.mean(rewards2[-100:]):.2f}")
    print("-" * 60)
    
    # Visualize policies
    print("Generating policy visualizations...")
    visualize_policy(Q1, "High Exploration Policy", f'{output_dir}policy_high_exploration.png')
    visualize_policy(Q2, "Moderate Exploration Policy", f'{output_dir}policy_moderate_exploration.png')
    
    # Plot learning curves
    print("Generating learning curve comparisons...")
    plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, output_dir, args.window_size)
    
    # Show final paths
    print("Generating path visualizations...")
    steps1_final = show_final_path(fourRoomsObj, Q1, f'{output_dir}path_high_exploration.png')
    print(f"High exploration final path: {steps1_final} steps")
    
    steps2_final = show_final_path(fourRoomsObj, Q2, f'{output_dir}path_moderate_exploration.png')
    print(f"Moderate exploration final path: {steps2_final} steps")
    
    # Compare different gamma values
    print("\n--- Comparing different discount factors (gamma) ---")
    gammas = [0.8, 0.9, 0.99]
    Q_gammas = [np.zeros((11, 11, 2, 4)) for _ in range(len(gammas))]
    rewards_gammas = []
    steps_gammas = []
    
    for i, gamma in enumerate(gammas):
        print(f"Training with γ={gamma}...")
        rewards, steps, _ = train(fourRoomsObj, Q_gammas[i], args.alpha, gamma, 
                                epsilon_start=1.0, epsilon_decay=0.995, 
                                min_epsilon=0.01, episodes=args.episodes, 
                                seed=args.seed, show_progress=False)
        rewards_gammas.append(rewards)
        steps_gammas.append(steps)
        
        # Evaluate policy
        avg_reward, avg_steps = evaluate_policy(fourRoomsObj, Q_gammas[i], seed=args.seed)
        print(f"γ={gamma} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
        
        # Visualize policy
        visualize_policy(Q_gammas[i], f"Policy with γ={gamma}", 
                        f'{output_dir}policy_gamma_{gamma}.png')
    
    # Plot gamma comparison
    plt.figure(figsize=(12, 6))
    for i, gamma in enumerate(gammas):
        if len(rewards_gammas[i]) > args.window_size:
            rewards_smooth = moving_average(rewards_gammas[i], args.window_size)
            plt.plot(range(args.window_size-1, len(rewards_gammas[i])), rewards_smooth, 
                     label=f'γ={gamma}', linewidth=2)
        else:
            plt.plot(rewards_gammas[i], label=f'γ={gamma}', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Effect of Discount Factor (γ) on Learning', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}gamma_comparison_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    for i, gamma in enumerate(gammas):
        if len(steps_gammas[i]) > args.window_size:
            steps_smooth = moving_average(steps_gammas[i], args.window_size)
            plt.plot(range(args.window_size-1, len(steps_gammas[i])), steps_smooth, 
                     label=f'γ={gamma}', linewidth=2)
        else:
            plt.plot(steps_gammas[i], label=f'γ={gamma}', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.title('Effect of Discount Factor (γ) on Steps per Episode', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}gamma_comparison_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Hyperparameter sensitivity analysis
    if args.sensitivity_analysis:
        print("\n--- Performing Hyperparameter Sensitivity Analysis ---")
        
        # Fixed parameters
        fixed_params = {
            'alpha': args.alpha,
            'gamma': args.gamma,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01
        }
        
        # Test different learning rates
        print("Analyzing learning rate sensitivity...")
        alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        sensitivity_analysis(fourRoomsObj, 'alpha', alpha_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)
        
        # Test different epsilon decay rates
        print("Analyzing epsilon decay sensitivity...")
        epsilon_decay_values = [0.99, 0.995, 0.998, 0.999]
        sensitivity_analysis(fourRoomsObj, 'epsilon_decay', epsilon_decay_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)
        
        # Test different minimum epsilon values
        print("Analyzing minimum epsilon sensitivity...")
        min_epsilon_values = [0.001, 0.01, 0.05, 0.1]
        sensitivity_analysis(fourRoomsObj, 'min_epsilon', min_epsilon_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("SCENARIO 1 COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print(f"  - {output_dir}reward_learning_curves.png")
    print(f"  - {output_dir}steps_learning_curves.png")
    print(f"  - {output_dir}epsilon_decay.png")
    print(f"  - {output_dir}policy_high_exploration.png")
    print(f"  - {output_dir}policy_moderate_exploration.png")
    print(f"  - {output_dir}path_high_exploration.png")
    print(f"  - {output_dir}path_moderate_exploration.png")
    print(f"  - {output_dir}gamma_comparison_*.png")
    if args.sensitivity_analysis:
        print(f"  - {output_dir}sensitivity_*.png (multiple files)")
    print("=" * 60)

if __name__ == "__main__":
    main()