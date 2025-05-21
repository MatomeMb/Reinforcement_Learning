import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm

def train(fourRoomsObj, Q, alpha, gamma, epsilon_start, epsilon_decay, min_epsilon, episodes=1000, seed=None, show_progress=True):
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
        
        # Update progress bar with current metrics
        if show_progress:
            episode_iterator.set_postfix({
                'ε': f"{epsilon:.3f}",
                'reward': f"{total_reward:.2f}",
                'steps': step_count,
                'avg_reward': f"{np.mean(rewards[-100:]):.2f}" if len(rewards) > 0 else "N/A"
            })
    
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

def moving_average(data, window_size):
    """Calculate the moving average of the given data."""
    if window_size > len(data):
        window_size = len(data)
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def plot_learning_curves(rewards1, steps1, epsilons1, rewards2, steps2, epsilons2, save_dir='', window_size=50):
    # Plot rewards (with moving average)
    plt.figure(figsize=(10, 5))
    if len(rewards1) > window_size:
        rewards1_smooth = moving_average(rewards1, window_size)
        rewards2_smooth = moving_average(rewards2, window_size)
        plt.plot(range(window_size-1, len(rewards1)), rewards1_smooth, 
                 label='High Exploration (Smoothed)')
        plt.plot(range(window_size-1, len(rewards2)), rewards2_smooth, 
                 label='Moderate Exploration (Smoothed)')
    plt.plot(rewards1, alpha=0.3, label='High Exploration: Raw')
    plt.plot(rewards2, alpha=0.3, label='Moderate Exploration: Raw')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward Learning Curves')
    plt.legend()
    plt.savefig(f'{save_dir}reward_learning_curves.png')
    plt.close()
    
    # Plot steps per episode (with moving average)
    plt.figure(figsize=(10, 5))
    if len(steps1) > window_size:
        steps1_smooth = moving_average(steps1, window_size)
        steps2_smooth = moving_average(steps2, window_size)
        plt.plot(range(window_size-1, len(steps1)), steps1_smooth, 
                 label='High Exploration (Smoothed)')
        plt.plot(range(window_size-1, len(steps2)), steps2_smooth, 
                 label='Moderate Exploration (Smoothed)')
    plt.plot(steps1, alpha=0.3, label='High Exploration: Raw')
    plt.plot(steps2, alpha=0.3, label='Moderate Exploration: Raw')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode Learning Curves')
    plt.legend()
    plt.savefig(f'{save_dir}steps_learning_curves.png')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons1, label='High Exploration: ε=1.0, decay=0.995')
    plt.plot(epsilons2, label='Moderate Exploration: ε=0.5, decay=0.99')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Over Training')
    plt.legend()
    plt.savefig(f'{save_dir}epsilon_decay.png')
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
            seed=seed
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
    plt.figure(figsize=(10, 5))
    plt.plot(parameter_values, results['avg_rewards'], 'o-')
    plt.xlabel(parameter_name)
    plt.ylabel('Average Reward')
    plt.title(f'Effect of {parameter_name} on Average Reward')
    plt.grid(True)
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_reward.png')
    plt.close()
    
    # Plot average steps for each parameter value
    plt.figure(figsize=(10, 5))
    plt.plot(parameter_values, results['avg_steps'], 'o-')
    plt.xlabel(parameter_name)
    plt.ylabel('Average Steps')
    plt.title(f'Effect of {parameter_name} on Average Steps')
    plt.grid(True)
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_steps.png')
    plt.close()
    
    # Plot learning curves for each parameter value
    plt.figure(figsize=(10, 5))
    for i, value in enumerate(parameter_values):
        if len(results['rewards'][i]) > 50:  # Apply smoothing if enough data
            rewards_smooth = moving_average(results['rewards'][i], 50)
            plt.plot(range(50-1, len(results['rewards'][i])), rewards_smooth, 
                     label=f'{parameter_name}={value}')
        else:
            plt.plot(results['rewards'][i], label=f'{parameter_name}={value}')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curves for Different {parameter_name} Values')
    plt.legend()
    plt.savefig(f'{output_dir}sensitivity_{parameter_name}_learning_curves.png')
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
                                   seed=args.seed, show_progress=args.show_progress)
    
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
    
    # Compare different gamma values
    print("\n--- Comparing different gamma values ---")
    gammas = [0.8, 0.9, 0.99]
    Q_gammas = [np.zeros((11, 11, 2, 4)) for _ in range(len(gammas))]
    rewards_gammas = []
    steps_gammas = []
    
    for i, gamma in enumerate(gammas):
        print(f"Training with gamma={gamma}...")
        rewards, steps, _ = train(fourRoomsObj, Q_gammas[i], args.alpha, gamma, 
                                epsilon_start=1.0, epsilon_decay=0.995, 
                                min_epsilon=0.01, episodes=args.episodes, 
                                seed=args.seed)
        rewards_gammas.append(rewards)
        steps_gammas.append(steps)
        
        # Evaluate policy
        avg_reward, avg_steps = evaluate_policy(fourRoomsObj, Q_gammas[i], seed=args.seed)
        print(f"Gamma={gamma} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
        
        # Visualize policy
        visualize_policy(Q_gammas[i], f"Policy with Gamma={gamma}", 
                        f'{output_dir}policy_gamma_{gamma}.png')
    
    # Plot comparison of rewards
    plt.figure(figsize=(10, 5))
    for i, gamma in enumerate(gammas):
        if len(rewards_gammas[i]) > args.window_size:  # Apply smoothing if enough data
            rewards_smooth = moving_average(rewards_gammas[i], args.window_size)
            plt.plot(range(args.window_size-1, len(rewards_gammas[i])), rewards_smooth, 
                     label=f'Gamma={gamma}')
        else:
            plt.plot(rewards_gammas[i], label=f'Gamma={gamma}')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Effect of Discount Factor (Gamma) on Learning')
    plt.legend()
    plt.savefig(f'{output_dir}gamma_comparison_rewards.png')
    plt.close()
    
    # Plot comparison of steps
    plt.figure(figsize=(10, 5))
    for i, gamma in enumerate(gammas):
        if len(steps_gammas[i]) > args.window_size:  # Apply smoothing if enough data
            steps_smooth = moving_average(steps_gammas[i], args.window_size)
            plt.plot(range(args.window_size-1, len(steps_gammas[i])), steps_smooth, 
                     label=f'Gamma={gamma}')
        else:
            plt.plot(steps_gammas[i], label=f'Gamma={gamma}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per Episode')
    plt.title('Effect of Discount Factor (Gamma) on Steps per Episode')
    plt.legend()
    plt.savefig(f'{output_dir}gamma_comparison_steps.png')
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
        alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        sensitivity_analysis(fourRoomsObj, 'alpha', alpha_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)
        
        # Test different epsilon decay rates
        epsilon_decay_values = [0.99, 0.995, 0.998, 0.999]
        sensitivity_analysis(fourRoomsObj, 'epsilon_decay', epsilon_decay_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)
        
        # Test different minimum epsilon values
        min_epsilon_values = [0.001, 0.01, 0.05, 0.1]
        sensitivity_analysis(fourRoomsObj, 'min_epsilon', min_epsilon_values, fixed_params, 
                            episodes=args.episodes//2, seed=args.seed, output_dir=output_dir)

if __name__ == "__main__":
    main()