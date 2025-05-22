import numpy as np
import argparse
import matplotlib.pyplot as plt
from FourRooms import FourRooms
import random
from tqdm import tqdm
import os

def train(fourRoomsObj, Q, alpha, gamma, epsilon_start, epsilon_decay, min_epsilon, episodes=1000, seed=None, show_progress=False):
    """
    Train the Q-learning agent for ordered package collection.
    
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
    successful_episodes = 0
    
    # Create progress bar if requested
    episode_iterator = tqdm(range(episodes), desc="Training Scenario 3", disable=not show_progress)
    
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
            
            # Special reward function for ordered collection
            if gridType > 0:
                # Give negative reward if episode terminates with packages remaining (wrong order)
                reward = -10 if isTerminal and packagesRemaining > 0 else 10
            else:
                reward = -0.01  # Small negative reward for each step
            
            total_reward += reward
            step_count += 1
            
            # Q-learning update
            next_state = newPos
            next_k = packagesRemaining
            max_next_Q = 0 if isTerminal else np.max(Q[next_state[0]-1, next_state[1]-1, next_k])
            Q[state[0]-1, state[1]-1, k, action] += alpha * (reward + gamma * max_next_Q - Q[state[0]-1, state[1]-1, k, action])
            
            # Update state
            state, k = next_state, next_k
        
        # Check if episode was successful (all packages collected)
        if packagesRemaining == 0:
            successful_episodes += 1
        
        rewards.append(total_reward)
        steps.append(step_count)
        
        # Update progress bar with current metrics
        if show_progress:
            success_rate = successful_episodes / (episode + 1) * 100
            episode_iterator.set_postfix({
                'ε': f"{epsilon:.3f}",
                'reward': f"{total_reward:.2f}",
                'steps': step_count,
                'success': f"{success_rate:.1f}%"
            })
    
    print(f"Training completed with {successful_episodes}/{episodes} successful episodes ({successful_episodes/episodes*100:.1f}%)")
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
        tuple: (average_reward, average_steps, success_rate)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    total_rewards = []
    total_steps = []
    successful_episodes = 0
    
    for _ in range(episodes):
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        episode_reward = 0
        steps = 0
        
        while not fourRoomsObj.isTerminal():
            # Use greedy policy (no exploration)
            action = np.argmax(Q[state[0]-1, state[1]-1, k])
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            
            if gridType > 0:
                reward = -10 if isTerminal and packagesRemaining > 0 else 10
            else:
                reward = -0.01
                
            episode_reward += reward
            steps += 1
            state, k = newPos, packagesRemaining
        
        if packagesRemaining == 0:
            successful_episodes += 1
            
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    success_rate = successful_episodes / episodes
    return np.mean(total_rewards), np.mean(total_steps), success_rate

def visualize_policy(Q, save_path="policy_scenario3.png"):
    """
    Visualize the learned policy as a heatmap with action arrows.
    
    Args:
        Q: Trained Q-table
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_symbols = ['↑', '↓', '←', '→']
    package_states = ['All Collected', 'Blue Remaining', 'Green+Blue Remaining', 'All Remaining']
    
    for k in range(4):
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
        
        axes[k].set_title(f"{package_states[k]} (k={k})", fontsize=12)
        axes[k].set_xticks(np.arange(11))
        axes[k].set_yticks(np.arange(11))
        axes[k].set_xticklabels(np.arange(1, 12))
        axes[k].set_yticklabels(np.arange(1, 12))
        axes[k].grid(False)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[k], shrink=0.8)
    
    plt.suptitle("Learned Policy for Scenario 3: Ordered Package Collection (R→G→B)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def moving_average(data, window_size):
    """
    Calculate the moving average of the data.
    
    Args:
        data: List of values
        window_size: Size of the moving window
    
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
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
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    if len(rewards) >= window_size:
        rewards_smooth = moving_average(rewards, window_size)
        plt.plot(range(window_size-1, len(rewards)), rewards_smooth, 
                 color='blue', linewidth=2, label=f'Moving Average (window={window_size})')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Rewards per Episode - Scenario 3 (Ordered Collection)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}reward_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot steps
    plt.figure(figsize=(12, 6))
    plt.plot(steps, alpha=0.3, color='green', label='Raw Steps')
    if len(steps) >= window_size:
        steps_smooth = moving_average(steps, window_size)
        plt.plot(range(window_size-1, len(steps)), steps_smooth, 
                 color='green', linewidth=2, label=f'Moving Average (window={window_size})')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.title('Steps per Episode - Scenario 3 (Ordered Collection)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}steps_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, color='red', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Epsilon Value', fontsize=12)
    plt.title('Epsilon Decay Over Training - Scenario 3', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}epsilon_decay.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 3: Ordered Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('-alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=0.95, help='Discount factor')
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

    print("=" * 60)
    print("Q-LEARNING FOR SCENARIO 3: ORDERED PACKAGE COLLECTION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Learning Rate (α): {args.alpha}")
    print(f"  - Discount Factor (γ): {args.gamma}")
    print(f"  - Stochastic Environment: {args.stochastic}")
    print(f"  - Random Seed: {args.seed}")
    print(f"  - Collection Order: Red → Green → Blue")
    print(f"  - Output Directory: {output_dir if output_dir else 'Current directory'}")
    print("-" * 60)

    # Initialize FourRooms environment for Scenario 3
    fourRoomsObj = FourRooms('rgb', stochastic=args.stochastic)
    Q = np.zeros((11, 11, 4, 4))  # State space: x: 1-11, y: 1-11, k: 0-3, actions: 0-3
    
    # Train the Q-learning agent
    print("Starting training...")
    rewards, steps, epsilons = train(fourRoomsObj, Q, args.alpha, args.gamma, 
                                    epsilon_start=1.0, epsilon_decay=0.998, 
                                    min_epsilon=0.05, episodes=args.episodes, 
                                    seed=args.seed, show_progress=args.show_progress)
    
    print(f"Training completed!")
    print(f"Final epsilon: {epsilons[-1]:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Average steps (last 100 episodes): {np.mean(steps[-100:]):.2f}")
    
    # Generate visualizations for learning curves and policy
    print("Generating learning curve visualizations...")
    plot_learning_curves(rewards, steps, epsilons, window_size=args.window_size, save_dir=output_dir)
    
    print("Generating policy visualization...")
    visualize_policy(Q, save_path=f"{output_dir}policy_scenario3.png")
    
    # Evaluate the learned policy
    print("Evaluating learned policy...")
    avg_reward, avg_steps, success_rate = evaluate_policy(fourRoomsObj, Q, episodes=100, seed=args.seed)
    print(f"Policy Evaluation Results:")
    print(f"  - Average Reward: {avg_reward:.2f}")
    print(f"  - Average Steps: {avg_steps:.2f}")
    print(f"  - Success Rate: {success_rate*100:.1f}%")
    
    # Demonstrate the final learned path
    print("Generating final path demonstration...")
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    path_steps = 0
    
    while not fourRoomsObj.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, k])
        _, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
        state, k = newPos, packagesRemaining
        path_steps += 1
    
    fourRoomsObj.showPath(-1, savefig=f"{output_dir}path_scenario3.png")
    
    if packagesRemaining == 0:
        print(f"SUCCESS: Final path completed in {path_steps} steps with all packages collected in order!")
    else:
        print(f"Final path completed in {path_steps} steps, but {packagesRemaining} packages remain (wrong order)")
    
    print("=" * 60)
    print("SCENARIO 3 COMPLETED!")
    print("Generated files:")
    print(f"  - {output_dir}reward_learning_curve.png")
    print(f"  - {output_dir}steps_learning_curve.png")
    print(f"  - {output_dir}epsilon_decay.png")
    print(f"  - {output_dir}policy_scenario3.png")
    print(f"  - {output_dir}path_scenario3.png")
    print("=" * 60)

if __name__ == "__main__":
    main()]-1, state[