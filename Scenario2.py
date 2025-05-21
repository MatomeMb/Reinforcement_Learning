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

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 2: Multiple Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    parser.add_argument('-seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)
    Q = np.zeros((11, 11, 4, 4))  # x: 1-11, y: 1-11, k: 0-3, actions: 0-3
    
    alpha, gamma = 0.1, 0.9
    
    # Train with high exploration
    rewards, steps, epsilons = train(fourRoomsObj, Q, alpha, gamma, 
                                    epsilon_start=1.0, epsilon_decay=0.995, 
                                    min_epsilon=0.01, seed=args.seed)
    
    # Show final path
    fourRoomsObj.newEpoch()
    state = fourRoomsObj.getPosition()
    k = fourRoomsObj.getPackagesRemaining()
    while not fourRoomsObj.isTerminal():
        action = np.argmax(Q[state[0]-1, state[1]-1, k])
        _, newPos, packagesRemaining, _ = fourRoomsObj.takeAction(action)
        state, k = newPos, packagesRemaining
    fourRoomsObj.showPath(-1, savefig='path_scenario2.png')

if __name__ == "__main__":
    main()