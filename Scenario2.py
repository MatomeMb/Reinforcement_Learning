import numpy as np
import argparse
from FourRooms import FourRooms

def main():
    parser = argparse.ArgumentParser(description='Q-learning for Scenario 2: Multiple Package Collection')
    parser.add_argument('-stochastic', action='store_true', help='Enable stochastic actions')
    args = parser.parse_args()

    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)
    Q = np.zeros((11, 11, 4, 4))  # x: 1-11, y: 1-11, k: 0-3, actions: 0-3
    alpha, gamma, epsilon, epsilon_decay, min_epsilon = 0.1, 0.9, 1.0, 0.995, 0.01
    
    for episode in range(1000):
        fourRoomsObj.newEpoch()
        state = fourRoomsObj.getPosition()
        k = fourRoomsObj.getPackagesRemaining()
        while not fourRoomsObj.isTerminal():
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state[0]-1, state[1]-1, k])
            
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            reward = 1 if gridType > 0 else -0.01
            
            next_state = newPos
            next_k = packagesRemaining
            max_next_Q = 0 if isTerminal else np.max(Q[next_state[0]-1, next_state[1]-1, next_k])
            Q[state[0]-1, state[1]-1, k, action] += alpha * (reward + gamma * max_next_Q - Q[state[0]-1, state[1]-1, k, action])
            state, k = next_state, next_k
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
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