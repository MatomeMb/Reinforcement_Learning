# CSC3022F Reinforcement Learning Assignment

This project implements Q-learning for the Four-Rooms domain across three scenarios.

## Files
- `FourRooms.py`: Environment implementation (updated 16 May 2025).
- `Scenario1.py`: Q-learning for single package collection with exploration strategy comparison.
- `Scenario2.py`: Q-learning for collecting three packages in any order.
- `Scenario3.py`: Q-learning for ordered collection (red, green, blue).
- `requirements.txt`: Python dependencies.
- `README.md`: This file.
- `learning_curves_scenario1.png`: Learning curves for Scenario 1.
- `path_scenario1.png`, `path_scenario2.png`, `path_scenario3.png`: Final paths.
- `scenario1_analysis.pdf`: Exploration strategy analysis for Scenario 1.

## Running
- Scenario 1: `python Scenario1.py [-stochastic]`
- Scenario 2: `python Scenario2.py [-stochastic]`
- Scenario 3: `python Scenario3.py [-stochastic]`

## Notes
- Use `-stochastic` for 20% random action chance.
- Scenario 1 analysis is in `scenario1_analysis.pdf`.
- All scripts save path images for submission.