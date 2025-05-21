# CSC3022F Reinforcement Learning Assignment

This project implements Q-learning for the Four-Rooms domain across three scenarios.

## Files
- `FourRooms.py`: Environment implementation (updated 16 May 2025).
- `Scenario1.py`: Q-learning for single package collection with exploration strategy comparison.
- `Scenario2.py`: Q-learning for collecting three packages in any order.
- `Scenario3.py`: Q-learning for ordered collection (red, green, blue).
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Scenario 1 Features
- Comparison of different exploration strategies
- Visualization of rewards and steps per episode
- Epsilon decay visualization
- Moving average smoothing for learning curves
- Discount factor (gamma) comparison
- Hyperparameter sensitivity analysis
- Progress bar for training

## Running
- Scenario 1: `python Scenario1.py [options]`
- Scenario 2: `python Scenario2.py [-stochastic]`
- Scenario 3: `python Scenario3.py [-stochastic]`

### Scenario 1 Command-Line Options
- `-stochastic`: Enable 20% random action chance
- `-episodes N`: Set number of training episodes (default: 1000)
- `-alpha A`: Set learning rate (default: 0.1)
- `-gamma G`: Set discount factor (default: 0.9)
- `-seed S`: Set random seed for reproducibility (default: 42)
- `-window_size W`: Set window size for moving average smoothing (default: 50)
- `-show_progress`: Show progress bar during training
- `-output_dir DIR`: Specify directory for saving output files
- `-sensitivity_analysis`: Perform hyperparameter sensitivity analysis

### Examples
```
# Basic run
python Scenario1.py

# Run with stochastic environment and progress bar
python Scenario1.py -stochastic -show_progress

# Run with custom parameters and save to specific directory
python Scenario1.py -episodes 2000 -alpha 0.2 -gamma 0.95 -output_dir results/

# Run hyperparameter sensitivity analysis
python Scenario1.py -sensitivity_analysis -episodes 500
```

## Output Files
Scenario 1 generates the following visualizations:
- `reward_learning_curves.png`: Learning curves showing rewards over episodes
- `steps_learning_curves.png`: Learning curves showing steps per episode
- `epsilon_decay.png`: Visualization of exploration rate decay
- `policy_high_exploration.png` and `policy_moderate_exploration.png`: Policy visualizations
- `path_high_exploration.png` and `path_moderate_exploration.png`: Final path visualizations
- `gamma_comparison_rewards.png` and `gamma_comparison_steps.png`: Comparison of different discount factors
- `sensitivity_*.png`: Hyperparameter sensitivity analysis plots

## Notes
- Use `-stochastic` for 20% random action chance.
- Scenario 1 analysis is in `scenario1_analysis.pdf`.
- All scripts save path images for submission.