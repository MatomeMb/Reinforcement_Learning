# Reinforcement Learning

This project implements Q-learning algorithms for the Four-Rooms domain across three distinct scenarios, demonstrating different aspects of reinforcement learning including exploration strategies, multi-objective learning, and sequential decision making.

## Project Structure

### Core Files
- `FourRooms.py`: Environment implementation (provided, updated 16 May 2025)
- `Scenario1.py`: Q-learning with exploration strategy comparison and analysis
- `Scenario2.py`: Q-learning for collecting multiple packages (any order)
- `Scenario3.py`: Q-learning for ordered package collection (Red → Green → Blue)
- `requirements.txt`: Python dependencies
- `quick_test.py`: Comprehensive testing script for all scenarios
- `README.md`: This documentation file

### Output Directory
- `results/`: Contains all generated visualizations and analysis plots

## Scenarios Overview

### Scenario 1: Single Package Collection with Exploration Analysis
**Objective**: Collect one package while comparing different exploration strategies

**Key Features**:
- Comparison of High vs. Moderate exploration strategies
- Comprehensive visualization of learning curves
- Epsilon decay analysis
- Policy visualization with action arrows
- Hyperparameter sensitivity analysis
- Moving average smoothing for cleaner analysis
- Discount factor comparison studies

**Performance**: Achieves optimal performance (0.96 reward, 5 steps average)

### Scenario 2: Multiple Package Collection
**Objective**: Collect all three packages in any order

**Key Features**:
- Extended state space handling (4 package states: 0-3 remaining)
- Enhanced reward structure (+10 for packages, -0.01 for movement)
- Safety mechanisms (step limits to prevent infinite loops)
- Policy visualization across all package states

**Performance**: Successfully collects packages with average reward of 18-29

### Scenario 3: Ordered Package Collection
**Objective**: Collect packages in strict order (Red → Green → Blue)

**Key Features**:
- Complex reward structure enforcing order constraints
- Enhanced penalties for wrong-order collection (-200)
- Large completion bonuses (+200)
- Higher learning rate (0.3) and discount factor (0.95)
- Success rate tracking

**Performance**: Challenging scenario requiring extended training for optimal results

## Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick functionality test
python quick_test.py
```

## Usage Instructions

### Basic Usage
```bash
# Run each scenario with default parameters
python Scenario1.py
python Scenario2.py  
python Scenario3.py
```

### Advanced Options

#### Scenario 1 (Full Feature Set)
```bash
# Basic run with progress bar
python Scenario1.py -show_progress

# Stochastic environment with custom parameters
python Scenario1.py -stochastic -episodes 2000 -alpha 0.2 -gamma 0.95

# Hyperparameter sensitivity analysis
python Scenario1.py -sensitivity_analysis -episodes 500

# Custom output directory
python Scenario1.py -output_dir custom_results/
```

#### Scenario 2 & 3 Options
```bash
# Enable stochastic transitions (20% random actions)
python Scenario2.py -stochastic
python Scenario3.py -stochastic

# Custom training parameters
python Scenario2.py -episodes 1500 -alpha 0.15 -gamma 0.95
python Scenario3.py -episodes 1000 -alpha 0.3 -gamma 0.98
```

### Command-Line Arguments

| Argument | Description | Default | Available In |
|----------|-------------|---------|--------------|
| `-stochastic` | Enable 20% random action chance | False | All scenarios |
| `-episodes N` | Number of training episodes | 1000 | All scenarios |
| `-alpha A` | Learning rate (0-1) | 0.1 | All scenarios |
| `-gamma G` | Discount factor (0-1) | 0.9 | All scenarios |
| `-seed S` | Random seed for reproducibility | 42 | All scenarios |
| `-show_progress` | Display training progress bar | False | All scenarios |
| `-output_dir DIR` | Output directory for files | results/ | All scenarios |
| `-window_size W` | Moving average window size | 50 | Scenario 1 |
| `-sensitivity` | Run hyperparameter analysis | False | Scenario 1 |

## Output Files and Visualizations

### Scenario 1 Outputs
- `reward_curves.png`: Learning curves comparing exploration strategies
- `steps_curves.png`: Efficiency comparison over training
- `epsilon_decay.png`: Exploration rate decay visualization
- `policy_high.png` & `policy_mod.png`: Policy visualizations with action arrows
- `path_high.png` & `path_mod.png`: Final path demonstrations
- `alpha_sensitivity.png` & `gamma_sensitivity.png`: Parameter sensitivity analysis

### Scenario 2 Outputs
- `reward_curve.png`: Learning progress visualization
- `steps_curve.png`: Efficiency improvement over time
- `epsilon_decay.png`: Exploration rate decay
- `policy_scenario2.png`: Multi-state policy visualization
- `path_scenario2.png`: Final path demonstration

### Scenario 3 Outputs
- `reward_curve.png`: Learning progress with order constraints
- `steps_curve.png`: Efficiency metrics
- `epsilon_decay.png`: Exploration strategy
- `policy_scenario3.png`: Ordered collection policy
- `path_scenario3.png`: Final path attempt

## Algorithm Implementation Details

### Q-Learning Core
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **State Representation**: (x, y, packages_remaining)
- **Action Space**: {Up, Down, Left, Right}
- **Exploration**: ε-greedy with exponential decay

### Key Design Decisions

1. **State Space Encoding**: Efficient indexing with (x-1, y-1) for 0-based arrays
2. **Reward Engineering**: 
   - Scenario 1: +1 for package, -0.01 for movement
   - Scenario 2: +10 for package, -0.01 for movement
   - Scenario 3: +50/+100 for correct order, -200 for violations
3. **Safety Mechanisms**: Step limits prevent infinite loops in complex scenarios
4. **Reproducibility**: Seed control for consistent results

## Performance Benchmarks

### Test Results (Latest Run)
- **Scenario 1**: 100% success rate, optimal performance (0.96 reward, 5 steps)
- **Scenario 2**: Functional performance (18-29 average reward)
- **Scenario 3**: Learning in progress (requires extended training for success)

### Stochastic Environment Performance
- All scenarios handle stochastic transitions effectively
- Performance degrades gracefully with 20% random actions
- Scenario 1 maintains good performance even with stochasticity

## Technical Notes

### Memory Requirements
- Q-tables sized appropriately for state-action space
- Scenario 1: 11×11×2×4 = 968 parameters
- Scenarios 2&3: 11×11×4×4 = 1,936 parameters

### Computational Complexity
- Training time scales linearly with episodes
- Typical runs: 10-15 seconds for 100-200 episodes
- Extended analysis may take several minutes

### Robustness Features
- Error handling for visualization failures
- Progress tracking and early stopping capabilities
- Comprehensive logging and result reporting

## Troubleshooting

### Common Issues
1. **Long training times**: Reduce episodes or enable progress bar
2. **Poor performance**: Adjust learning rate or increase episodes
3. **Visualization errors**: Check output directory permissions

### Optimization Tips
1. **Scenario 3**: Increase episodes to 1000+ for better success rates
2. **Stochastic mode**: Use higher exploration for longer periods
3. **Analysis**: Use sensitivity analysis to find optimal parameters

## Submission Checklist

- [x] All three scenarios implemented and tested
- [x] Comprehensive visualization suite
- [x] Command-line interface with full option support
- [x] Stochastic environment support
- [x] Path visualization and analysis
- [x] Progress tracking and performance metrics
- [x] Error handling and robustness features
- [x] Complete documentation and usage examples

## Academic Context

This implementation demonstrates key reinforcement learning concepts:
- **Exploration vs. Exploitation**: ε-greedy strategies with decay
- **Credit Assignment**: Q-learning temporal difference updates  
- **State Representation**: Effective encoding of complex environments
- **Reward Engineering**: Shaping behavior through incentive design
- **Policy Analysis**: Visualization and interpretation of learned behaviors

The project successfully implements Q-learning across scenarios of increasing complexity, from simple single-objective tasks to complex multi-constraint sequential decision problems.
