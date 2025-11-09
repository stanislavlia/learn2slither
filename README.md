# Learn2Slither - Reinforcement Learning Snake

A reinforcement learning project implementing Q-learning to train an intelligent snake agent that learns to maximize its score through trial and error.

## Project Overview

This project trains a snake agent using **Q-learning**, a value-based reinforcement learning algorithm. The snake learns optimal behavior by interacting with its environment and receiving rewards/penalties for its actions.

## Key Implementation Details

### 1. Environment (Snake Game)

**Board Configuration:**
- 10×10 grid
- 2 green apples (increase length +1, give positive reward)
- 1 red apple (decrease length -1, give negative reward)
- Snake starts with length 3

**Game Rules:**
- Hitting wall → Game Over
- Hitting self → Game Over
- Length reaches 0 → Game Over

### 2. State Representation

The agent has **limited vision** - it only sees in 4 directions (straight, left, right, behind) relative to its head direction.

**State Space:** 12 binary variables
- `danger_straight`, `danger_left`, `danger_right`, `danger_behind` (walls or snake body)
- `green_straight`, `green_left`, `green_right`, `green_behind` (green apples)
- `red_straight`, `red_left`, `red_right`, `red_behind` (red apples)

**Total possible states:** 2^12 = 4,096 states

### 3. Action Space

The agent chooses from **3 relative actions**:
- `straight` - continue in current direction
- `left` - turn 90° left
- `right` - turn 90° right

### 4. Reward System

```python
alive_reward = -2.5      # Small penalty to encourage efficiency
green_reward = +25.0     # Positive reinforcement for good apples
red_reward = -25.0       # Penalty for bad apples
death_reward = -200.0    # Large penalty for dying
```

These rewards guide the agent to:
- Avoid walls and self-collision (death penalty)
- Seek green apples (positive reward)
- Avoid red apples (negative reward)
- Act efficiently (small alive penalty prevents endless loops)

### 5. Q-Learning Algorithm

**Core Concept:** Q-learning learns a Q-table that estimates the expected cumulative reward for taking action `a` in state `s`.

**Q-table Structure:**
```
Q(state, action) → expected reward value
```

**Update Rule:**
```python
Q(s, a) ← Q(s, a) + α[R + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `α` (alpha) = learning rate (default: 0.1)
- `γ` (gamma) = discount factor (default: 0.95) 
- `R` = immediate reward
- `s'` = next state
- `max(Q(s', a'))` = best possible Q-value from next state

**What this means:**
- The agent updates its Q-values based on the immediate reward AND the discounted future value
- Over time, Q-values converge to represent true expected returns

### 6. Exploration vs Exploitation

**Epsilon-Greedy Strategy:**
- With probability `ε`: choose random action (exploration)
- With probability `1-ε`: choose best action from Q-table (exploitation)

**Epsilon Decay:**
```python
epsilon_start = 0.5      # Start with 50% exploration
epsilon_min = 0.05       # Decay to 5% exploration
decay_steps = 50000      # Decay over 50,000 steps
```

Early training → more exploration (learning)  
Later training → more exploitation (using learned policy)

### 7. Training Process

1. **Initialize Q-table** with zeros (or random/optimistic values)
2. **For each training episode:**
   - Reset game
   - While game not over:
     - Observe current state
     - Choose action (ε-greedy)
     - Execute action, receive reward
     - Observe next state
     - Update Q-table using Q-learning formula
     - Decay epsilon
3. **Save model** periodically

### 8. Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Learning rate (α) | 0.1 | How much to update Q-values |
| Discount factor (γ) | 0.95 | Importance of future rewards |
| Epsilon start | 0.5 | Initial exploration rate |
| Epsilon min | 0.05 | Final exploration rate |
| Epsilon decay steps | 50000 | Steps to decay epsilon |
| Alive reward | -2.5 | Per-step living penalty |
| Green apple reward | +25 | Reward for eating good apple |
| Red apple reward | -25 | Penalty for eating bad apple |
| Death reward | -200 | Penalty for game over |

## Usage

### Training a Model

```bash
# Train for 1000 episodes (fast, no visual)
python run.py --sessions 1000 --no-visual --runname my_model

# Train with visualization (slower)
python run.py --sessions 100 --visual --fps 10
```

### Loading and Testing a Model

```bash
# Test trained model in inference mode (no learning)
python run.py --load models/my_model_qtable_final.json --inference --visual --sessions 10

# Step-by-step mode for detailed observation
python run.py --load models/my_model_qtable_final.json --inference --visual --step-by-step
```

### Command-Line Options

```bash
Options:
  --sessions, -s        Number of training episodes (default: 1000)
  --runname, -r         Name for saving files (default: timestamp)
  --load, -l            Path to load pretrained model
  --visual/--no-visual  Enable/disable visualization (default: False)
  --fps                 Visualization speed (default: 10)
  --step-by-step        Advance with spacebar
  --inference           Disable learning, pure exploitation
  --lr                  Learning rate (default: 0.1)
  --discount            Discount factor (default: 0.95)
  --epsilon             Minimum exploration (default: 0.05)
  --epsilon-start       Starting exploration (default: 0.5)
  --init-strategy       Q-table init: zero/positive/random (default: zero)
```

## Project Structure

```
learn2slither/
├── game.py              # Snake game environment
├── agents.py            # Agent classes (RandomAgent, QAgent)
├── q_table.py           # Q-table implementation & Q-learning
├── stats_tracker.py     # Training metrics and plotting
├── run.py               # Main training/inference script
├── requirements.txt     # Python dependencies
├── models/              # Saved Q-tables
└── plots/               # Training progress visualizations
```

## File Format

**Q-table JSON structure:**
```json
{
  "q_table": {
    "[False, True, ...]": {
      "straight": 12.34,
      "left": -5.67,
      "right": 8.90
    }
  },
  "hyperparameters": { ... },
  "metadata": {
    "visited_states": 2048,
    "current_epsilon": 0.05
  }
}
```

## Training Tips

1. **Start with many episodes** (10,000+) for better convergence
2. **Monitor metrics:**
   - Average reward should increase over time
   - Average score should improve
   - Average lifetime should extend
3. **Adjust rewards** if snake isn't learning desired behavior
4. **Use epsilon decay** to balance exploration/exploitation
5. **Save checkpoints** during long training runs

## How Q-Learning Works (Simplified)

1. **Agent observes state** (what it sees around it)
2. **Consults Q-table** for action values
3. **Chooses action** (mostly best, sometimes random)
4. **Receives reward** from environment
5. **Updates Q-table** to reflect new knowledge
6. **Repeat** thousands of times until convergence

The Q-table becomes a "cheat sheet" mapping:
- "In this situation (state), this action (straight/left/right) is worth X points"


## Technical Notes

- **State space is fully enumerable:** 4,096 states fit in memory
- **Q-learning is off-policy:** Learns optimal policy while following ε-greedy
- **Temporal Difference learning:** Updates based on immediate reward + estimated future
- **Model-free:** Agent learns without explicit model of environment dynamics
