from itertools import product
from typing import List, Dict, Tuple
from loguru import logger
import random
import numpy as np
import json

# State space: 12 binary variables (4 directions × 3 object types)
BINARY_STATE_VARIABLES = [
    'danger_straight', 'danger_left', 'danger_right', 'danger_behind',
    'green_straight', 'green_left', 'green_right', 'green_behind',
    'red_straight', 'red_left', 'red_right', 'red_behind'
]

# Action space: 3 relative actions
ACTION_SPACE = ['straight', 'left', 'right']


class QTable():
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 exploration_prob: float = 0.1,
                 init_strategy: str = "zero",
                 use_epsilon_decay: bool = False,
                 epsilon_start: float = 0.9,
                 epsilon_decay_steps: int = 50000
                 ):

        # Stores Q-values for each state
        # Map: state tuple -> Expected Reward for each Action
        self.q_table = {}
        self.state_visit_stats = {}
        self._init_table(init_strategy)

        # hyperparams
        self.learning_rate = learning_rate
        self.discount = discount
        # This becomes min_epsilon when decay is enabled
        self.exploration_prob = exploration_prob
        self.init_strategy = init_strategy

        # Epsilon decay parameters
        self.use_epsilon_decay = use_epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.current_step = 0  # Track total steps for decay

        if self.use_epsilon_decay:
            logger.info(
                f"Epsilon decay enabled: {epsilon_start:.2f} → "
                f"{exploration_prob:.2f} over {epsilon_decay_steps} steps"
            )

    def _init_table(self, strategy="zero"):
        """Initialize Q-table with given strategy"""
        # Generate all combinations of True/False for 12 binary variables
        # product([False, True], repeat=12) generates all 4096 combinations
        all_states = product(
            [False, True],
            repeat=len(BINARY_STATE_VARIABLES)
        )

        def zero_init():
            return 0

        def optimistic_init():
            return 1

        def positive_init():
            return 10

        def random_init():
            return np.random.randn()

        if strategy == "zero":
            init_qvalue_func = zero_init
        elif strategy == "optimistic":
            init_qvalue_func = optimistic_init
        elif strategy == "positive":
            init_qvalue_func = positive_init
        else:
            init_qvalue_func = random_init

        for state_tuple in all_states:
            self.q_table[state_tuple] = {
                action: init_qvalue_func() for action in ACTION_SPACE
            }
            self.state_visit_stats[state_tuple] = 0

        logger.info(
            f"Q-table preallocated {len(self.q_table)} states "
            f"with {strategy} strategy"
        )

    @staticmethod
    def _state_to_tuple(state: List[Dict]) -> Tuple:
        """Convert state list to tuple for dictionary key"""
        state_values = []
        for var in BINARY_STATE_VARIABLES:
            value = False  # Default to False instead of None
            for state_item in state:
                if state_item['label'] == var:
                    value = bool(state_item['value'])
                    break
            state_values.append(value)

        return tuple(state_values)

    @staticmethod
    def get_epsilon_linear_decay(
            step: int,
            start_epsilon: float = 0.9,
            min_epsilon: float = 0.01,
            decay_steps: int = 50000) -> float:
        """Linear epsilon decay scheduler"""
        if step >= decay_steps:
            return min_epsilon

        # Linear decay
        epsilon = start_epsilon - (start_epsilon - min_epsilon) * (
            step / decay_steps
        )

        return max(epsilon, min_epsilon)

    def get_current_epsilon(self) -> float:
        """Get current epsilon value based on decay settings"""
        if self.use_epsilon_decay:
            return self.get_epsilon_linear_decay(
                step=self.current_step,
                start_epsilon=self.epsilon_start,
                min_epsilon=self.exploration_prob,
                decay_steps=self.epsilon_decay_steps
            )
        else:
            return self.exploration_prob

    def update_q_table(self, state: dict, action: str,
                       reward: float, next_state: dict):
        """Update Q-table using Q-learning update rule"""
        state_tuple = self._state_to_tuple(state)
        next_state_tuple = self._state_to_tuple(next_state)

        max_reward_next_step = max(
            self.q_table[next_state_tuple].values()
        )

        # Q-learning update rule
        current_q = self.q_table[state_tuple][action]
        td_target = reward + self.discount * max_reward_next_step
        self.q_table[state_tuple][action] += (
            self.learning_rate * (td_target - current_q)
        )

        # Update visit stats
        self.state_visit_stats[state_tuple] += 1

    def get_best_action(self, state: List[Dict]) -> str:
        """Get action with highest Q-value for given state"""
        state_tuple = self._state_to_tuple(state)
        q_values = self.q_table[state_tuple]
        return max(q_values, key=q_values.get)

    def get_action(self, state: List[Dict]) -> str:
        """Choose action using epsilon-greedy strategy"""
        epsilon = self.get_current_epsilon()

        # Increment step counter (for decay)
        if self.use_epsilon_decay:
            self.current_step += 1

        if random.random() < epsilon:
            # Explore: random action
            action = random.choice(ACTION_SPACE)
            logger.debug(
                f"Exploring (ε={epsilon:.3f}, "
                f"step={self.current_step}): "
                f"chose random action '{action}'"
            )
            return action
        else:
            # Exploit: best action
            action = self.get_best_action(state)
            logger.debug(
                f"Exploiting (ε={epsilon:.3f}, "
                f"step={self.current_step}): "
                f"chose best action '{action}'"
            )
            return action

    def save_qtable(self, filepath: str):
        """Save Q-table to JSON file"""
        # Convert tuple keys to lists for JSON serialization
        q_table_serializable = {
            str(list(state)): actions
            for state, actions in self.q_table.items()
        }

        state_visits_serializable = {
            str(list(state)): visits
            for state, visits in self.state_visit_stats.items()
        }

        data = {
            'q_table': q_table_serializable,
            'state_visit_stats': state_visits_serializable,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount': self.discount,
                'exploration_prob': self.exploration_prob,
                'init_strategy': self.init_strategy,
                'use_epsilon_decay': self.use_epsilon_decay,
                'epsilon_start': self.epsilon_start,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'current_step': self.current_step
            },
            'metadata': {
                'total_states': len(self.q_table),
                'visited_states': sum(
                    1 for v in self.state_visit_stats.values() if v > 0
                ),
                'total_updates': sum(self.state_visit_stats.values()),
                'current_epsilon': self.get_current_epsilon()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Q-table saved to {filepath}")
        logger.info(f"Total states: {data['metadata']['total_states']}")
        logger.info(
            f"Visited states: {data['metadata']['visited_states']}"
        )
        logger.info(
            f"Total updates: {data['metadata']['total_updates']}"
        )
        logger.info(
            f"Current epsilon: {data['metadata']['current_epsilon']:.4f}"
        )

    @staticmethod
    def load_qtable(filepath: str) -> 'QTable':
        """Load Q-table from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract hyperparameters
        hyperparams = data['hyperparameters']

        # Create new QTable instance using __new__
        qtable = QTable.__new__(QTable)
        qtable.learning_rate = hyperparams['learning_rate']
        qtable.discount = hyperparams['discount']
        qtable.exploration_prob = hyperparams['exploration_prob']
        qtable.init_strategy = hyperparams['init_strategy']

        # Load epsilon decay parameters
        qtable.use_epsilon_decay = hyperparams.get(
            'use_epsilon_decay', False
        )
        qtable.epsilon_start = hyperparams.get('epsilon_start', 0.9)
        qtable.epsilon_decay_steps = hyperparams.get(
            'epsilon_decay_steps', 50000
        )
        qtable.current_step = hyperparams.get('current_step', 0)

        # Load Q-values - convert string keys back to tuples
        qtable.q_table = {}
        for state_str, actions in data['q_table'].items():
            # Parse string representation back to list
            state_str_clean = state_str.strip('[]')
            if state_str_clean:  # Check if not empty
                # Split and convert to booleans
                state_list = []
                for val in state_str_clean.split(','):
                    val = val.strip()
                    if val == 'True':
                        state_list.append(True)
                    elif val == 'False':
                        state_list.append(False)
                state_tuple = tuple(state_list)
                qtable.q_table[state_tuple] = actions

        # Load visit stats
        qtable.state_visit_stats = {}
        for state_str, visits in data['state_visit_stats'].items():
            state_str_clean = state_str.strip('[]')
            if state_str_clean:
                state_list = []
                for val in state_str_clean.split(','):
                    val = val.strip()
                    if val == 'True':
                        state_list.append(True)
                    elif val == 'False':
                        state_list.append(False)
                state_tuple = tuple(state_list)
                qtable.state_visit_stats[state_tuple] = visits

        logger.info(f"Q-table loaded from {filepath}")
        logger.info(f"Total states: {data['metadata']['total_states']}")
        logger.info(
            f"Visited states: {data['metadata']['visited_states']}"
        )
        logger.info(
            f"Total updates: {data['metadata']['total_updates']}"
        )
        if qtable.use_epsilon_decay:
            logger.info(
                f"Epsilon decay: enabled "
                f"(current step: {qtable.current_step}, "
                f"ε={qtable.get_current_epsilon():.4f})"
            )
        else:
            logger.info(
                f"Epsilon decay: disabled "
                f"(fixed ε={qtable.exploration_prob})"
            )

        return qtable
