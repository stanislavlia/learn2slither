import numpy as np
from loguru import logger
from itertools import product
from typing import List, Dict, Optional, Tuple
import random
import json

BINARY_STATE_VARIABLES = [
    'last_move_straight',
    'last_move_left',
    'last_move_right',
    'danger_straight',
    'danger_left',
    'danger_right',
    'green_apple_straight',
    'green_apple_left',
    'green_apple_right',
    'red_apple_straight',
    'red_apple_left',
    'red_apple_right',
]

ACTION_SPACE = [
    "left", "straight", "right"
]

class QTable():
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 exploration_prob: float = 0.1,
                 init_strategy: str = "zero"
                 ):

        # stores Q-values for each state Map state tuple (1, 0, ... 1, 1) -> Expected Reward for each Action
        self.q_table = {}
        self.state_visit_stats = {}
        self._init_table(init_strategy)

        #hyperparams
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.init_strategy = init_strategy

    def _init_table(self, strategy="zero"):  #strategy can be 'zero', 'optimistic', 'random'
        
        
        # Generate all combinations of True/False for 12 binary variables
        # product([False, True], repeat=12) generates all 4096 combinations
        all_states = product([False, True], repeat=len(BINARY_STATE_VARIABLES))

        
        if strategy == "zero":
            init_qvalue_func = lambda : 0

        elif strategy == "optimistic":
            init_qvalue_func = lambda : 1  #positive value
        
        else:
            init_qvalue_func = lambda : np.random.randn()

        
        for state_tuple in all_states:
            self.q_table[state_tuple] = {action: init_qvalue_func() for action in ACTION_SPACE}
            self.state_visit_stats[state_tuple] = 0
        
        logger.info(f"Q-table preallocated {len(self.q_table)} states with {strategy} strategy")

    @staticmethod
    def _state_to_tuple(state: List[Dict]) -> Tuple:
        """
        Convert state list to tuple for use as dictionary key
        """
        state_values = []
        for var in BINARY_STATE_VARIABLES:
            value = None
            for state_item in state:
                if state_item['label'] == var:
                    value = bool(state_item['value'])
                    break
            state_values.append(value)
        
        return tuple(state_values)
    
    def update_q_table(self, state: dict, action: str, reward: float, next_state: dict):
        
        state_tuple = self._state_to_tuple(state)
        next_state_tuple = self._state_to_tuple(next_state)

        max_reward_next_step = max(self.q_table[next_state_tuple].values())

        #update rule
        self.q_table[state_tuple][action] += self.learning_rate * \
        (reward + self.discount * max_reward_next_step - self.q_table[state_tuple][action])

        #update visit stats
        self.state_visit_stats[state_tuple] += 1


    def get_best_action(self, state: List[Dict]) -> str:
        state_tuple = self._state_to_tuple(state)
        q_values = self.q_table[state_tuple]
        return max(q_values, key=q_values.get)
    
    def get_action(self, state: List[Dict]) -> str:
        """
        Choose action using epsilon-greedy strategy
        """

        if random.random() < self.exploration_prob:
            # Explore: random action
            action = random.choice(ACTION_SPACE)
            logger.debug(f"Exploring: chose random action '{action}'")
            return action
        else:
            # Exploit: best action
            action = self.get_best_action(state)
            logger.debug(f"Exploiting: chose best action '{action}'")
            return action
        
    def save_qtable(self, filepath: str):
        """
        Save Q-table to JSON file
        """
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
                'init_strategy': self.init_strategy
            },
            'metadata': {
                'total_states': len(self.q_table),
                'visited_states': sum(1 for v in self.state_visit_stats.values() if v > 0),
                'total_updates': sum(self.state_visit_stats.values())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Q-table saved to {filepath}")
        logger.info(f"Total states: {data['metadata']['total_states']}")
        logger.info(f"Visited states: {data['metadata']['visited_states']}")
        logger.info(f"Total updates: {data['metadata']['total_updates']}")

    @staticmethod
    def load_qtable(filepath: str) -> 'QTable':
        """
        Load Q-table from JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract hyperparameters
        hyperparams = data['hyperparameters']
        
        # Create new QTable instance with empty tables
        qtable = QTable.__new__(QTable)
        qtable.learning_rate = hyperparams['learning_rate']
        qtable.discount = hyperparams['discount']
        qtable.exploration_prob = hyperparams['exploration_prob']
        qtable.init_strategy = hyperparams['init_strategy']
        
        # Load Q-values - convert string keys back to tuples
        qtable.q_table = {}
        for state_str, actions in data['q_table'].items():
            # JSON loads the list from the string representation
            state_list = json.loads(state_str)
            state_tuple = tuple(state_list)
            qtable.q_table[state_tuple] = actions
        
        # Load visit stats
        qtable.state_visit_stats = {}
        for state_str, visits in data['state_visit_stats'].items():
            state_list = json.loads(state_str)
            state_tuple = tuple(state_list)
            qtable.state_visit_stats[state_tuple] = visits
        
        logger.info(f"Q-table loaded from {filepath}")
        logger.info(f"Total states: {data['metadata']['total_states']}")
        logger.info(f"Visited states: {data['metadata']['visited_states']}")
        logger.info(f"Total updates: {data['metadata']['total_updates']}")
        
        return qtable
