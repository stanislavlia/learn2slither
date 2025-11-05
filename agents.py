from loguru import logger
from game import Direction, CLOCKWISE
from typing import List, Dict
import random
from q_table import QTable


class AgentBase():
    def __init__(self):
        pass
    
    def make_move_decision(self, game_state: List[Dict]):
        raise NotImplementedError(f"Make move decision is not implemented")


class RandomAgent(AgentBase):
    def make_move_decision(self, game_state: List[Dict]):
        """
        Makes a random decision: go straight, turn left, or turn right
        Returns a one-hot encoded relative action [straight, left, right]
        """
        # Random choice: straight, left, or right
        rel_move = [0, 0, 0]
        choice = random.randint(0, 2)
        rel_move[choice] = 1
        action_name = ["STRAIGHT", "LEFT", "RIGHT"][choice]
        logger.debug(f"RandomAgent chose: {action_name}")
        return rel_move


class QAgent(AgentBase):
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 exploration_prob: float = 0.1,
                 init_strategy: str = "zero",
                 epochs: int = 1000,
                 use_epsilon_decay: bool = False,
                 epsilon_start: float = 0.9,
                 epsilon_decay_steps: int = 50000):
        self.epochs = epochs
        self.qtable = QTable(
            learning_rate=learning_rate,
            discount=discount,
            exploration_prob=exploration_prob,
            init_strategy=init_strategy,
            use_epsilon_decay=use_epsilon_decay,
            epsilon_start=epsilon_start,
            epsilon_decay_steps=epsilon_decay_steps
        )
        self.current_epoch = 0
        self.last_action_name = None  # Store the last action name for updates
    
    def make_move_decision(self, game_state: List[Dict]) -> List[int]:
        """
        Makes a decision using Q-table with epsilon-greedy strategy
        Returns a one-hot encoded relative action [straight, left, right]
        """
        # Get action from Q-table (uses epsilon-greedy internally with decay)
        action_name = self.qtable.get_action(game_state)
        
        # Store the action name for later update
        self.last_action_name = action_name
        
        # Convert action name to one-hot encoded array
        action_map = {
            "straight": [1, 0, 0],
            "left": [0, 1, 0],
            "right": [0, 0, 1]
        }
        
        rel_move = action_map[action_name]
        
        logger.debug(f"QAgent chose: {action_name.upper()}")
        
        return rel_move
    
    def update_policy(self, state: List[Dict], reward: float, next_state: List[Dict]):
        """
        Update Q-table based on experience
        Uses the last action taken (stored in self.last_action_name)
        """
        if self.last_action_name is None:
            logger.warning("No action to update - make_move_decision must be called first")
            return
        
        self.qtable.update_q_table(
            state=state,
            action=self.last_action_name,
            reward=reward,
            next_state=next_state
        )
    
    def load_from_file(self, path: str):
        """Load Q-table from file"""
        self.qtable = QTable.load_qtable(path)
    
    def get_current_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.qtable.get_current_epsilon()