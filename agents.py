from loguru import logger
from game import Direction, CLOCKWISE
from typing import List, Dict
import random

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