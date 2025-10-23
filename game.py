import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from loguru import logger

# ============DATA STRUCTURES================
class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    FT_RED = (232, 33, 39)
    FT_BLUE = (0, 186, 188)
    FT_GREEN = (15, 218, 83)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


CLOCKWISE = [Direction.RIGHT,
             Direction.DOWN,
             Direction.LEFT,
             Direction.UP]

Point = namedtuple("Point", "x, y")



# ==============GAME Implementation============
class SnakeGame():
    def __init__(self,
                 window_width: int = 800,
                 window_height: int = 600,
                 block_size: int = 20,
                 fps: int = 0,
                 green_apple_reward: float = 25.0,
                 red_apple_reward: float = -30.0,
                 alive_reward: float = -1.0,
                 death_reward: float = -1000,
                 is_invisible: bool = False
                 ):
        
        if (window_height % (block_size * 2)) or (window_width % (block_size * 2)):
            raise ValueError(f"Window height and width should be divsible by {block_size * 2}")


        self.width = window_width
        self.height = window_height
        self.block_size = block_size
        self.fps = fps
        self.is_invisible = is_invisible
        
        # rewards
        self.green_apple_reward = green_apple_reward
        self.green_apple_reward = red_apple_reward
        self.alive_reward = alive_reward
        self.death_reward = death_reward

        # estimate counts of apples
        self.green_apple_count = (np.mean([window_height, window_width]) // (block_size * 10)) * 2
        self.red_apple_count  = self.green_apple_count // 2

        self.green_apples = []
        self.red_apples = []
        
        #pygame-specific
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title="Learn2Slither - RL Agent for SnakeGame")
        self.clock = pygame.time.Clock()
        self.reset()


        self.snake = []


    def _get_random_coordinates(self):

        x = random.randint(0, (self.width - self.block_size)
                               // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size)
                               // self.block_size) * self.block_size
        
        return Point(x, y)

    def _generate_green_apples(self):
        
        while len(self.green_apples) < self.green_apple_count:
            coord = self._get_random_coordinates()
            if (coord in self.snake
                or coord in self.green_apples
                or coord in self.green_apples):
                continue
            
            self.green_apples.append(coord)
            logger.debug(f"Generated GREEN APPLE at {coord}")
    
    def _generate_red_apples(self):
        
        while len(self.red_apples) < self.red_apple_count:
            coord = self._get_random_coordinates()
            if (coord in self.snake
                or coord in self.green_apples
                or coord in self.green_apples):
                continue
            
            self.red_apples.append(coord)
            logger.debug(f"Generated RED APPLE at {coord}")