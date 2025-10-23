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
    SNAKE_HEAD = (144, 238, 144)  # Light green for head
    SNAKE_BODY = (34, 139, 34)    # Dark green for body
    GRID_COLOR = (40, 40 , 40) # grey

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

        #Settings
        self.width = window_width
        self.height = window_height
        self.block_size = block_size
        self.fps = fps
        self.is_invisible = is_invisible
        self.average_over_n_prev_steps = 10


        # rewards
        self.green_apple_reward = green_apple_reward
        self.red_apple_reward = red_apple_reward
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

    def reset(self):

        self.move_history = []
        self.direction = random.choice(list(Direction))
        self.head = Point(
            random.randint(0, (self.width - self.block_size)
                           // self.block_size) * self.block_size,
            random.randint(0, (self.height - self.block_size)
                           // self.block_size) * self.block_size,
        )
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y),
        ]
        self.score = 0
        self.green_apples = []
        self.red_apples = []
        self._generate_green_apples()
        self._generate_red_apples()


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

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size

        #set head of snake
        self.head = Point(x, y)

        return direction

    def relative_to_absolute(self, direction):
        i = CLOCKWISE.index(self.direction)
        choices = [
            i,  # straight
            (i - 1) % 4,  # left
            (i + 1) % 4,  # right
        ]
        return CLOCKWISE[choices[np.argmax(direction)]]


    def _update_ui(self, to_display=[]):

        self.display.fill(Colors.BLACK.value)


        #block grid
        grid_color = Colors.GRID_COLOR.value
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.display, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(self.display, grid_color, (0, y), (self.width, y))

        # Draw snake body
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display, Colors.SNAKE_BODY.value,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
            )
        
        # Draw snake head (first segment) in a different color
        if len(self.snake) > 0:
            head = self.snake[0]
            pygame.draw.rect(
                self.display, Colors.SNAKE_HEAD.value,
                pygame.Rect(head.x, head.y, self.block_size, self.block_size)
            )

        for pt in self.green_apples:
            pygame.draw.circle(
                self.display, Colors.FT_GREEN.value,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        for pt in self.red_apples:
            pygame.draw.circle(
                self.display, Colors.RED.value,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        x, y = self.block_size // 2, self.block_size // 2
        for line in to_display:
            line = self.font.render(line, True, Colors.WHITE.value)
            self.display.blit(line, (x, y))
            y += self.font_size * 1.5

        pygame.display.flip()

    def _get_movement_std(self):
        """Calculates how much snake moving from one dir to another"""
        if len(self.move_history) < 2:
            return 1
        
        movement_history = [trace["head"] for trace in self.move_history[:self.average_over_n_prev_steps]]
        
        #compute std + normalize by block size
        x_coord_std = np.std(np.array([p.x for p in movement_history])) / self.block_size
        y_coord_std = np.std(np.array([p.y for p in movement_history])) / self.block_size

        #average std of 2 coords
        return (x_coord_std + y_coord_std) / 2
    
    def _is_there_point(self, from_point, to_points, direction):
        """ Given a starting point, a list of points and a direction,
            return True if there is a point directly in the given direction
            from the starting point
        """
        if direction == Direction.RIGHT:
            return any([from_point.x < to_point.x
                        and from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.LEFT:
            return any([from_point.x > to_point.x
                        and from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.UP:
            return any([from_point.y > to_point.y
                        and from_point.x == to_point.x
                        for to_point in to_points])
        elif direction == Direction.DOWN:
            return any([from_point.y < to_point.y
                        and from_point.x == to_point.x
                        for to_point in to_points])

        return False

    def _is_collision(self, point):

        if (
            point.x > self.width - self.block_size
            or point.x < 0
            or point.y > self.height - self.block_size
            or point.y < 0
        ):
            return True
        if point in self.snake[1:]:
            return True

        return False

    def play_step(self,
                  direction: Direction = None,
                  to_display: list[str] = []):

        self.direction = self._move(
            direction=direction if direction else self.direction
        )
        self.move_history.append(
            {
                "head" : self.head,
                "move" : self.direction
            }
        )
        
        #insert head of snake to begining
        self.snake.insert(0, self.head)
        self.score = len(self.snake) - 3
        
        #Estimate Reward
        game_over = False
        reward  = None
        if len(self.move_history) > 2:
            reward = self.alive_reward / (self._get_movement_std() ** 3)
        else:
            reward = self.alive_reward
        
        #eaten green apple
        if self.head in self.green_apples:
            reward = self.green_apple_reward
            self.green_apples.remove(self.head) #remove apple after eaten
            self._generate_green_apples()
        
        #eaten red apple
        elif self.head in self.red_apples:
            reward = self.red_apple_reward
            self.red_apples.remove(self.head) #remove apple after eaten
            self._generate_red_apples()

            #lose 2 pieces of tail
            if len(self.snake) > 1:
                self.snake.pop()
            if len(self.snake) > 1:
                self.snake.pop()
        
        else:
            #lose 1 piece of tail
            self.snake.pop()

        if not self.is_invisible:
            logger.debug(f"Updating UI...")
            self._update_ui(to_display)
            self.clock.tick(self.fps)
        
        if len(self.snake) <= 0 or self._is_collision(self.head):
            logger.warning(f"GAME OVER | len of snake: {len(self.snake)} | _is_collision: {self._is_collision(self.head)}")
            reward = self.death_reward
            game_over = True
        
        return reward, game_over, self.score
            

    def get_state(self):
        """ Return the current state of the game
        """
        head = self.snake[0] if len(self.snake) > 0 else self.head
        direct_left = Point(head.x - self.block_size, head.y)
        direct_right = Point(head.x + self.block_size, head.y)
        direct_up = Point(head.x, head.y - self.block_size)
        direct_down = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        current = clock.index(self.direction)
        straight = clock[current]
        left = clock[(current - 1) % 4]
        right = clock[(current + 1) % 4]

        previous = self.move_history[-2]['move'] if len(
            self.move_history) > 1 else self.direction
        previous = clock.index(previous)
        last_move_straight = previous == current
        last_move_left = previous == (current - 1) % 4
        last_move_right = previous == (current + 1) % 4

        state = [
            {
                "label": "move_index",
                "value": self._get_movement_std(),
            },
            {
                "label": "last_move_straight",
                "value": last_move_straight,
            },
            {
                "label": "last_move_left",
                "value": last_move_left,
            },
            {
                "label": "last_move_right",
                "value": last_move_right,
            },
            {
                "label": "danger_straight",
                "value": (dir_r and self._is_collision(direct_right))
                or (dir_l and self._is_collision(direct_left))
                or (dir_u and self._is_collision(direct_up))
                or (dir_d and self._is_collision(direct_down)),
            },
            {
                "label": "danger_left",
                "value": (dir_d and self._is_collision(direct_right))
                or (dir_u and self._is_collision(direct_left))
                or (dir_r and self._is_collision(direct_up))
                or (dir_l and self._is_collision(direct_down)),
            },
            {
                "label": "danger_right",
                "value": (dir_u and self._is_collision(direct_right))
                or (dir_d and self._is_collision(direct_left))
                or (dir_l and self._is_collision(direct_up))
                or (dir_r and self._is_collision(direct_down)),
            },
            {"label": "green_apple_straight",
             "value": self._is_there_point(head, self.green_apples, straight)},
            {"label": "green_apple_left",
             "value": self._is_there_point(head, self.green_apples, left)},
            {"label": "green_apple_right",
             "value": self._is_there_point(head, self.green_apples, right)},
            {"label": "red_apple_straight",
             "value": self._is_there_point(head, self.red_apples, straight)},
            {"label": "red_apple_left",
             "value": self._is_there_point(head, self.red_apples, left)},
            {"label": "red_apple_right",
             "value": self._is_there_point(head, self.red_apples, right)}
        ]

        return state
