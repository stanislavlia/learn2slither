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
    SNAKE_BODY = (34, 139, 34)  # Dark green for body
    GRID_COLOR = (40, 40, 40)  # grey


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

        if ((window_height % (block_size * 2)) or
                (window_width % (block_size * 2))):
            raise ValueError(
                f"Window height and width should be divisible by "
                f"{block_size * 2}"
            )
        pygame.init()
        pygame.font.init()

        # Panel settings
        self.panel_height = 40

        # Settings
        self.width = window_width
        self.height = window_height
        # Game area excludes panel
        self.game_height = window_height - self.panel_height
        self.block_size = block_size
        self.fps = fps
        self.is_invisible = is_invisible
        self.average_over_n_prev_steps = 10
        self.font_size = 25
        self.font = pygame.font.SysFont('arial', self.font_size)

        # rewards
        self.green_apple_reward = green_apple_reward
        self.red_apple_reward = red_apple_reward
        self.alive_reward = alive_reward
        self.death_reward = death_reward

        # estimate counts of apples
        apple_base = np.mean([window_height, window_width])
        self.green_apple_count = int((apple_base // (block_size * 10)) * 2)
        self.red_apple_count = self.green_apple_count // 2

        self.green_apples = []
        self.red_apples = []

        # pygame-specific
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(
            title="Learn2Slither - RL Agent for SnakeGame"
        )
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.move_history = []
        self.direction = random.choice(list(Direction))

        # Generate head position in game area (below panel)
        x_pos = random.randint(
            0,
            (self.width - self.block_size) // self.block_size
        ) * self.block_size
        y_pos = self.panel_height + random.randint(
            0,
            (self.game_height - self.block_size) // self.block_size
        ) * self.block_size

        self.head = Point(x_pos, y_pos)

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
        """Generate random coordinates in game area (below panel)"""
        x = random.randint(
            0,
            (self.width - self.block_size) // self.block_size
        ) * self.block_size
        y = self.panel_height + random.randint(
            0,
            (self.game_height - self.block_size) // self.block_size
        ) * self.block_size

        return Point(x, y)

    def _generate_green_apples(self):
        while len(self.green_apples) < self.green_apple_count:
            coord = self._get_random_coordinates()
            if (coord in self.snake or
                    coord in self.green_apples or
                    coord in self.red_apples):
                continue

            self.green_apples.append(coord)
            logger.debug(f"Generated GREEN APPLE at {coord}")

    def _generate_red_apples(self):
        while len(self.red_apples) < self.red_apple_count:
            coord = self._get_random_coordinates()
            if (coord in self.snake or
                    coord in self.green_apples or
                    coord in self.red_apples):
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

        # Set head of snake
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

        # Draw score panel FIRST (at top)
        self._draw_score_panel()

        # Draw grid in game area only (below panel)
        grid_color = Colors.GRID_COLOR.value
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(
                self.display,
                grid_color,
                (x, self.panel_height),
                (x, self.height)
            )
        for y in range(self.panel_height, self.height, self.block_size):
            pygame.draw.line(
                self.display,
                grid_color,
                (0, y),
                (self.width, y)
            )

        # Draw separator line between panel and game
        pygame.draw.line(
            self.display,
            Colors.WHITE.value,
            (0, self.panel_height),
            (self.width, self.panel_height),
            2
        )

        # Draw snake body
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display,
                Colors.SNAKE_BODY.value,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
            )

        # Draw snake head (first segment) in a different color
        if len(self.snake) > 0:
            head = self.snake[0]
            pygame.draw.rect(
                self.display,
                Colors.SNAKE_HEAD.value,
                pygame.Rect(head.x, head.y, self.block_size, self.block_size)
            )

        # Draw apples
        for pt in self.green_apples:
            pygame.draw.circle(
                self.display,
                Colors.FT_GREEN.value,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        for pt in self.red_apples:
            pygame.draw.circle(
                self.display,
                Colors.RED.value,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        # Draw stats overlay (if provided)
        if to_display:
            self._draw_stats_overlay(to_display)

        pygame.display.flip()

    def _draw_score_panel(self):
        """Draw a score panel at the top of the screen"""
        # Panel background
        panel_rect = pygame.Rect(0, 0, self.width, self.panel_height)
        pygame.draw.rect(self.display, (50, 50, 50), panel_rect)

        # Draw border
        pygame.draw.rect(self.display, Colors.WHITE.value, panel_rect, 2)

        # Score text
        score_text = f"Score: {self.score}"
        score_surface = self.font.render(
            score_text,
            True,
            Colors.WHITE.value
        )
        self.display.blit(score_surface, (10, 8))

        # Snake length text
        length_text = f"Length: {len(self.snake)}"
        length_surface = self.font.render(
            length_text,
            True,
            Colors.WHITE.value
        )
        self.display.blit(length_surface, (150, 8))

        # Direction text
        direction_text = f"Direction: {self.direction.name}"
        direction_surface = self.font.render(
            direction_text,
            True,
            Colors.WHITE.value
        )
        self.display.blit(direction_surface, (320, 8))

    def _draw_stats_overlay(self, stats_text: list):
        """Draw training stats overlay (top-left corner)"""
        overlay_width = 280
        overlay_height = len(stats_text) * 28 + 20

        # Create more transparent overlay
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(180)  # More transparent
        overlay.fill((30, 30, 40))

        # Position at top-left corner, below panel
        overlay_x = 10  # Left side with small margin
        overlay_y = self.panel_height + 10  # Below panel

        self.display.blit(overlay, (overlay_x, overlay_y))

        # Draw border for better visibility
        border_rect = pygame.Rect(
            overlay_x,
            overlay_y,
            overlay_width,
            overlay_height
        )
        pygame.draw.rect(self.display, (100, 100, 100), border_rect, 1)

        # Draw text
        y = overlay_y + 10
        small_font = pygame.font.SysFont('arial', 18)
        for line in stats_text:
            if line:  # Skip empty lines
                text = small_font.render(line, True, (255, 255, 255))
                self.display.blit(text, (overlay_x + 10, y))
                y += 25

    def _get_movement_std(self):
        """Calculate snake movement variation"""
        if len(self.move_history) < 2:
            return 1

        movement_history = [
            trace["head"] for trace in
            self.move_history[:self.average_over_n_prev_steps]
        ]

        # Compute std + normalize by block size
        x_coords = np.array([p.x for p in movement_history])
        y_coords = np.array([p.y for p in movement_history])
        x_coord_std = np.std(x_coords) / self.block_size
        y_coord_std = np.std(y_coords) / self.block_size

        # Average std of 2 coords
        return (x_coord_std + y_coord_std) / 2

    def _is_there_point(self, from_point, to_points, direction):
        """Check if point exists in given direction"""
        if direction == Direction.RIGHT:
            return any([from_point.x < to_point.x and
                        from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.LEFT:
            return any([from_point.x > to_point.x and
                        from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.UP:
            return any([from_point.y > to_point.y and
                        from_point.x == to_point.x
                        for to_point in to_points])
        elif direction == Direction.DOWN:
            return any([from_point.y < to_point.y and
                        from_point.x == to_point.x
                        for to_point in to_points])

        return False

    def _is_collision(self, point):
        """Check collision with walls or self"""
        if (point.x >= self.width or
                point.x < 0 or
                point.y >= self.height or
                point.y < self.panel_height):
            return True

        # Check self-collision
        if point in self.snake[1:]:
            return True

        return False

    def play_step(self,
                  direction: Direction = None,
                  to_display: list = []):

        self.direction = self._move(
            direction=direction if direction else self.direction
        )
        self.move_history.append({
            "head": self.head,
            "move": self.direction
        })

        # Insert head of snake to beginning
        self.snake.insert(0, self.head)
        self.score = len(self.snake) - 3

        # Estimate Reward
        game_over = False
        reward = None
        if len(self.move_history) > 2:
            reward = self.alive_reward / (self._get_movement_std() ** 3)
        else:
            reward = self.alive_reward

        # Eaten green apple
        if self.head in self.green_apples:
            reward = self.green_apple_reward
            self.green_apples.remove(self.head)
            self._generate_green_apples()

        # Eaten red apple
        elif self.head in self.red_apples:
            reward = self.red_apple_reward
            self.red_apples.remove(self.head)
            self._generate_red_apples()

            # Lose 2 pieces of tail
            if len(self.snake) > 1:
                self.snake.pop()
            if len(self.snake) > 1:
                self.snake.pop()

        else:
            # Lose 1 piece of tail
            self.snake.pop()

        if not self.is_invisible:
            logger.debug("Updating UI...")
            self._update_ui(to_display)
            self.clock.tick(self.fps)

        if len(self.snake) <= 0 or self._is_collision(self.head):
            logger.warning(
                f"GAME OVER | len of snake: {len(self.snake)} | "
                f"_is_collision: {self._is_collision(self.head)}"
            )
            reward = self.death_reward
            game_over = True

        return reward, game_over, self.score

    def get_state(self):
        """Return the current state of the game"""
        head = self.snake[0] if len(self.snake) > 0 else self.head

        # Calculate relative directions
        clock = [Direction.RIGHT, Direction.DOWN,
                 Direction.LEFT, Direction.UP]
        current = clock.index(self.direction)
        straight = clock[current]
        left = clock[(current - 1) % 4]
        right = clock[(current + 1) % 4]
        behind = clock[(current + 2) % 4]

        # Calculate danger in each relative direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        direct_left = Point(head.x - self.block_size, head.y)
        direct_right = Point(head.x + self.block_size, head.y)
        direct_up = Point(head.x, head.y - self.block_size)
        direct_down = Point(head.x, head.y + self.block_size)

        # Danger straight
        danger_straight = (
            (dir_r and self._is_collision(direct_right)) or
            (dir_l and self._is_collision(direct_left)) or
            (dir_u and self._is_collision(direct_up)) or
            (dir_d and self._is_collision(direct_down))
        )

        # Danger left (relative to current direction)
        danger_left = (
            (dir_d and self._is_collision(direct_right)) or
            (dir_u and self._is_collision(direct_left)) or
            (dir_r and self._is_collision(direct_up)) or
            (dir_l and self._is_collision(direct_down))
        )

        # Danger right (relative to current direction)
        danger_right = (
            (dir_u and self._is_collision(direct_right)) or
            (dir_d and self._is_collision(direct_left)) or
            (dir_l and self._is_collision(direct_up)) or
            (dir_r and self._is_collision(direct_down))
        )

        # Danger behind (relative to current direction)
        danger_behind = (
            (dir_l and self._is_collision(direct_right)) or
            (dir_r and self._is_collision(direct_left)) or
            (dir_d and self._is_collision(direct_up)) or
            (dir_u and self._is_collision(direct_down))
        )

        state = [
            # Danger states
            {"label": "danger_straight", "value": danger_straight},
            {"label": "danger_left", "value": danger_left},
            {"label": "danger_right", "value": danger_right},
            {"label": "danger_behind", "value": danger_behind},

            # Green apple states
            {
                "label": "green_straight",
                "value": self._is_there_point(
                    head, self.green_apples, straight
                )
            },
            {
                "label": "green_left",
                "value": self._is_there_point(
                    head, self.green_apples, left
                )
            },
            {
                "label": "green_right",
                "value": self._is_there_point(
                    head, self.green_apples, right
                )
            },
            {
                "label": "green_behind",
                "value": self._is_there_point(
                    head, self.green_apples, behind
                )
            },

            # Red apple states
            {
                "label": "red_straight",
                "value": self._is_there_point(
                    head, self.red_apples, straight
                )
            },
            {
                "label": "red_left",
                "value": self._is_there_point(
                    head, self.red_apples, left
                )
            },
            {
                "label": "red_right",
                "value": self._is_there_point(
                    head, self.red_apples, right
                )
            },
            {
                "label": "red_behind",
                "value": self._is_there_point(
                    head, self.red_apples, behind
                )
            }
        ]

        return state

    def print_state_vision(self, state: list, action: str = None):
        """Print state vision in terminal format (PDF page 8)"""
        # Convert state list to dict
        state_dict = {item['label']: item['value'] for item in state}

        # Map state values to characters
        def get_char(danger, green, red):
            if danger:
                return 'W'
            elif green:
                return 'G'
            elif red:
                return 'R'
            else:
                return '0'

        # Get vision in each direction
        straight = get_char(
            state_dict.get('danger_straight', False),
            state_dict.get('green_straight', False),
            state_dict.get('red_straight', False)
        )

        left = get_char(
            state_dict.get('danger_left', False),
            state_dict.get('green_left', False),
            state_dict.get('red_left', False)
        )

        right = get_char(
            state_dict.get('danger_right', False),
            state_dict.get('green_right', False),
            state_dict.get('red_right', False)
        )

        behind = get_char(
            state_dict.get('danger_behind', False),
            state_dict.get('green_behind', False),
            state_dict.get('red_behind', False)
        )

        # Print in simple format
        print("\n" + "="*50)
        print("SNAKE VISION (Relative to snake's head)")
        print("="*50)
        print(f"STRAIGHT:  {straight}")
        print(f"LEFT:      {left}")
        print("HEAD:      H (reference point)")
        print(f"RIGHT:     {right}")
        print(f"BEHIND:    {behind}")
        print("-"*50)
        print("Legend: W=Wall/Danger, G=Green Apple, "
              "R=Red Apple, 0=Empty, H=Head")

        if action:
            print(f"\nACTION TAKEN: {action.upper()}")

        print("="*50 + "\n")
