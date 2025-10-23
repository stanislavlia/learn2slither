from game import SnakeGame, Direction, CLOCKWISE
import pygame
from loguru import logger
from agents import RandomAgent

MANUAL = False  # Set to False to use agent
RANDOM = True   # Set to True to use RandomAgent
FPS = 30

def main():
    game = SnakeGame(
        window_height=600,
        window_width=800,
        block_size=20,
        fps=FPS
    )
    
    agent = RandomAgent()
    
    while True:
        move = None
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                # Manual controls override agent
                if event.key == pygame.K_LEFT:
                    move = Direction.LEFT
                elif event.key == pygame.K_UP:
                    move = Direction.UP
                elif event.key == pygame.K_RIGHT:
                    move = Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    move = Direction.DOWN
        
        # Get current game state
        prev_state_with_labels = game.get_state()
        prev_state = [element['value'] for element in prev_state_with_labels]
        
        # Agent makes decision if no manual input
        if move is None and RANDOM and not MANUAL:
            rel_move = agent.make_move_decision(prev_state_with_labels)
            move = game.relative_to_absolute(rel_move)
        
        # Execute move
        (reward, done, score) = game.play_step(direction=move)
        
        if done:
            logger.success(f"GAME OVER! Final Score: {score}")
            game.reset()
            return
        
        logger.info(f"Move: {move.name if move else 'None'} | Reward: {reward:.2f} | Score: {score}")
        
        # Get next state
        next_state_with_labels = game.get_state()
        next_state = [element['value'] for element in next_state_with_labels]

if __name__ == "__main__":
    main()