from game import SnakeGame, Direction, CLOCKWISE
import pygame
from loguru import logger

def main():

    manual=True

    game = SnakeGame(
        window_height=600,
        window_width=800,
        block_size=20,
        fps=10
    )


    while True:
        move = None
        execute = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            elif event.type == pygame.KEYDOWN:
                execute = True
                if event.key == pygame.K_SPACE:
                    execute = not execute
                if event.key == pygame.K_LEFT:
                    move = Direction.LEFT
                elif event.key == pygame.K_UP:
                    move = Direction.UP
                elif event.key == pygame.K_RIGHT:
                    move = Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    move = Direction.DOWN

        prev_state_with_labels = game.get_state()
        prev_state = [element['value'] for element in prev_state_with_labels]

        # if not manual:
        #     action = agent.get_action(prev_state)
        #     move = game.relative_to_absolute(action)

        (reward, done, score) = game.play_step(
            direction=move,
        )
        if done:
            logger.success(f"GAME OVER!")
            return

        logger.info(f"Move: {move} | Reward: {reward}")

        next_state_with_labels = game.get_state()
        next_state = [element['value'] for element in next_state_with_labels]


if __name__ == "__main__":
    main()