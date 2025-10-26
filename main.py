from game import SnakeGame, Direction, CLOCKWISE
import pygame
from loguru import logger
from agents import QAgent
from stats_tracker import TrainingStatsTracker
import os

MANUAL = False
FPS = 300  # Slower so you can see the snake learning
LR = 0.8
DISCOUNT = 0.98
EXPLOR_PROB = 0.35
INIT_STRATEGY = "zero"
EPOCHS = 5000

# REWARDS
ALIVE_REWARD = -0.2
GREEN_APPLE_REWARD = 50
RED_APPLE_REWARD = -15
DEATH_REWARD = -1000

SAVE_EACH_N_EPOCHS = 500


def main():
    CURRENT_EPOCH = 0
    
    # Position the game window on the left side of screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "50,100"
    
    game = SnakeGame(
        window_height=1000,
        window_width=1000,
        block_size=25,
        fps=FPS,
        alive_reward=ALIVE_REWARD,
        green_apple_reward=GREEN_APPLE_REWARD,
        red_apple_reward=RED_APPLE_REWARD,
        death_reward=DEATH_REWARD,
        is_invisible=False  # Show the game window!
    )
    
    agent = QAgent(
        learning_rate=LR,
        discount=DISCOUNT,
        exploration_prob=EXPLOR_PROB,
        init_strategy=INIT_STRATEGY,
        epochs=EPOCHS
    )
    
    # Initialize stats window (updates every epoch)
    stats_window = TrainingStatsTracker(
        window_size=100, 
    )
    
    logger.info("Starting training... Both windows should be visible.")
    
    try:
        while CURRENT_EPOCH < EPOCHS:
            # Handle pygame events for both windows
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Quit event detected")
                    return
            
            # Get current game state
            prev_state_with_labels = game.get_state()
            state_tuple = agent.qtable._state_to_tuple(prev_state_with_labels)
            
            # Agent makes decision
            rel_move = agent.make_move_decision(prev_state_with_labels)
            move = game.relative_to_absolute(rel_move)
            
            # Prepare stats text
            stats_text = [
                f"Epoch: {CURRENT_EPOCH}",
                f"Avg Reward: {stats_window.avg_rewards[-1]:.1f}" if stats_window.avg_rewards else "Avg Reward: N/A",
                f"Avg Score: {stats_window.avg_scores[-1]:.1f}" if stats_window.avg_scores else "Avg Score: N/A",
                f"Avg Life: {stats_window.avg_steps[-1]:.0f}" if stats_window.avg_steps else "Avg Life: N/A",
                f"States: {len(stats_window.visited_states)}"
            ]

            # Execute move with stats overlay
            (reward, done, score) = game.play_step(direction=move, to_display=stats_text)
                        
            # Track step (including state for unique count)
            stats_window.add_step(reward, state_tuple)
            
            # Get next state
            next_state_with_labels = game.get_state()
            
            # Update policy
            agent.update_policy(
                state=prev_state_with_labels,
                reward=reward,
                next_state=next_state_with_labels
            )
            
            if done:
                # End epoch tracking
                stats_window.end_epoch(final_score=score)
                
                game.reset()
                CURRENT_EPOCH += 1
                
                # Save periodically
                if CURRENT_EPOCH % SAVE_EACH_N_EPOCHS == 0 and CURRENT_EPOCH > 0:
                    agent.qtable.save_qtable(filepath=f"./models/qtable_{CURRENT_EPOCH}.json")
                    stats_window.save_metrics(filepath=f"./models/metrics_{CURRENT_EPOCH}.json")
                    logger.success(f"Checkpoint saved at epoch {CURRENT_EPOCH}")
        
        # Final save
        logger.success(f"Training complete! {EPOCHS} epochs finished.")
        agent.qtable.save_qtable(filepath="./models/qtable_final.json")
        stats_window.save_metrics(filepath="./models/metrics_final.json")
        
        # Print final statistics
        stats = stats_window.get_statistics()
        logger.info(f"Final Statistics: {stats}")
        
        input("Press Enter to close...")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        stats_window.close()
        pygame.quit()


if __name__ == "__main__":
    main()