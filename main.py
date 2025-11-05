from game import SnakeGame, Direction, CLOCKWISE
import pygame
from loguru import logger
from agents import QAgent
from stats_tracker import TrainingStatsTracker
import sys
import os

#==============SETTINGS=============
MANUAL = False
FPS = 10000
LR = 0.1
DISCOUNT = 0.95
EXPLOR_PROB = 0.05 # This becomes min epsilon when decay is enabled
INIT_STRATEGY = "zero"
EPOCHS = 500000
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
BLOCK_SIZE = 40
SHOW_UI = False
LOG_LEVEL = "INFO"

# Epsilon decay settings
USE_EPSILON_DECAY =True
EPSILON_START = 0.5
EPSILON_DECAY_STEPS = 50000 * 320

# Checkpoint settings
CHECKPOINT_TO_LOAD = "models/qtable_final.json"
LOAD_CHECKPOINT = False

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

# REWARDS
ALIVE_REWARD = -2.5
GREEN_APPLE_REWARD = 25
RED_APPLE_REWARD = -25
DEATH_REWARD = -200
SAVE_EACH_N_EPOCHS = 50_000


def main():
    CURRENT_EPOCH = 0
    
    game = SnakeGame(
        window_height=WINDOW_HEIGHT,
        window_width=WINDOW_WIDTH,
        block_size=BLOCK_SIZE,
        fps=FPS,
        alive_reward=ALIVE_REWARD,
        green_apple_reward=GREEN_APPLE_REWARD,
        red_apple_reward=RED_APPLE_REWARD,
        death_reward=DEATH_REWARD,
        is_invisible=not SHOW_UI
    )
    
    agent = QAgent(
        learning_rate=LR,
        discount=DISCOUNT,
        exploration_prob=EXPLOR_PROB,
        init_strategy=INIT_STRATEGY,
        epochs=EPOCHS,
        use_epsilon_decay=USE_EPSILON_DECAY,
        epsilon_start=EPSILON_START,
        epsilon_decay_steps=EPSILON_DECAY_STEPS
    )
    
    if LOAD_CHECKPOINT:
        agent.load_from_file(CHECKPOINT_TO_LOAD)

    # Initialize stats tracker
    stats_tracker = TrainingStatsTracker(window_size=100)
    
    logger.info("Starting training...")
    if USE_EPSILON_DECAY:
        logger.info(f"Epsilon decay: {EPSILON_START} → {EXPLOR_PROB} over {EPSILON_DECAY_STEPS} steps")
    else:
        logger.info(f"Fixed epsilon: {EXPLOR_PROB}")
    
    try:
        while CURRENT_EPOCH < EPOCHS:
            # Handle pygame events
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
            
            # Get current epsilon for display
            current_epsilon = agent.get_current_epsilon()
            
            # Prepare stats text
            stats_text = [
                f"Epoch: {CURRENT_EPOCH}",
                f"Epsilon: {current_epsilon:.3f}",
                f"Step: {agent.qtable.current_step}" if USE_EPSILON_DECAY else "",
                f"Avg Reward: {stats_tracker.avg_rewards[-1]:.1f}" if stats_tracker.avg_rewards else "Avg Reward: N/A",
                f"Avg Score: {stats_tracker.avg_scores[-1]:.1f}" if stats_tracker.avg_scores else "Avg Score: N/A",
                f"Avg Life: {stats_tracker.avg_steps[-1]:.0f}" if stats_tracker.avg_steps else "Avg Life: N/A",
                f"States: {len(stats_tracker.visited_states)}"
            ]

            # Execute move with stats overlay
            (reward, done, score) = game.play_step(direction=move, to_display=stats_text)
                        
            # Track step
            stats_tracker.add_step(reward, state_tuple)
            
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
                current_epsilon = agent.get_current_epsilon()
                stats_tracker.end_epoch(final_score=score,
                                        epsilon=current_epsilon)
                
                game.reset()
                CURRENT_EPOCH += 1
                
                # Save periodically
                if CURRENT_EPOCH % SAVE_EACH_N_EPOCHS == 0 and CURRENT_EPOCH > 0:
                    agent.qtable.save_qtable(filepath=f"./models/qtable_{CURRENT_EPOCH}.json")
                    stats_tracker.save_metrics(filepath=f"./models/metrics_{CURRENT_EPOCH}.json")
                    stats_tracker.save_plots(filepath=f"./plots/training_{CURRENT_EPOCH}.png")
                    logger.success(f"Checkpoint saved at epoch {CURRENT_EPOCH}")
        
        # Final save
        logger.success(f"Training complete! {EPOCHS} epochs finished.")
        agent.qtable.save_qtable(filepath="./models/qtable_final.json")
        stats_tracker.save_metrics(filepath="./models/metrics_final.json")
        stats_tracker.save_plots(filepath="./plots/training_final.png", title="Final Training Results")
        stats_tracker.save_detailed_plots(filepath="./plots/detailed_final.png", title="Final Detailed Analysis")
        
        # Print final statistics
        stats = stats_tracker.get_statistics()
        logger.info(f"Final Statistics: {stats}")
        
        input("Press Enter to close...")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
