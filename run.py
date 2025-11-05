from game import SnakeGame, Direction, CLOCKWISE
import pygame
from loguru import logger
from agents import QAgent
from stats_tracker import TrainingStatsTracker
import sys
import os
import click
from datetime import datetime


@click.command()
@click.option('--sessions', '-s', default=1000, type=int, 
              help='Number of training sessions/episodes')
@click.option('--runname', '-r', default=None, type=str,
              help='Run name for saving files (default: timestamp)')
@click.option('--load', '-l', default=None, type=str,
              help='Path to load a pretrained model')
@click.option('--save-interval', default=1000, type=int,
              help='Save checkpoint every N epochs')
@click.option('--visual/--no-visual', default=False,
              help='Enable/disable visual display')
@click.option('--fps', default=10, type=int,
              help='Frames per second for visualization')
@click.option('--step-by-step', is_flag=True,
              help='Step-by-step mode (press space to advance)')
@click.option('--inference', is_flag=True,
              help='Inference mode: disable learning (pure exploitation)')
@click.option('--lr', default=0.1, type=float,
              help='Learning rate')
@click.option('--discount', default=0.95, type=float,
              help='Discount factor')
@click.option('--epsilon', default=0.05, type=float,
              help='Minimum exploration probability')
@click.option('--epsilon-start', default=0.5, type=float,
              help='Starting exploration probability')
@click.option('--epsilon-decay/--no-epsilon-decay', default=True,
              help='Enable/disable epsilon decay')
@click.option('--epsilon-decay-epochs', default=50000, type=int,
              help='Number of epochs for epsilon decay')
@click.option('--init-strategy', type=click.Choice(['zero', 'positive', 'random']),
              default='zero', help='Q-table initialization strategy')
@click.option('--alive-reward', default=-2.5, type=float,
              help='Reward for staying alive')
@click.option('--green-reward', default=25.0, type=float,
              help='Reward for eating green apple')
@click.option('--red-reward', default=-25.0, type=float,
              help='Reward for eating red apple')
@click.option('--death-reward', default=-200.0, type=float,
              help='Reward for dying')
@click.option('--window-height', default=800, type=int,
              help='Game window height')
@click.option('--window-width', default=800, type=int,
              help='Game window width')
@click.option('--block-size', default=40, type=int,
              help='Size of each block')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
def main(sessions, runname, load, save_interval, visual, fps, step_by_step,
         inference, lr, discount, epsilon, epsilon_start, epsilon_decay,
         epsilon_decay_epochs, init_strategy, alive_reward, green_reward,
         red_reward, death_reward, window_height, window_width, block_size,
         log_level):
    
    # Generate run name if not provided
    if runname is None:
        if inference:
            runname = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            runname = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Display banner
    logger.info("=" * 70)
    logger.info("Learn2Slither - Reinforcement Learning Snake")
    logger.info("=" * 70)
    logger.info(f"Run Name: {runname}")
    logger.info(f"Mode: {'INFERENCE (Testing)' if inference else 'TRAINING'}")
    logger.info(f"Sessions: {sessions}")
    logger.info(f"Visual Display: {'Enabled' if visual else 'Disabled'}")
    if load:
        logger.info(f"Loading Model: {load}")
    if not inference:
        logger.info(f"Learning Rate: {lr}")
        logger.info(f"Discount Factor: {discount}")
        logger.info(f"Epsilon: {epsilon_start} → {epsilon} (Decay: {epsilon_decay})")
        logger.info(f"Init Strategy: {init_strategy}")
    logger.info("=" * 70)
    
    # Determine FPS and UI visibility
    show_ui = visual
    actual_fps = fps if show_ui else 0  # Max speed if no visual
    
    # Initialize game
    game = SnakeGame(
        window_height=window_height,
        window_width=window_width,
        block_size=block_size,
        fps=actual_fps,
        alive_reward=alive_reward,
        green_apple_reward=green_reward,
        red_apple_reward=red_reward,
        death_reward=death_reward,
        is_invisible=not show_ui
    )
    
    # Initialize agent
    agent = QAgent(
        learning_rate=lr,
        discount=discount,
        exploration_prob=epsilon if not inference else 0.0,
        init_strategy=init_strategy,
        epochs=sessions,
        use_epsilon_decay=epsilon_decay and not inference,
        epsilon_start=epsilon_start,
        epsilon_decay_steps=epsilon_decay_epochs
    )
    
    # Load model if specified
    if load:
        try:
            agent.load_from_file(load)
            logger.success(f"Model loaded from {load}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {load}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    # Inference mode: disable learning
    if inference:
        agent.qtable.exploration_prob = 0.0
        agent.qtable.use_epsilon_decay = False
        logger.info("Inference mode enabled - Pure exploitation (ε=0)")
    
    # Initialize stats tracker
    stats_tracker = TrainingStatsTracker(window_size=100)
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    CURRENT_EPOCH = 0
    
    try:
        while CURRENT_EPOCH < sessions:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Quit event detected")
                    return
                elif step_by_step and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        logger.info("Escape pressed - exiting")
                        return
            
            # Step-by-step mode: wait for spacebar
            if step_by_step and show_ui:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
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
                f"Run: {runname}",
                f"Epoch: {CURRENT_EPOCH}/{sessions}",
                f"Epsilon: {current_epsilon:.3f}",
                f"Mode: {'INFERENCE' if inference else 'TRAINING'}",
                f"Avg Reward: {stats_tracker.avg_rewards[-1]:.1f}" if stats_tracker.avg_rewards else "Avg Reward: N/A",
                f"Avg Score: {stats_tracker.avg_scores[-1]:.1f}" if stats_tracker.avg_scores else "Avg Score: N/A",
                f"Avg Life: {stats_tracker.avg_steps[-1]:.0f}" if stats_tracker.avg_steps else "Avg Life: N/A",
                f"States: {len(stats_tracker.visited_states)}"
            ]
            
            # Execute move
            (reward, done, score) = game.play_step(
                direction=move, 
                to_display=stats_text if show_ui else None
            )
            
            # Track step
            stats_tracker.add_step(reward, state_tuple)
            
            # Get next state
            next_state_with_labels = game.get_state()
            
            # Update policy (only if NOT in inference mode)
            if not inference:
                agent.update_policy(
                    state=prev_state_with_labels,
                    reward=reward,
                    next_state=next_state_with_labels
                )
            
            if done:
                # End epoch tracking
                current_epsilon = agent.get_current_epsilon()
                stats_tracker.end_epoch(final_score=score, epsilon=current_epsilon)
                game.reset()
                CURRENT_EPOCH += 1
                
                # Save periodically (only during training)
                if not inference and CURRENT_EPOCH % save_interval == 0 and CURRENT_EPOCH > 0:
                    model_path = f"./models/{runname}_qtable_{CURRENT_EPOCH}.json"
                    metrics_path = f"./models/{runname}_metrics_{CURRENT_EPOCH}.json"
                    plot_path = f"./plots/{runname}_training_{CURRENT_EPOCH}.png"
                    
                    agent.qtable.save_qtable(filepath=model_path)
                    stats_tracker.save_metrics(filepath=metrics_path)
                    stats_tracker.save_plots(filepath=plot_path)
                    logger.success(f"Checkpoint saved at epoch {CURRENT_EPOCH}")
        
        # Final save
        if not inference:
            # Training mode: save model and plots
            final_model_path = f"./models/{runname}_qtable_final.json"
            final_plot_path = f"./plots/{runname}_training_final.png"
            
            agent.qtable.save_qtable(filepath=final_model_path)
            stats_tracker.save_plots(filepath=final_plot_path, title=f"Final Training Results - {runname}")
            
            logger.success(f"Training complete! {sessions} epochs finished.")
            logger.success(f"Final model saved to: {final_model_path}")
        else:
            # Inference mode: only save evaluation results
            eval_results_path = f"./models/{runname}_evaluation_results.json"
            stats_tracker.save_metrics(filepath=eval_results_path)
            
            logger.success(f"Inference complete! {sessions} episodes finished.")
            logger.success(f"Evaluation results saved to: {eval_results_path}")
        
        # Print final statistics
        stats = stats_tracker.get_statistics()
        logger.info("=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        if not inference:
            # Save on interruption (only in training mode)
            interrupt_path = f"./models/{runname}_qtable_interrupted.json"
            agent.qtable.save_qtable(filepath=interrupt_path)
            logger.info(f"Progress saved to: {interrupt_path}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()