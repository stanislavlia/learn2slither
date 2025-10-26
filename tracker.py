import matplotlib
matplotlib.use('Qt5Agg')  # or Qt5Agg if tkinter not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from typing import List, Dict
import numpy as np
from loguru import logger
import json
from datetime import datetime


class TrainingTracker():
    def __init__(self, window_size: int = 100, enable_live_plot: bool = True, max_display_points: int = 200):
        """
        Track training metrics and plot progress in real-time using animation
        
        Args:
            window_size: Number of epochs to use for moving average
            enable_live_plot: Whether to show live plotting window
            max_display_points: Maximum number of points to display on plot (for performance)
        """
        self.window_size = window_size
        self.enable_live_plot = enable_live_plot
        self.max_display_points = max_display_points
        
        # Track metrics per epoch
        self.epoch_rewards = []
        self.epoch_scores = []
        self.epoch_steps = []
        
        # Track metrics per step (for current epoch)
        self.current_epoch_rewards = []
        self.current_epoch_steps = 0
        
        # Moving averages
        self.avg_rewards = []
        self.avg_scores = []
        self.avg_steps = []
        
        # Setup live plotting
        self.fig = None
        self.axes = None
        self.lines = None
        self.stats_text = None
        self.ani = None
        
        if self.enable_live_plot:
            self._setup_live_plot()
        
        logger.info(f"TrainingTracker initialized with window_size={window_size}, live_plot={enable_live_plot}")
    
    def _setup_live_plot(self):
        """Setup matplotlib animation for real-time plotting"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Training Progress (Live)', fontsize=16)
        
        # Create subplots
        self.axes = [
            plt.subplot(2, 2, 1),
            plt.subplot(2, 2, 2),
            plt.subplot(2, 2, 3),
            plt.subplot(2, 2, 4)
        ]
        
        # Initialize empty line objects
        self.lines = {
            'reward': self.axes[0].plot([], [], alpha=0.4, color='blue', label='Reward')[0],
            'reward_avg': self.axes[0].plot([], [], color='blue', linewidth=2, label='Moving Avg')[0],
            'score': self.axes[1].plot([], [], alpha=0.4, color='green', label='Score')[0],
            'score_avg': self.axes[1].plot([], [], color='green', linewidth=2, label='Moving Avg')[0],
            'steps': self.axes[2].plot([], [], alpha=0.4, color='orange', label='Steps')[0],
            'steps_avg': self.axes[2].plot([], [], color='orange', linewidth=2, label='Moving Avg')[0],
        }
        
        # Configure axes
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Total Reward')
        self.axes[0].set_title('Total Reward per Epoch')
        self.axes[0].legend(loc='upper left')
        self.axes[0].grid(True, alpha=0.3)
        
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Score')
        self.axes[1].set_title('Score per Epoch')
        self.axes[1].legend(loc='upper left')
        self.axes[1].grid(True, alpha=0.3)
        
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('Steps')
        self.axes[2].set_title('Steps per Epoch (Survival Time)')
        self.axes[2].legend(loc='upper left')
        self.axes[2].grid(True, alpha=0.3)
        
        self.axes[3].axis('off')
        self.stats_text = self.axes[3].text(0.1, 0.5, 'Waiting for data...', 
                                           fontsize=11, 
                                           verticalalignment='center',
                                           family='monospace')
        
        plt.tight_layout()
        
        # Start animation with FuncAnimation
        self.ani = animation.FuncAnimation(
            self.fig, 
            self._animate, 
            interval=100,  # Update every 100ms
            blit=False,  # Set to False for better compatibility
            cache_frame_data=False
        )
        
        plt.show(block=False)
        plt.pause(0.1)
        
        logger.info("Live plot animation started")
    
    def _animate(self, frame):
        """Animation function called by FuncAnimation"""
        if not self.epoch_rewards:
            return []
        
        # Determine which epochs to display (keep last max_display_points)
        start_idx = max(0, len(self.epoch_rewards) - self.max_display_points)
        epochs = list(range(start_idx + 1, len(self.epoch_rewards) + 1))
        
        display_rewards = self.epoch_rewards[start_idx:]
        display_avg_rewards = self.avg_rewards[start_idx:]
        display_scores = self.epoch_scores[start_idx:]
        display_avg_scores = self.avg_scores[start_idx:]
        display_steps = self.epoch_steps[start_idx:]
        display_avg_steps = self.avg_steps[start_idx:]
        
        # Update reward plot
        self.lines['reward'].set_data(epochs, display_rewards)
        self.lines['reward_avg'].set_data(epochs, display_avg_rewards)
        self.axes[0].relim()
        self.axes[0].autoscale_view()
        
        # Update score plot
        self.lines['score'].set_data(epochs, display_scores)
        self.lines['score_avg'].set_data(epochs, display_avg_scores)
        self.axes[1].relim()
        self.axes[1].autoscale_view()
        
        # Update steps plot
        self.lines['steps'].set_data(epochs, display_steps)
        self.lines['steps_avg'].set_data(epochs, display_avg_steps)
        self.axes[2].relim()
        self.axes[2].autoscale_view()
        
        # Update statistics text
        if self.epoch_rewards:
            stats_text = f"""
Training Statistics:

Total Epochs: {len(self.epoch_rewards)}

Best Score: {max(self.epoch_scores)}
Avg Score (last {self.window_size}): {self.avg_scores[-1]:.2f}

Best Reward: {max(self.epoch_rewards):.2f}
Avg Reward (last {self.window_size}): {self.avg_rewards[-1]:.2f}

Max Survival: {max(self.epoch_steps)} steps
Avg Survival (last {self.window_size}): {self.avg_steps[-1]:.2f} steps

Latest Epoch:
  - Score: {self.epoch_scores[-1]}
  - Total Reward: {self.epoch_rewards[-1]:.2f}
  - Steps: {self.epoch_steps[-1]}
            """
            self.stats_text.set_text(stats_text)
        
        return list(self.lines.values()) + [self.stats_text]
    
    def add_step(self, reward: float):
        """Add a step's reward to current epoch"""
        self.current_epoch_rewards.append(reward)
        self.current_epoch_steps += 1
    
    def end_epoch(self, final_score: int):
        """
        Mark end of epoch and calculate statistics
        
        Args:
            final_score: Final score achieved in this epoch
        """
        # Calculate epoch statistics
        total_reward = sum(self.current_epoch_rewards)
        avg_reward = np.mean(self.current_epoch_rewards) if self.current_epoch_rewards else 0
        
        # Store epoch data
        self.epoch_rewards.append(total_reward)
        self.epoch_scores.append(final_score)
        self.epoch_steps.append(self.current_epoch_steps)
        
        # Calculate moving averages
        start_idx = max(0, len(self.epoch_rewards) - self.window_size)
        self.avg_rewards.append(np.mean(self.epoch_rewards[start_idx:]))
        self.avg_scores.append(np.mean(self.epoch_scores[start_idx:]))
        self.avg_steps.append(np.mean(self.epoch_steps[start_idx:]))
        
        current_epoch = len(self.epoch_rewards)
        logger.info(f"Epoch {current_epoch}: Total Reward={total_reward:.2f}, "
                   f"Avg Reward={avg_reward:.2f}, Score={final_score}, "
                   f"Steps={self.current_epoch_steps}")
        
        # Reset for next epoch
        self.current_epoch_rewards = []
        self.current_epoch_steps = 0
    
    def save_plot(self, filepath: str):
        """Save current plot to file"""
        if self.enable_live_plot and self.fig is not None:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
    
    def close_plot(self):
        """Close the live plot window"""
        if self.enable_live_plot and self.fig is not None:
            if self.ani is not None:
                self.ani.event_source.stop()
            plt.close(self.fig)
    
    def save_metrics(self, filepath: str):
        """Save training metrics to JSON"""
        data = {
            'epoch_rewards': self.epoch_rewards,
            'epoch_scores': self.epoch_scores,
            'epoch_steps': self.epoch_steps,
            'avg_rewards': self.avg_rewards,
            'avg_scores': self.avg_scores,
            'avg_steps': self.avg_steps,
            'window_size': self.window_size,
            'total_epochs': len(self.epoch_rewards),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training metrics saved to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get current training statistics"""
        if not self.epoch_rewards:
            return {}
        
        return {
            'total_epochs': len(self.epoch_rewards),
            'best_score': max(self.epoch_scores),
            'best_reward': max(self.epoch_rewards),
            'avg_score_recent': self.avg_scores[-1] if self.avg_scores else 0,
            'avg_reward_recent': self.avg_rewards[-1] if self.avg_rewards else 0,
            'max_survival': max(self.epoch_steps),
            'avg_survival_recent': self.avg_steps[-1] if self.avg_steps else 0
        }