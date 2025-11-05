import numpy as np
from typing import List, Dict
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')


class TrainingStatsTracker():
    def __init__(self, window_size: int = 100):
        """
        Track training statistics without any display
        
        Args:
            window_size: Number of epochs for moving average
        """
        self.window_size = window_size
        
        # Track metrics per epoch
        self.epoch_rewards = []
        self.epoch_scores = []
        self.epoch_steps = []
        self.epoch_epsilons = []  # Track epsilon values
        
        # Track metrics per step (for current epoch)
        self.current_epoch_rewards = []
        self.current_epoch_steps = 0
        
        # Moving averages
        self.avg_rewards = []
        self.avg_scores = []
        self.avg_steps = []
        
        # Unique states tracking
        self.visited_states = set()
        
        logger.info(f"TrainingStatsTracker initialized with window_size={window_size}")
    
    def add_step(self, reward: float, state_tuple: tuple = None):
        """
        Add a step's reward to current epoch
        
        Args:
            reward: Reward received this step
            state_tuple: State tuple to track unique states visited
        """
        self.current_epoch_rewards.append(reward)
        self.current_epoch_steps += 1
        
        if state_tuple is not None:
            self.visited_states.add(state_tuple)
    
    def end_epoch(self, final_score: int, epsilon: float = None):
        """
        Mark end of epoch and calculate statistics
        
        Args:
            final_score: Final score achieved in this epoch
            epsilon: Current epsilon value (optional)
        """
        # Calculate epoch statistics
        total_reward = sum(self.current_epoch_rewards)
        avg_reward = np.mean(self.current_epoch_rewards) if self.current_epoch_rewards else 0
        
        # Store epoch data
        self.epoch_rewards.append(total_reward)
        self.epoch_scores.append(final_score)
        self.epoch_steps.append(self.current_epoch_steps)
        
        # Store epsilon if provided
        if epsilon is not None:
            self.epoch_epsilons.append(epsilon)
        
        # Calculate moving averages
        start_idx = max(0, len(self.epoch_rewards) - self.window_size)
        self.avg_rewards.append(np.mean(self.epoch_rewards[start_idx:]))
        self.avg_scores.append(np.mean(self.epoch_scores[start_idx:]))
        self.avg_steps.append(np.mean(self.epoch_steps[start_idx:]))
        
        current_epoch = len(self.epoch_rewards)
        epsilon_str = f", ε={epsilon:.4f}" if epsilon is not None else ""
        logger.info(f"Epoch {current_epoch}: Total Reward={total_reward:.2f}, "
                   f"Avg Reward={avg_reward:.2f}, Score={final_score}, "
                   f"Steps={self.current_epoch_steps}{epsilon_str}")
        
        # Reset for next epoch
        self.current_epoch_rewards = []
        self.current_epoch_steps = 0
    
    def get_overlay_text(self) -> List[str]:
        """
        Get formatted text lines for overlay display
        
        Returns:
            List of strings to display as overlay
        """
        if not self.epoch_rewards:
            return [
                "Training Stats",
                "Waiting for data..."
            ]
        
        avg_reward = self.avg_rewards[-1]
        avg_score = self.avg_scores[-1]
        avg_lifetime = self.avg_steps[-1]
        
        return [
            "Training Stats",
            f"Epoch: {len(self.epoch_rewards)}",
            f"Avg Reward: {avg_reward:.1f}",
            f"Avg Score: {avg_score:.1f}",
            f"Avg Life: {avg_lifetime:.0f} steps",
            f"States: {len(self.visited_states)}",
            "",
            "Best Performance:",
            f"Score: {max(self.epoch_scores)}",
            f"Reward: {max(self.epoch_rewards):.1f}",
            f"Life: {max(self.epoch_steps)} steps"
        ]
    
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
            'avg_survival_recent': self.avg_steps[-1] if self.avg_steps else 0,
            'unique_states': len(self.visited_states)
        }
    
    
    def save_plots(self, filepath: str, title: str = "Training Progress"):
        """
        Save training plots to file
        
        Args:
            filepath: Path to save the plot (e.g., 'plots/training_progress.png')
            title: Title for the plot
        """
        if not self.epoch_rewards:
            logger.warning("No data to plot yet")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine number of subplots (add epsilon plot if we have epsilon data)
        has_epsilon = len(self.epoch_epsilons) > 0
        n_rows = 3 if has_epsilon else 2
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5 * n_rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        epochs = list(range(1, len(self.epoch_rewards) + 1))
        
        # Plot 1: Average Reward over Time
        ax1 = plt.subplot(n_rows, 2, 1)
        ax1.plot(epochs, self.epoch_rewards, alpha=0.3, color='blue', label='Total Reward', linewidth=1)
        ax1.plot(epochs, self.avg_rewards, color='blue', linewidth=2, label=f'Moving Avg ({self.window_size})')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.set_title('Average Reward Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot 2: Average Score over Time
        ax2 = plt.subplot(n_rows, 2, 2)
        ax2.plot(epochs, self.epoch_scores, alpha=0.3, color='green', label='Score', linewidth=1)
        ax2.plot(epochs, self.avg_scores, color='green', linewidth=2, label=f'Moving Avg ({self.window_size})')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Average Score Over Time', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Lifetime (Steps) over Time
        ax3 = plt.subplot(n_rows, 2, 3)
        ax3.plot(epochs, self.epoch_steps, alpha=0.3, color='orange', label='Steps', linewidth=1)
        ax3.plot(epochs, self.avg_steps, color='orange', linewidth=2, label=f'Moving Avg ({self.window_size})')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Steps (Lifetime)', fontsize=12)
        ax3.set_title('Average Lifetime Over Time', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics Summary
        ax4 = plt.subplot(n_rows, 2, 4)
        ax4.axis('off')
        
        # Calculate statistics
        stats_text = f"""
Training Statistics Summary

Total Epochs: {len(self.epoch_rewards):,}
Unique States Visited: {len(self.visited_states):,}

Recent Performance (last {self.window_size} epochs):
  • Avg Reward: {self.avg_rewards[-1]:.2f}
  • Avg Score: {self.avg_scores[-1]:.2f}
  • Avg Lifetime: {self.avg_steps[-1]:.1f} steps

Best Performance:
  • Best Score: {max(self.epoch_scores)}
  • Best Reward: {max(self.epoch_rewards):.2f}
  • Max Lifetime: {max(self.epoch_steps)} steps

Latest Epoch:
  • Score: {self.epoch_scores[-1]}
  • Total Reward: {self.epoch_rewards[-1]:.2f}
  • Steps: {self.epoch_steps[-1]}
        """
        
        if has_epsilon:
            stats_text += f"\n  • Epsilon: {self.epoch_epsilons[-1]:.4f}"
        
        ax4.text(0.1, 0.5, stats_text, 
                fontsize=11, 
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot 5: Epsilon Decay (if available)
        if has_epsilon:
            ax5 = plt.subplot(n_rows, 2, 5)
            epsilon_epochs = list(range(1, len(self.epoch_epsilons) + 1))
            ax5.plot(epsilon_epochs, self.epoch_epsilons, color='purple', linewidth=2, label='Epsilon (ε)')
            ax5.set_xlabel('Epoch', fontsize=12)
            ax5.set_ylabel('Epsilon Value', fontsize=12)
            ax5.set_title('Epsilon Decay Over Time', fontsize=14, fontweight='bold')
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, max(self.epoch_epsilons) * 1.1])
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Training plots saved to {filepath}")
   