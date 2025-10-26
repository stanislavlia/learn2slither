import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
import json
from datetime import datetime


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
    
    def save_metrics(self, filepath: str):
        """Save training metrics to JSON"""
        data = {
            'epoch_rewards': self.epoch_rewards,
            'epoch_scores': self.epoch_scores,
            'epoch_steps': self.epoch_steps,
            'avg_rewards': self.avg_rewards,
            'avg_scores': self.avg_scores,
            'avg_steps': self.avg_steps,
            'unique_states_visited': len(self.visited_states),
            'window_size': self.window_size,
            'total_epochs': len(self.epoch_rewards),
            'best_score': max(self.epoch_scores) if self.epoch_scores else 0,
            'best_reward': max(self.epoch_rewards) if self.epoch_rewards else 0,
            'max_survival': max(self.epoch_steps) if self.epoch_steps else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training metrics saved to {filepath}")