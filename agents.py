from loguru import logger
from typing import List, Dict
import random
from q_table import QTable
import torch
from deepqnetwork import DeepQNetwork, ReplayBuffer

class AgentBase():
    def __init__(self):
        pass

    def make_move_decision(self, game_state: List[Dict]):
        raise NotImplementedError("Make move decision is not implemented")


class RandomAgent(AgentBase):
    def make_move_decision(self, game_state: List[Dict]):
        """
        Makes a random decision: go straight, turn left, or turn right
        Returns a one-hot encoded relative action [straight, left, right]
        """
        # Random choice: straight, left, or right
        rel_move = [0, 0, 0]
        choice = random.randint(0, 2)
        rel_move[choice] = 1
        action_name = ["STRAIGHT", "LEFT", "RIGHT"][choice]
        logger.debug(f"RandomAgent chose: {action_name}")
        return rel_move


class QAgent(AgentBase):
    def __init__(self,
                 learning_rate: float = 0.1,
                 discount: float = 0.95,
                 exploration_prob: float = 0.1,
                 init_strategy: str = "zero",
                 epochs: int = 1000,
                 use_epsilon_decay: bool = False,
                 epsilon_start: float = 0.9,
                 epsilon_decay_steps: int = 50000):
        self.epochs = epochs
        self.qtable = QTable(
            learning_rate=learning_rate,
            discount=discount,
            exploration_prob=exploration_prob,
            init_strategy=init_strategy,
            use_epsilon_decay=use_epsilon_decay,
            epsilon_start=epsilon_start,
            epsilon_decay_steps=epsilon_decay_steps
        )
        self.current_epoch = 0
        self.last_action_name = None  # Store last action name for updates

    def make_move_decision(self, game_state: List[Dict]) -> List[int]:
        """
        Makes a decision using Q-table with epsilon-greedy strategy
        Returns a one-hot encoded relative action [straight, left, right]
        """
        # Get action from Q-table (uses epsilon-greedy with decay)
        action_name = self.qtable.get_action(game_state)

        # Store the action name for later update
        self.last_action_name = action_name

        # Convert action name to one-hot encoded array
        action_map = {
            "straight": [1, 0, 0],
            "left": [0, 1, 0],
            "right": [0, 0, 1]
        }

        rel_move = action_map[action_name]

        logger.debug(f"QAgent chose: {action_name.upper()}")

        return rel_move

    def update_policy(self, state: List[Dict], reward: float,
                      next_state: List[Dict]):
        """
        Update Q-table based on experience
        Uses the last action taken (stored in self.last_action_name)
        """
        if self.last_action_name is None:
            logger.warning(
                "No action to update - "
                "make_move_decision must be called first"
            )
            return

        self.qtable.update_q_table(
            state=state,
            action=self.last_action_name,
            reward=reward,
            next_state=next_state
        )

    def load_from_file(self, path: str):
        """Load Q-table from file"""
        self.qtable = QTable.load_qtable(path)

    def get_current_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.qtable.get_current_epsilon()


class DeepQAgent(AgentBase):
    def __init__(
        self,
        n_rows=20,
        n_cols=20,
        n_channels=4,
        n_actions=3,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        use_double_dqn=False
    ):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        
        
        self.policy_network = DeepQNetwork(n_cols=n_cols,
                                            n_rows=n_rows,
                                            n_channels=n_channels).to(self.device)

        #frozen 'stable network'
        self.target_network = DeepQNetwork(n_cols=n_cols,
                                            n_rows=n_rows,
                                            n_channels=n_channels).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval() 

        #replay buffer
        self.memory = ReplayBuffer(
            capacity=buffer_capacity
        )
        logger.info(f"Initialized buffer with {buffer_capacity} capacity")


        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(params=self.policy_network.parameters(), lr=learning_rate)

        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.total_loss = 0
        self.loss_count = 0

        logger.info(
            f"DQN Agent initialized: "
            f"lr={learning_rate}, gamma={gamma}, "
            f"batch_size={batch_size}, "
            f"double_dqn={use_double_dqn}"
        )

        self.loss_history = []

    def add_experience_to_buffer(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def make_move_decision(self, state, training=True):

        #explore
        if training and random.random() <= self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            logger.debug(f"Selected Exploration: {action}")
            return action
        
        #exploit
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            state_tensor = state_tensor.unsqueeze(0) #add batch dimension
            state_tensor = state_tensor.to(self.device)

            q_vals = self.policy_network(state_tensor).to(self.device)
            action = torch.argmax(q_vals, dim=1).item()
            logger.debug(f"Selected Exploitation: {action}")

            return action
            

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones  = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        current_q_values = self.policy_network(states)
        #select q-value for taken actions
        current_q_values = current_q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1) #(batch, )
 
        with torch.no_grad():
            #no need to update target network
            next_q_target_values = self.target_network(next_states)
            max_next_q = next_q_target_values.max(dim=1)[0] # (batch, )
            target_q = rewards + (1 - dones) * self.gamma * max_next_q


        loss = self.criterion(current_q_values, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps += 1
        logger.info(f"Step: {self.steps} | Training Loss: {loss}")
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            logger.info(f"Target network updated at step {self.steps}")

        self.loss_history.append(loss.item())
        return loss.item()


