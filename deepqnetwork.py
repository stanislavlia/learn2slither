import torch
from loguru import logger
from collections import deque
import random
import numpy as np


class DeepQNetwork(torch.nn.Module):
    def __init__(self, n_rows, n_cols, n_channels):
        super().__init__()
        #assume 20x20
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels

        self.conv_layer1 = torch.nn.Conv2d(kernel_size=5, in_channels=4, out_channels=8) # output 16x16
        self.conv_layer2 = torch.nn.Conv2d(kernel_size=5, in_channels=8, out_channels=32) # output 12x12
        self.relu = torch.nn.ReLU()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.flatten = torch.nn.Flatten()

        self.fc = torch.nn.Linear(in_features=32*2*2, out_features=3)
    
    def forward(self, x : torch.Tensor): 
        #expects tensor (batch, n_row, n_col, n_channels)
        #returns (batch, 3) q-vals for actions
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv_layer1(x)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    



class ReplayBuffer():
    def __init__(self, capacity = 20_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, n):
        batch = random.sample(self.buffer, n)
    
        states, actions, rewards, next_states, dones = zip(batch)

        return (
            np.array(states, dtype=np.float32),      # (batch, 20, 20, 4)
            np.array(actions, dtype=np.int64),       # (batch,)
            np.array(rewards, dtype=np.float32),     # (batch,)
            np.array(next_states, dtype=np.float32), # (batch, 20, 20, 4)
            np.array(dones, dtype=np.float32)        # (batch,)
        )

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":

    dqn = DeepQNetwork(n_rows=20,
                       n_cols=20,
                       n_channels=4)
    
    test_batch = torch.randn((32, 20, 20, 4))

    y = dqn.forward(test_batch)

    print(y.shape)