import torch
from loguru import logger


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