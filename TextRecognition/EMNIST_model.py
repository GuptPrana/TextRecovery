## Model
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv Layers
        # Input: (64, 1, 28, 28) (N, C, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Dense Layers 
        # Input: (64, 14, 14, 32)
        self.dense = nn.Sequential(
            nn.Flatten(), # Output: (64, 6272)
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.dense(out)
        return out
