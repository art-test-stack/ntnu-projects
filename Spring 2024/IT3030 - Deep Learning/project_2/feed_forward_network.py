import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size = 2):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_size, 10),
            nn.Linear(input_size, input_size//2),
            nn.Linear(input_size//2, 10),
            nn.Linear(10, 4),
            nn.Linear(4, 1),
            # nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)