import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
            )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h = torch.randn(self.num_layers, batch_size, self.hidden_size) #.requires_grad_()
        c = torch.randn(self.num_layers, batch_size, self.hidden_size) #.requires_grad_()

        out, (hn, cn) = self.lstm(x, (h, c))
        y = self.linear(out)
        return y