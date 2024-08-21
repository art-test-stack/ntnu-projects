import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 1, device = torch.device("mps")):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1,
            device=device
            )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)

        if x.is_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        h = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h, c))
        y = self.linear(out)
        return y