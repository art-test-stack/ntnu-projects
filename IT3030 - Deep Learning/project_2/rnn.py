import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.ffn = nn.Linear(hidden_size, 1)

        
    def forward(self, x):
        h = torch.randn(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.rnn(x, h.detach())

        y = self.ffn(out[:, -1, :]) 
        return y