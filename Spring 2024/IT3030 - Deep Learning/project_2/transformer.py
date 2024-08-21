# Embedding inspired by 
# https://github.com/CVxTz/time_series_forecasting/blob/main/time_series_forecasting/model.py

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, num_layers=2, hidden_size=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embedding = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True, 
            dropout=dropout
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        
        y = self.transformer_encoder(x)
        z = self.linear(y)
        return z