
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Define a fully connected layer to output the estimated properties
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for output normalization to [0, 1]
        # self.output_activation = nn.Sigmoid()
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Use the last time step's output for property estimation
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        out = self.output_activation(out)

        return out
