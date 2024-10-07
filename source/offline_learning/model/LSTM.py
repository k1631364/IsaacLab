
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_num, hidden_num, num_layers, output_num):
        super(LSTM, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_layers = num_layers
        self.output_num = output_num
        
        # Define the LSTM layer
        self.rnn = nn.LSTM(self.input_num, self.hidden_num, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layer for final output
        self.fc = nn.Linear(self.hidden_num, self.output_num)

    def forward(self, x, h_prev, c_prev):

        prev_cell = (h_prev.detach(), c_prev.detach())
        y_rnn, (h, c) = self.rnn(x, prev_cell)
        
        # Take the output from the last time step
        y = self.fc(y_rnn[:, -1, :])
        
        return y, h, c  # Return output, hidden state, and cell state


# class RNN_OneStep(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(RNN_OneStep, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Define the RNN layer
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.5)
        
#         # Define a fully connected layer to output the estimated properties
#         self.fc = nn.Linear(hidden_size, output_size)

#         # Sigmoid activation for output normalization to [0, 1]
#         self.output_activation = nn.Sigmoid()
    
#     def forward(self, x):
#         # Initialize hidden state
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Forward propagate the RNN
#         out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
#         # Use the last time step's output for property estimation
#         out = out[:, -1, :]  # (batch_size, hidden_size)
        
#         # Pass through the fully connected layer
#         out = self.fc(out)  # (batch_size, output_size)

#         out = self.output_activation(out)

#         return out
