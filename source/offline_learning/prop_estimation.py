# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

from skrl.memories.torch import RandomMemory

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import time
from datetime import datetime
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import pickle

import model.VAE as vaemodel
import model.RNNPropertyEstimator as rnnmodel

# from skrl.resources.preprocessors.torch import RunningStandardScaler

import wandb

class PolicyOfflineDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # return len(self.data)
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return self.data[idx,:], self.labels[idx,0]
        return self.data[idx,:,:], self.labels[idx,:,:]    
    

class LSTMPropertyEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPropertyEstimator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer to output the estimated properties
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for output normalization to [0, 1]
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Use the last time step's output for property estimation
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        out = self.output_activation(out)

        return out
    

# class RNNPropertyEstimator(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(RNNPropertyEstimator, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # Define the RNN layer
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
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

def main():

    ### Weights and biases for recording ###
    run = wandb.init(project="offline_property_estimation")
    config = wandb.config

    # Use gpu if available. Otherwise, use cpu. 
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # H5 file path (to read)
    log_root_path = os.path.join("logs", "prop_estimation", "offline_prop_estimation")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-17_12-56-47/test.hdf5"
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-19_17-52-50/test.hdf5"
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-20_22-58-43/test.hdf5"
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-08-06_10-19-43/test.hdf5"

    # logs/exp_data/predefined_slide/2024-08-06_09-53-14/test.hdf5

    # Fig file path (to save)
    log_fig_dir = os.path.join(log_dir, "figure")
    if not os.path.exists(log_fig_dir):
        os.makedirs(log_fig_dir)

    # Model path (to save)
    log_model_dir = os.path.join(log_dir, "model")
    if not os.path.exists(log_model_dir):
        os.makedirs(log_model_dir)

    # Read memories
    # memory_path = "/workspace/isaaclab/logs/skrl/sliding_direct/2024-08-16_21-51-14/memory_all/memories/24-08-16_21-52-19-857841_memory_0x7f440ff1c130.pt"
    memory_path = "/workspace/isaaclab/logs/skrl/sliding_direct/2024-08-18_17-35-42/memory_all/memories/24-08-18_18-24-00-446689_memory_0x7f87d00f41f0.pt"
    memory_all = torch.load(memory_path)
    # print(memory_all)
    print(type(memory_all))  # Should show <class 'dict'>
    tensor_names = list(memory_all.keys())
    print("Tensor names:", tensor_names) 

    # tensor_values = [memory_all[name] for name in tensor_names]
    # print(tensor_values)
    print(memory_all["states"].shape) 
    print(memory_all["actions"].shape)
    print(memory_all["terminated"].shape)
    print(memory_all["log_prob"].shape)
    print(memory_all["rewards"].shape)
    print(memory_all["props"].shape)

    # obs = torch.cat((normalized_past_puck_pos_obs_x, normalized_past_puck_pos_obs_y, normalized_past_puck_vel_obs_x, normalized_past_puck_vel_obs_y, normalized_past_pusher_pos_obs_x, normalized_past_pusher_pos_obs_y, normalized_past_pusher_vel_obs_x, normalized_past_pusher_vel_obs_y, normalized_goal_tensor_x), dim=1)

    # ### Stack data from all envs ###
    selected_envs_num = 10

    memory_all_states = memory_all["states"][:,:selected_envs_num,[0,1,4,5]]
    states_transposed_data = memory_all_states.transpose(0, 1)
    # states_transposed_data = memory_all["states"].transpose(0, 1)
    states_reshaped_data = states_transposed_data.reshape(-1, memory_all_states.shape[2])

    print(states_reshaped_data.shape)
    
    # State normalisation
    state_min = torch.min(states_reshaped_data)
    state_max = torch.max(states_reshaped_data)
    state_range_min = -0.9
    state_range_max = 0.9
    
    normalised_states_reshaped_data = state_range_min + ((states_reshaped_data - state_min) * (state_range_max - state_range_min)) / (state_max - state_min)

    # xpoints = np.arange(0, 10)
    # plt.plot(xpoints, states_reshaped_data[0:10,0].cpu().detach().numpy(), label="before reshape dynamic friction")
    # log_fig_path = os.path.join(log_fig_dir, 'test_states.png')
    # plt.savefig(log_fig_path)

    memory_all_actions = memory_all["actions"][:,:selected_envs_num,:]
    actions_transposed_data = memory_all_actions.transpose(0, 1)
    # actions_transposed_data = memory_all["actions"].transpose(0, 1)
    actions_reshaped_data = actions_transposed_data.reshape(-1, memory_all["actions"].shape[2])

    print(actions_reshaped_data.shape)

    # xpoints = np.arange(0, 10)
    # plt.plot(xpoints, actions_reshaped_data[0:10,0].cpu().detach().numpy(), label="before reshape dynamic friction")
    # log_fig_path = os.path.join(log_fig_dir, 'test_actions.png')
    # plt.savefig(log_fig_path)

    memory_all_terminated = memory_all["terminated"][:,:selected_envs_num,:]
    terminated_transposed_data = memory_all_terminated.transpose(0, 1)
    # terminated_transposed_data = memory_all["terminated"].transpose(0, 1)
    terminated_reshaped_data = terminated_transposed_data.reshape(-1, memory_all["terminated"].shape[2])

    print(terminated_reshaped_data.shape)

    # xpoints = np.arange(0, 10)
    # plt.plot(xpoints, terminated_reshaped_data[0:10,0].cpu().detach().numpy(), label="before reshape dynamic friction")
    # log_fig_path = os.path.join(log_fig_dir, 'test_terminated.png')
    # plt.savefig(log_fig_path)

    memory_all_props = memory_all["props"][:,:selected_envs_num,:]
    props_transposed_data = memory_all_props.transpose(0, 1)
    # props_transposed_data = memory_all["props"].transpose(0, 1)
    props_reshaped_data = props_transposed_data.reshape(-1, memory_all["props"].shape[2])

    print(props_reshaped_data.shape)

    # Props normalisation
    dynamic_fric_data = props_reshaped_data[:,0]
    dynamic_fric_min = torch.min(dynamic_fric_data)
    dynamic_fric_max = torch.max(dynamic_fric_data)
    target_range_min = 0.1
    target_range_max = 0.9
    
    normalised_dynamic_fric_data = target_range_min + ((dynamic_fric_data - dynamic_fric_min) * (target_range_max - target_range_min)) / (dynamic_fric_max - dynamic_fric_min)

    comx_data = props_reshaped_data[:,1]
    comx_min = torch.min(comx_data)
    comx_max = torch.max(comx_data)
    target_range_min = 0.1
    target_range_max = 0.9
    
    normalised_comx_data = target_range_min + ((comx_data - comx_min) * (target_range_max - target_range_min)) / (comx_max - comx_min)

    comy_data = props_reshaped_data[:,1]
    comy_min = torch.min(comy_data)
    comy_max = torch.max(comy_data)
    target_range_min = 0.1
    target_range_max = 0.9
    
    normalised_comy_data = target_range_min + ((comy_data - comy_min) * (target_range_max - target_range_min)) / (comy_max - comy_min)

    normalised_props_reshaped_data = torch.hstack((normalised_dynamic_fric_data.unsqueeze(1), normalised_comx_data.unsqueeze(1), normalised_comy_data.unsqueeze(1)))
    
    xpoints = np.arange(0, 10)
    plt.plot(xpoints, props_reshaped_data[0:10,0].cpu().detach().numpy(), label="before reshape dynamic friction")
    log_fig_path = os.path.join(log_fig_dir, 'test_props.png')
    plt.savefig(log_fig_path)

    print(states_reshaped_data.shape)
    print(actions_reshaped_data.shape)
    print(terminated_reshaped_data.shape)
    print(props_reshaped_data.shape)

    ### Remove data with NaN ###
    # print(actions_reshaped_data)
    nan_mask = torch.isnan(states_reshaped_data[:, 0])
    print(states_reshaped_data.shape)
    print(terminated_reshaped_data.shape)
    data = torch.cat((normalised_states_reshaped_data, terminated_reshaped_data), dim=1) 
    print(data.shape)
    data = data[~nan_mask, :]
    label = torch.cat((normalised_props_reshaped_data, terminated_reshaped_data), dim=1)
    label = label[~nan_mask, :]

    ### Chunk into shorter senquences ###
    # Set sequence length
    sequence_length = 20

    # Ensure the total number of data points is divisible by sequence_length
    remainder = data.size(0) % sequence_length
    if remainder != 0:
        data = data[:-remainder]  # Trim the excess data points

    num_sequences = data.size(0) // sequence_length
    chunked_data = data.view(num_sequences, sequence_length, -1)

    # Ensure the total number of label points is divisible by sequence_length
    remainder = label.size(0) % sequence_length
    if remainder != 0:
        label = label[:-remainder]  # Trim the excess label points

    num_sequences = label.size(0) // sequence_length
    chunked_label = label.view(num_sequences, sequence_length, -1)

    print("Chunked data")
    print(chunked_data.shape)
    print(chunked_label.shape)

    # time.sleep(300)

    ### Remove data containing episode termination ###
    # last_column = chunked_data[:, :, -1]
    # mask = (last_column != 1).all(dim=1)
    # filtered_data = chunked_data[mask]
    # filtered_label = chunked_label[mask]

    # print("Filtered")
    # print(filtered_data.shape)
    # print(filtered_label.shape)

    ### Cut and pad the sequence with termination ### 
    # first_one_idx = (chunked_data[:,:,-1] == 1).int().argmax(dim=1)
    terminated_mask = (chunked_data[:, :, -1] == 1).int()
    last_one_indices = terminated_mask.cumsum(dim=1).argmax(dim=1)
    last_one_indices = torch.clamp(last_one_indices + 1, max=chunked_data.size(1))

    cutoff_mask = torch.arange(chunked_data.size(1)).expand(chunked_data.size(0), -1).to(torch_device) >= last_one_indices.unsqueeze(1)
    chunked_data[~cutoff_mask.unsqueeze(-1).expand_as(chunked_data)] = 0

    # Filter out sequences that are all zeros
    non_zero_sequences = torch.any(chunked_data != 0, dim=(1, 2))
    padded_data = chunked_data[non_zero_sequences]

    terminated_mask = (chunked_label[:, :, -1] == 1).int()
    last_one_indices = terminated_mask.cumsum(dim=1).argmax(dim=1)
    last_one_indices = torch.clamp(last_one_indices + 1, max=chunked_label.size(1))

    cutoff_mask = torch.arange(chunked_label.size(1)).expand(chunked_label.size(0), -1).to(torch_device) >= last_one_indices.unsqueeze(1)
    chunked_label[~cutoff_mask.unsqueeze(-1).expand_as(chunked_label)] = 0

    # Filter out sequences that are all zeros
    non_zero_sequences = torch.any(chunked_label != 0, dim=(1, 2))
    padded_label = chunked_label[non_zero_sequences]

    # xpoints = np.arange(0, 10)
    # plt.plot(xpoints, chunked_data[1278,:,0].cpu().detach().numpy(), label="before reshape dynamic friction")
    # log_fig_path = os.path.join(log_fig_dir, 'test_chunked.png')
    # plt.savefig(log_fig_path)

    print(log_fig_path)

    ### Remove termination data ###
    filtered_data = padded_data[:, :, :-1]
    filtered_label = padded_label[:, :, :-1]
    # filtered_data = filtered_data[:, :, :-1]
    # filtered_label = filtered_label[:, :, :-1]

    full_dataset = PolicyOfflineDataset(filtered_data, filtered_label)

    train_size = int(0.8 * len(full_dataset))  # 80% of the data for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing

    # Split dataset into training and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example dimensions
    input_size = 4  # State and action concatenated size
    hidden_size = 64    # Number of features in hidden state
    num_layers = 1      # Number of LSTM layers
    output_size = 1     # Number of physical properties (e.g., friction, CoM)
    num_epochs = 500
    learning_rate = 0.001

    model = rnnmodel.RNNPropertyEstimator(input_size, hidden_size, num_layers, output_size).to(torch_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Concatenate state and action vectors
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            # targets = targets[:, -1, :]
            targets = targets[:, -1, 0].view(-1,1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Record average loss after each epoch to weights and biases
        wandb.log({"loss": loss.item()})

    # Save the learnt model to the file
    model_path = log_model_dir + "/LSTM_best.pth"
    torch.save(model.to(torch_device).state_dict(), model_path)

    model_params_dict = {
        "input_size": input_size, 
        "hidden_size": hidden_size, 
        "num_layers": num_layers, 
        "output_size": output_size, 
        "num_epochs": num_epochs, 
        "learning_rate": learning_rate, 
        "criterion": criterion, 
        "optimizer": optimizer, 
        "model_path": model_path, 
        "state_min": state_min, 
        "state_max": state_max, 
        "state_range_min": state_range_min, 
        "state_range_max": state_range_max, 
        "dynamic_fric_min": dynamic_fric_min, 
        "dynamic_fric_max": dynamic_fric_max, 
        "comx_min": comx_min, 
        "comx_max": comx_max, 
        "comy_min": comy_min, 
        "comy_max": comy_max, 
        "target_range_min": target_range_min, 
        "target_range_max": target_range_max, 
    }
    model_params_path = log_model_dir + "/model_params_dict.pkl"
    with open(model_params_path, "wb") as fp: 
        pickle.dump(model_params_dict, fp)

    # /workspace/isaaclab/logs/prop_estimation/offline_prop_estimation/LSTM_best.pth

    # Eval

    # Load the learnt model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))

    model.eval()

    targets_record_list = []
    outputs_record_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            # targets = targets[:, -1, :]
            targets = targets[:, -1, 0].view(-1,1)
            for i in range(len(inputs)):
                input = inputs[i].unsqueeze(0)  # Add batch dimension
                target = targets[i].unsqueeze(0)  # Add batch dimension

                # print(f"Sample index: {i}")
                # print(f"Sample shape: {input.shape}")
                # print(f"Label shape: {target.shape}")

                # Forward pass
                # Input size torch.Size([1, 20, 4]) [batch_size, , seq_length, 4]
                # Output size torch.Size([1, 1])    [batch_size, 1]
                output = model(input)
                loss = criterion(output, target)

                # print(f"Sample index: {i}")
                # # print(f"Sample shape: {input}")
                # print(f"Label shape: {target.shape}")
                # print(f"Output shape: {output.shape}")
                
                targets_record_list.append(target)
                outputs_record_list.append(output)

    targets_record = torch.cat(targets_record_list, dim=0)
    outputs_record = torch.cat(outputs_record_list, dim=0)

    denormalised_target_dynamic_fric_data = ((targets_record - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # denormalised_target_dynamic_fric_data = ((targets_record[:,0] - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # denormalised_target_comx_data = ((targets_record[:,1] - target_range_min) / (target_range_max - target_range_min)) * (comx_max - comx_min) + comx_min
    # denormalised_target_comy_data = ((targets_record[:,2] - target_range_min) / (target_range_max - target_range_min)) * (comy_max - comy_min) + comy_min

    denormalised_outputs_dynamic_fric_data = ((outputs_record - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # denormalised_outputs_dynamic_fric_data = ((outputs_record[:,0] - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # denormalised_outputs_comx_data = ((outputs_record[:,1] - target_range_min) / (target_range_max - target_range_min)) * (comx_max - comx_min) + comx_min
    # denormalised_outputs_comy_data = ((outputs_record[:,2] - target_range_min) / (target_range_max - target_range_min)) * (comy_max - comy_min) + comy_min
    
    record = torch.hstack((denormalised_target_dynamic_fric_data.unsqueeze(1), denormalised_outputs_dynamic_fric_data.unsqueeze(1)))
    # record = torch.hstack((denormalised_target_dynamic_fric_data.unsqueeze(1), denormalised_target_comx_data.unsqueeze(1), denormalised_target_comy_data.unsqueeze(1), denormalised_outputs_dynamic_fric_data.unsqueeze(1), denormalised_outputs_comx_data.unsqueeze(1), denormalised_outputs_comy_data.unsqueeze(1)))
    # record = torch.hstack((targets_record, outputs_record))

    record = record.tolist()

    print(np.array(record).shape)

    record = np.array(record)[:,:,0]

    targets = record[:, 0]
    predictions = record[:, 1]

    # Compute Mean Squared Error
    mse = np.mean((targets - predictions) ** 2)

    # Compute Root Mean Squared Error
    rmse = np.sqrt(mse)

    print("RMSE")
    print(rmse)

    # Define the column names (optional)
    # columns = ["Target Friction", "Target CoM x", "Target CoM y", "Output Friction", "Output CoM x", "Output CoM y"]
    columns = ["Target Friction", "Output Friction"]

    # Create a wandb Table
    table = wandb.Table(data=record, columns=columns)

    # Log the table to wandb
    wandb.log({"my_table": table})

if __name__ == "__main__":
    # run the main function
    main()
