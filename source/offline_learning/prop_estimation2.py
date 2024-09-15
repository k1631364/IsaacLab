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

def normalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def denormalize(tensor, min_val, max_val, new_min, new_max):
    return (tensor - new_min) / (new_max - new_min) * (max_val - min_val) + min_val

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

    all_data_list = []
    all_labels_list = []
    # Read memories
    memories_path = "/workspace/isaaclab/logs/skrl/sliding_direct_test/2024-09-10_13-09-18_test2"    
    # /workspace/isaaclab/logs/skrl/sliding_direct_test/2024-09-10_13-09-18
    for memory_dir in os.listdir(memories_path):
        print(memory_dir)

        memory_path = os.path.join(memories_path, memory_dir, "memories")
        # print(memory_path)
        
        files = [f for f in os.listdir(memory_path) if os.path.isfile(os.path.join(memory_path, f))]
        # print(files)

        # numbers = re.findall(r'\d+', checkpoint_file)
        # if numbers:
        #     number_part = int(numbers[0])

        memory_filename = os.path.join(memory_path, files[0])
        print(memory_filename)

        memory_all = torch.load(memory_filename)
        # print(type(memory_all))  # Should show <class 'dict'>
        tensor_names = list(memory_all.keys())
        # print("Tensor names:", tensor_names) 
        # print(memory_all["props"][:100,:,:]) 

        ### Stack data from all envs ###
        selected_envs_num = 1

        # memory_all_states = memory_all["states"][:,:selected_envs_num,0:9]
        memory_all_states = memory_all["states"][:,:selected_envs_num,[0,1,2,5,6]]
        states_transposed_data = memory_all_states.transpose(0, 1)
        states_reshaped_data = states_transposed_data.reshape(-1, memory_all_states.shape[2])

        print(states_reshaped_data.shape)

        memory_all_actions = memory_all["actions"][:,:selected_envs_num,:]
        actions_transposed_data = memory_all_actions.transpose(0, 1)
        actions_reshaped_data = actions_transposed_data.reshape(-1, memory_all_actions.shape[2])

        print(actions_reshaped_data.shape)

        memory_all_terminated = memory_all["terminated"][:,:selected_envs_num,:]
        terminated_transposed_data = memory_all_terminated.transpose(0, 1)
        terminated_reshaped_data = terminated_transposed_data.reshape(-1, memory_all_terminated.shape[2])

        print(terminated_reshaped_data.shape)
        
        memory_all_props = memory_all["props"][:,:selected_envs_num,:]
        props_transposed_data = memory_all_props.transpose(0, 1)
        props_reshaped_data = props_transposed_data.reshape(-1, memory_all_props.shape[2])

        print(props_reshaped_data.shape)

        ### Remove data with NaN ###
        nan_mask = torch.isnan(states_reshaped_data[:, 0])
        # print(states_reshaped_data.shape)
        # print(terminated_reshaped_data.shape)
        # data = torch.cat((states_reshaped_data, terminated_reshaped_data), dim=1) 
        data = torch.cat((states_reshaped_data, actions_reshaped_data, terminated_reshaped_data), dim=1) 
        # print(data.shape)
        data = data[~nan_mask, :]
        label = torch.cat((props_reshaped_data, terminated_reshaped_data), dim=1)
        label = label[~nan_mask, :]
        print(data.shape)
        print(label.shape)

        ### Chunk into shorter senquences ###
        # Set sequence length
        sequence_length = 15

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

        ### Cut and pad the sequence with termination ### 
        # Find sequences that contain a `1` in the last dimension
        contains_one_mask = (chunked_data[:, :, -1] == 1).any(dim=1)
        # Remove sequences that contain `1` in the last dimension
        padded_data = chunked_data[~contains_one_mask]

        # Find sequences that contain a `1` in the last dimension
        contains_one_mask = (chunked_label[:, :, -1] == 1).any(dim=1)
        # Remove sequences that contain `1` in the last dimension
        padded_label = chunked_label[~contains_one_mask]

        # # For data
        # # Get the termination mask where `1` occurs in the third dimension
        # terminated_mask = (chunked_data[:, :, -1] == 1).int()

        # # Find the index of the first occurrence of `1` for each sequence
        # first_one_index = terminated_mask.argmax(dim=1)

        # # Calculate the length of the original sequence
        # seq_len = chunked_data.size(1)

        # # Create a tensor of index [0, 1, 2, ..., seq_len-1]
        # seq_index = torch.arange(seq_len, device=chunked_data.device).unsqueeze(0).expand(chunked_data.size(0), -1)

        # # Create a mask to keep only the values up to and including the first occurrence of `1`
        # cutoff_mask = seq_index <= first_one_index.unsqueeze(1)

        # # Prepare to cut off and pad
        # padded_data = torch.zeros_like(chunked_data)

        # # Copy the data up to the first occurrence of `1` into the padded_data tensor
        # for i in range(chunked_data.size(0)):
        #     length = cutoff_mask[i].sum().item()
        #     padded_data[i, -length:] = chunked_data[i, :length]

        # # For labels
        # terminated_mask = (chunked_label[:, :, -1] == 1).int()

        # # Find the index of the first occurrence of `1` for each sequence
        # first_one_index = terminated_mask.argmax(dim=1)

        # # Calculate the length of the original sequence
        # seq_len = chunked_label.size(1)

        # # Create a tensor of index [0, 1, 2, ..., seq_len-1]
        # seq_index = torch.arange(seq_len, device=chunked_label.device).unsqueeze(0).expand(chunked_label.size(0), -1)

        # # Create a mask to keep only the values up to and including the first occurrence of `1`
        # cutoff_mask = seq_index <= first_one_index.unsqueeze(1)

        # # Prepare to cut off and pad
        # padded_label = torch.zeros_like(chunked_label)

        # # Extract the first value of each sequence for padding
        # first_values = chunked_label[:, 0, :]  # Assuming the first value to pad with is the first value in the sequence

        # # Copy the label up to the first occurrence of `1` into the padded_label tensor with the first value = fric not 0
        # for i in range(chunked_label.size(0)):
        #     length = cutoff_mask[i].sum().item()
        #     # Fill the padded_label with the first value for padding
        #     padded_label[i, -length:] = chunked_label[i, :length]
        #     # Pad the beginning with the first value of the sequence
        #     if length < seq_len:
        #         padded_label[i, :-length] = first_values[i]

        ### Remove termination data ###
        filtered_data = padded_data[:, :, :-1]
        filtered_label = padded_label[:, :, :-1]

        print(filtered_data.shape)
        print(filtered_label.shape)

        all_data_list.append(filtered_data)
        all_labels_list.append(filtered_label)

    all_data_tensor = torch.cat(all_data_list, dim=0)
    all_labels_tensor = torch.cat(all_labels_list, dim=0)

    print(all_data_tensor.shape)
    print(all_labels_tensor.shape)

    # Define index for position, rotation, and velocity features
    # position_index = [0, 1, 5, 6]
    # rotation_index = 2
    # velocity_index = [3, 4, 7, 8, 9, 10]

    position_index = [0, 1, 3, 4]
    rotation_index = 2
    velocity_index = [5, 6]

    # Extract features
    positions = all_data_tensor[:, :, position_index]
    rotation = all_data_tensor[:, :, rotation_index]
    velocities = all_data_tensor[:, :, velocity_index]

    # Compute min and max for each feature category
    pos_min = positions.min()
    pos_max = positions.max()
    rot_min = rotation.min()
    rot_max = rotation.max()
    vel_min = velocities.min()
    vel_max = velocities.max()

    feature_target_min = -0.9
    feature_target_max = 0.9

    # Normalize features
    normalized_positions = normalize(positions, pos_min, pos_max, feature_target_min, feature_target_max)
    normalized_rotation = normalize(rotation, rot_min, rot_max, feature_target_min, feature_target_max)
    normalized_velocities = normalize(velocities, vel_min, vel_max, feature_target_min, feature_target_max)

    # Reconstruct the normalized data tensor
    normalized_data = all_data_tensor.clone()
    normalized_data[:, :, position_index] = normalized_positions
    normalized_data[:, :, rotation_index] = normalized_rotation
    normalized_data[:, :, velocity_index] = normalized_velocities

    # Define index for position, rotation, and velocity features
    friction_index = 0
    com_index = [1, 2]

    # Extract features
    frictions = all_labels_tensor[:, :, friction_index]
    coms = all_labels_tensor[:, :, com_index]

    # Compute min and max for each feature category
    fric_min = frictions.min()
    fric_max = frictions.max()
    com_min = coms.min()
    com_max = coms.max()

    print(fric_min)
    print(fric_max)
    print(com_min)
    print(com_max)

    action_target_min = 0.1
    action_target_max = 0.9

    # Normalize features
    normalized_friction = normalize(frictions, fric_min, fric_max, action_target_min, action_target_max)
    normalized_com = normalize(coms, com_min, com_max, action_target_min, action_target_max)

    # Reconstruct the normalized data tensor
    normalized_labels = all_labels_tensor.clone()
    normalized_labels[:, :, friction_index] = normalized_friction
    normalized_labels[:, :, com_index] = normalized_com

    print(normalized_labels.shape)

    full_dataset = PolicyOfflineDataset(normalized_data, normalized_labels)

    train_size = int(0.8 * len(full_dataset))  # 80% of the data for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing

    # Split dataset into training and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 64 # 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example dimensions
    input_size = 7 # 11  # State and action concatenated size
    hidden_size = 64    # Number of features in hidden state
    num_layers = 1      # Number of LSTM layers
    output_size = 1     # Number of physical properties (e.g., friction, CoM)
    num_epochs = 1000
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
    print("Model path")
    print(model_path)

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

        "sequence_length": sequence_length, 

        "position_index": position_index, 
        "rotation_index": rotation_index, 
        "velocity_index": velocity_index, 
        "pos_min": pos_min, 
        "pos_max": pos_max, 
        "rot_min": rot_min, 
        "rot_max": rot_max, 
        "vel_min": vel_min, 
        "vel_max": vel_max, 
        "feature_target_min": feature_target_min, 
        "feature_target_max": feature_target_max,

        "friction_index": friction_index, 
        "com_index": com_index, 
        "fric_min": fric_min, 
        "fric_max": fric_max, 
        "com_min": com_min, 
        "com_max": com_max, 
        "action_target_min": action_target_min, 
        "action_target_max": action_target_max, 

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

    denormalized_targets_record = denormalize(targets_record, fric_min, fric_max, action_target_min, action_target_max)
    denormalized_outputs_record = denormalize(outputs_record, fric_min, fric_max, action_target_min, action_target_max)
    
    record = torch.hstack((denormalized_targets_record.unsqueeze(1), denormalized_outputs_record.unsqueeze(1)))

    print(record.shape)

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

    # denormalised_target_dynamic_fric_data = ((targets_record - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # # denormalised_target_dynamic_fric_data = ((targets_record[:,0] - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # # denormalised_target_comx_data = ((targets_record[:,1] - target_range_min) / (target_range_max - target_range_min)) * (comx_max - comx_min) + comx_min
    # # denormalised_target_comy_data = ((targets_record[:,2] - target_range_min) / (target_range_max - target_range_min)) * (comy_max - comy_min) + comy_min

    # denormalised_outputs_dynamic_fric_data = ((outputs_record - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # # denormalised_outputs_dynamic_fric_data = ((outputs_record[:,0] - target_range_min) / (target_range_max - target_range_min)) * (dynamic_fric_max - dynamic_fric_min) + dynamic_fric_min
    # # denormalised_outputs_comx_data = ((outputs_record[:,1] - target_range_min) / (target_range_max - target_range_min)) * (comx_max - comx_min) + comx_min
    # # denormalised_outputs_comy_data = ((outputs_record[:,2] - target_range_min) / (target_range_max - target_range_min)) * (comy_max - comy_min) + comy_min
    
    # record = torch.hstack((denormalised_target_dynamic_fric_data.unsqueeze(1), denormalised_outputs_dynamic_fric_data.unsqueeze(1)))
    # # record = torch.hstack((denormalised_target_dynamic_fric_data.unsqueeze(1), denormalised_target_comx_data.unsqueeze(1), denormalised_target_comy_data.unsqueeze(1), denormalised_outputs_dynamic_fric_data.unsqueeze(1), denormalised_outputs_comx_data.unsqueeze(1), denormalised_outputs_comy_data.unsqueeze(1)))
    # # record = torch.hstack((targets_record, outputs_record))

    # record = record.tolist()

    # print(np.array(record).shape)

    # record = np.array(record)[:,:,0]

    # targets = record[:, 0]
    # predictions = record[:, 1]

    # # Compute Mean Squared Error
    # mse = np.mean((targets - predictions) ** 2)

    # # Compute Root Mean Squared Error
    # rmse = np.sqrt(mse)

    # print("RMSE")
    # print(rmse)

    # # Define the column names (optional)
    # # columns = ["Target Friction", "Target CoM x", "Target CoM y", "Output Friction", "Output CoM x", "Output CoM y"]
    # columns = ["Target Friction", "Output Friction"]

    # # Create a wandb Table
    # table = wandb.Table(data=record, columns=columns)

    # # Log the table to wandb
    # wandb.log({"my_table": table})

if __name__ == "__main__":
    # run the main function
    main()
