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

# from skrl.resources.preprocessors.torch import RunningStandardScaler

import wandb

class PredefinedExpDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        # return len(self.data)
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx,:], self.labels[idx,0]

def main():

    ### Weights and biases for recording ###
    run = wandb.init(project="exploration_predefined_ssl")
    config = wandb.config

    # Fig file path (to save)
    log_fig_eval_dir = os.path.join("logs", "exp_data", "exploration_ssl")
    if not os.path.exists(log_fig_eval_dir):
        os.makedirs(log_fig_eval_dir)

    # Dataset

    # Read data
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-20_22-58-43/test.hdf5"
    log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-08-06_10-19-43/test.hdf5"

    hf = h5py.File(log_h5_path, 'r')
    states = hf['states'][:]    # shape: (max_count*num_epoch, num_envs, obs_dim)
    exp_traj = hf['exp_traj'][:]
    num_itr = hf['num_itr'][()]

    max_count = int(states.shape[0]/num_itr)

    swapped_array = np.swapaxes(states, 0, 1)
    states_reshaped = swapped_array.reshape((-1,8))

    # Create dataset
    states_reshaped[np.random.permutation(states_reshaped.shape[0]), :]
    puck_pos_traj_data = states_reshaped[:,0].reshape((-1, max_count))
    print(puck_pos_traj_data.shape)

    puck_dynamicfric_data = states_reshaped[:,6].reshape((-1, max_count))
    print(puck_dynamicfric_data.shape)

    full_dataset = PredefinedExpDataset(puck_pos_traj_data, puck_dynamicfric_data)

    # Split dataset into training and test sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the data for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Use gpu if available. Otherwise, use cpu. 
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Embedding look up table path (to save)
    log_model_dir = os.path.join("logs", "exp_lookuptable", "predefined")
    if not os.path.exists(log_model_dir):
        os.makedirs(log_model_dir)


    # model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
    model_params_path = "/workspace/isaaclab/logs/exp_model/exploration_sslmodel_params_dict.pkl"
    with open(model_params_path, "rb") as fp: 
        model_params_dict = pickle.load(fp)

    input_dim = model_params_dict["input_dim"]
    latent_dim = model_params_dict["latent_dim"]
    char_dim = model_params_dict["char_dim"]
    output_dim = model_params_dict["output_dim"]
    dropout = model_params_dict["dropout"]        
    KLbeta = model_params_dict["KLbeta"]
    rec_weight = model_params_dict["rec_weight"]
    learning_rate = model_params_dict["learning_rate"]
    model_path = model_params_dict["model_path"]
    max_count = model_params_dict["max_count"]
    exp_traj = model_params_dict["exp_traj"]
    
    # Create VAE model
    model = vaemodel.VAE(x_dim=input_dim, z_dim=latent_dim, char_dim=char_dim, y_dim=output_dim, dropout=dropout, beta=KLbeta, alpha=rec_weight).to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Eval

    # Load the learnt model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(torch_device)))

    model.eval()
    total_loss = 0.0
    KL_loss = 0.0
    rec_loss = 0.0

    latent_z_list = []
    label_list = []

    with torch.no_grad():
        for batch_idx, (samples, labels) in enumerate(test_loader):
            for i in range(len(samples)):
                sample = samples[i].unsqueeze(0)  # Add batch dimension
                label = labels[i].unsqueeze(0)  # Add batch dimension

                # print(f"Sample index: {i}")
                # print(f"Sample shape: {sample.shape}")
                # print(f"Label shape: {label.shape}")

                # VAE input traj
                x = sample.clone().to(torch_device).float()

                # Reconstruction target traj
                y_opt = x
                y_opt = torch.Tensor(y_opt).to(torch_device)

                x_char = label
                # x_char = torch.zeros(x.shape[0])

                # Reconstruct input exploration force/torque traj
                lower_bound, z, y, y_char = model(x, x_char, y_opt, torch_device)
                loss = -sum(lower_bound)

                latent_z_list.append(z)
                label_list.append(x_char)

                total_loss += loss.item()
                KL_loss += lower_bound[0]
                rec_loss += lower_bound[1]

        # print(f"Test Loss: {(avg_total_loss):.4f}")

        latent_z_tensor = torch.cat(latent_z_list, dim=0)
        latent_z_np = latent_z_tensor.cpu().detach().numpy()

        label_tensor = torch.cat(label_list, dim=0)
        label_np = label_tensor.cpu().detach().numpy()

        print(label_np.shape)

        latent_z_df = pd.DataFrame(latent_z_np, columns = range(latent_dim))
        latent_z_df['dynamic friction'] = pd.DataFrame(label_np)
        print(latent_z_df)

        embedding_lookuptable_path = log_model_dir + "/dynamicfriction_z2.pkl"
        with open(embedding_lookuptable_path, "wb") as fp: 
            pickle.dump(latent_z_df, fp)

        if latent_dim <=2: 
            n_c = 2
        else: 
            n_c = 3
        pca = PCA(n_components = n_c)
        data = latent_z_df.loc[:,range(latent_dim)]
        df_pca = pd.DataFrame(pca.fit_transform(data)) 

        df_pca['dynamic friction'] = latent_z_df['dynamic friction']

        # contribution rate
        pca_cont = pca.explained_variance_ratio_
        if n_c==2: 
            eval_latent_pca_table = wandb.Table(columns=['Variable', 'PC0', 'PC1'])
            eval_latent_pca_table.add_data('PCA latent space (eval data)', pca_cont[0], pca_cont[1])
        else: 
            eval_latent_pca_table = wandb.Table(columns=['Variable', 'PC0', 'PC1', 'PC2'])
            eval_latent_pca_table.add_data('PCA latent space (eval data)', pca_cont[0], pca_cont[1], pca_cont[2])

        for pc_x in range(2):
            for pc_y in range(2):
                if pc_x < pc_y:
                    #print(pc_x)
                    #print(pc_y)
                    sns.scatterplot(data=df_pca, x=pc_x, y=pc_y, hue='dynamic friction').set(title='VAE latent pca (eval data)')
                    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
                    plot_path = os.path.join(log_fig_eval_dir, 'eval_VAE_latent_pca'+ str(pc_x) + str(pc_y) + '.png')
                    plt.savefig(plot_path, bbox_inches="tight")
                    #plt.show()
                    plt.clf()
                    
                    # Record PCA plot to weights and biase
                    wandb.log({'eval_VAE_latent_pca': wandb.Image(plot_path)})

        # Record pca contribution ratio to weights and biases
        run.log({"eval_VAE_latent_pca_table": eval_latent_pca_table})

if __name__ == "__main__":
    # run the main function
    main()
