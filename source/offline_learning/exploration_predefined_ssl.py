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

    # Use gpu if available. Otherwise, use cpu. 
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # H5 file path (to read)
    log_root_path = os.path.join("logs", "exp_data", "predefined_slide")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-17_12-56-47/test.hdf5"
    # log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-19_17-52-50/test.hdf5"
    log_h5_path = "/workspace/isaaclab/logs/exp_data/predefined_slide/2024-07-20_22-58-43/test.hdf5"

    # Fig file path (to save)
    log_fig_dir = os.path.join("logs", "exp_data", "exploration_ssl")
    if not os.path.exists(log_fig_dir):
        os.makedirs(log_fig_dir)

    # Model path (to save)
    log_model_dir = os.path.join("logs", "exp_model", "exploration_ssl")
    if not os.path.exists(log_model_dir):
        os.makedirs(log_model_dir)

    # Read data
    hf = h5py.File(log_h5_path, 'r')
    states = hf['states'][:]    # shape: (max_count*num_epoch, num_envs, obs_dim)
    truncated = hf['truncated'][:]
    exp_traj = hf['exp_traj'][:]
    num_itr = hf['num_itr'][()]

    print(exp_traj.shape[0])
    print(num_itr)

    # # Get the maximum step index
    # max_step = len(trajectory) - 1

    # # Use np.clip to ensure steps are within valid range
    # clipped_steps = np.clip(current_steps, 0, max_step)

    # # Get actions using the clipped steps
    # actions = trajectory[clipped_steps]

    env_id = 27
    env_num = states.shape[1]
    # max_count = int(states.shape[0]/env_num)
    max_count = int(states.shape[0]/num_itr)
    traj_id = 27

    # # Plot original data (before reshape)
    # xpoints = np.arange(0, max_count)
    # ypoints = states[max_count*traj_id:max_count*(traj_id+1),env_id,0]
    # ypoints2 = states[max_count*traj_id:max_count*(traj_id+1),env_id,6]

    # plt.plot(xpoints, ypoints, label="before reshape puck pos")
    # plt.plot(xpoints, ypoints2, label="before reshape dynamic friction")

    # log_fig_path = os.path.join(log_fig_dir, 'test.png')
    # plt.savefig(log_fig_path)

    swapped_array = np.swapaxes(states, 0, 1)
    states_reshaped = swapped_array.reshape((-1,8))

    # # Plot original data (after reshape)
    # xpoints = np.arange(0, max_count)
    # ypoints = states_reshaped[(max_count*env_num)*(env_id)+(max_count*traj_id):(max_count*env_num)*(env_id)+(max_count*(traj_id+1)),0]
    # ypoints2 = states_reshaped[(max_count*env_num)*(env_id)+(max_count*traj_id):(max_count*env_num)*(env_id)+(max_count*(traj_id+1)),6]

    # plt.plot(xpoints, ypoints, label="after reshape puck pos")
    # plt.plot(xpoints, ypoints2, label="after reshape dynamic friction")

    # log_fig_path = os.path.join(log_fig_dir, 'test2.png')
    # plt.savefig(log_fig_path)

    # Create dataset
    states_reshaped[np.random.permutation(states_reshaped.shape[0]), :]
    puck_pos_traj_data = states_reshaped[:,0].reshape((-1, max_count))
    print(puck_pos_traj_data.shape)

    puck_dynamicfric_data = states_reshaped[:,6].reshape((-1, max_count))
    print(puck_dynamicfric_data.shape)

    # xpoints = np.arange(0, max_count)
    # ypoints = puck_pos_traj_data[env_id*traj_id,:]
    # ypoints2 = puck_dynamicfric_data[env_id*traj_id,:]

    # plt.plot(xpoints, ypoints, label="dataset puck pos")
    # plt.plot(xpoints, ypoints2, label="dataset dynamic fric label")

    # log_fig_path = os.path.join(log_fig_dir, 'test3.png')
    # plt.savefig(log_fig_path)

    full_dataset = PredefinedExpDataset(puck_pos_traj_data, puck_dynamicfric_data)

    # print(full_dataset.__len__())

    # xpoints = np.arange(0, max_count)
    # curr_data, curr_label = full_dataset.__getitem__(461)
    # ypoints = curr_data
    # plt.plot(xpoints, ypoints, label="dynamic friction: "+str(curr_label))

    # curr_data2, curr_label2 = full_dataset.__getitem__(267)
    # ypoints2 = curr_data2
    # plt.plot(xpoints, ypoints2, label="dynamic friction: "+str(curr_label2))

    # plt.legend()
    # plt.xlabel("Time (step)")
    # plt.ylabel("Puck pos x (m)")

    # log_fig_path = os.path.join(log_fig_dir, 'test3.png')
    # plt.savefig(log_fig_path)

    train_size = int(0.8 * len(full_dataset))  # 80% of the data for training
    test_size = len(full_dataset) - train_size  # Remaining 20% for testing

    # Split dataset into training and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Data info
    input_dim = max_count
    latent_dim = 5	# latent space dimension
    output_dim = input_dim
    char_dim = 1 # # of explicit characteristics

    # Learning params
    learning_rate = 0.0001	# Learning rate
    num_epochs = 1000

    dropout = 0.1	# Dropout rate, larger = prevent more from overfitting
    KLbeta = 0.06		# Weight on KL loss, smaller = focuses more on reconstruction
    rec_weight = 5.0

    # Create VAE model
    model = vaemodel.VAE(x_dim=input_dim, z_dim=latent_dim, char_dim=char_dim, y_dim=output_dim, dropout=dropout, beta=KLbeta, alpha=rec_weight).to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# Train
	# Set model to training mode
    model.train()

    # for epoch in range(tqdm(num_epochs)):
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        total_loss = 0.0
        KL_loss = 0.0
        rec_loss = 0.0

        for batch_idx, (samples, labels) in enumerate(train_loader):
            # print(batch_idx)
            # print(samples.shape)
            # print(labels.shape)

            # VAE input traj
            x = samples.clone().to(torch_device).float()

            # Reconstruction target traj
            y_opt = x
            y_opt = torch.Tensor(y_opt).to(torch_device)

            x_char = labels

            # Reconstruct input exploration force/torque traj
            lower_bound, z, y, y_char = model(x, x_char, y_opt, torch_device)
            loss = -sum(lower_bound)

            # Back propagation to update the model
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            KL_loss += lower_bound[0]
            rec_loss += lower_bound[1]

            # print(total_loss)

        # Print average loss after each epoch
        avg_total_loss = total_loss / len(train_loader)
        avg_KL_loss = KL_loss / len(train_loader)
        avg_rec_loss = rec_loss / len(train_loader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_total_loss:.4f}")

        print("EPOCH: {} loss: {}".format(epoch, avg_total_loss))

        # Record average loss after each epoch to weights and biases
        wandb.log({"loss": avg_total_loss})
        wandb.log({"KL": avg_KL_loss})
        wandb.log({"reconstruction": avg_rec_loss})

    # Save the learnt model to the file
    model_path = log_model_dir + "/VAE_best.pth"
    torch.save(model.to(torch_device).state_dict(), model_path)


    model_params_dict = {
        "input_dim": input_dim, 
        "latent_dim": latent_dim, 
        "char_dim": char_dim, 
        "output_dim": output_dim, 
        "dropout": dropout, 
        "KLbeta": KLbeta, 
        "rec_weight": rec_weight, 
        "learning_rate": learning_rate, 
        "model_path": model_path, 
        "max_count": max_count, 
        "exp_traj": exp_traj
    }
    model_params_path = log_model_dir + "model_params_dict.pkl"
    with open(model_params_path, "wb") as fp: 
        pickle.dump(model_params_dict, fp)

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

                # Reconstruct input exploration force/torque traj
                lower_bound, z, y, y_char = model(x, x_char, y_opt, torch_device)
                loss = -sum(lower_bound)

                latent_z_list.append(z)
                label_list.append(x_char)

                total_loss += loss.item()
                KL_loss += lower_bound[0]
                rec_loss += lower_bound[1]

        print(f"Test Loss: {(avg_total_loss):.4f}")

        latent_z_tensor = torch.cat(latent_z_list, dim=0)
        latent_z_np = latent_z_tensor.cpu().detach().numpy()

        label_tensor = torch.cat(label_list, dim=0)
        label_np = label_tensor.cpu().detach().numpy()

        print(label_np.shape)

        latent_z_df = pd.DataFrame(latent_z_np, columns = range(latent_dim))
        latent_z_df['dynamic friction'] = pd.DataFrame(label_np)
        print(latent_z_df)

        n_c = 3
        pca = PCA(n_components = n_c)
        data = latent_z_df.loc[:,range(latent_dim)]
        df_pca = pd.DataFrame(pca.fit_transform(data)) 

        df_pca['dynamic friction'] = latent_z_df['dynamic friction']

        # contribution rate
        pca_cont = pca.explained_variance_ratio_
        eval_latent_pca_table = wandb.Table(columns=['Variable', 'PC0', 'PC1', 'PC2'])
        eval_latent_pca_table.add_data('PCA latent space (eval data)', pca_cont[0], pca_cont[1], pca_cont[2])

        for pc_x in range(2):
            for pc_y in range(2):
                if pc_x < pc_y:
                    #print(pc_x)
                    #print(pc_y)
                    sns.scatterplot(data=df_pca, x=pc_x, y=pc_y, hue='dynamic friction').set(title='VAE latent pca (eval data)')
                    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
                    plot_path = os.path.join(log_fig_dir, 'eval_VAE_latent_pca'+ str(pc_x) + str(pc_y) + '.png')
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
