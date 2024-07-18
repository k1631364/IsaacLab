import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt


class VAE(nn.Module):
	def __init__(self, x_dim, z_dim, char_dim, y_dim, dropout, beta, alpha):
	
		"""
		Constructor
		
		Args: 
			x_dim: input data dimension
			z_dim: latent space dimension
			char_dim: characteristics dimension
			y_dim: output data dimension
			dropout: VAE dropout
			beta: weight on KL divergence
		
		Returns: 
			None. 
			
		"""
		
		super(VAE, self).__init__()
		self.x_dim = x_dim	# Input data dimension
		self.z_dim = z_dim	# Latent space dimension
		self.char_dim = char_dim
		self.y_dim = y_dim	# Output data dimension
		self.dropout = dropout	# Dropout rate, larger = prevent more from overfitting
		self.beta = beta	# Strength of KL
		self.alpha = alpha
		
		self.enc_fc1 = nn.Linear(x_dim, 20)		# First encoder layer
		self.enc_fc2_mean = nn.Linear(20, z_dim)	# Approximated posterior distribution mean
		self.enc_fc2_var = nn.Linear(20, z_dim)	# Approximated posterior distribution variance
		self.dec_fc1 = nn.Linear(z_dim, 20)		# First decoder layer
		self.dec_drop1 = nn.Dropout(p=dropout)	# Dropout before the last layer to prevent overfitting
		self.dec_fc2 = nn.Linear(20, y_dim)		# Second decoder layer
		
	def encoder(self, x):
		x = x.view(-1, self.x_dim)
		x = F.relu(self.enc_fc1(x))
		mean = self.enc_fc2_mean(x)
		log_var = self.enc_fc2_var(x)
		return mean, log_var
		
	def sample_z(self, mean, log_var, device):
		epsilon = torch.rand(mean.shape, device=device)
		return mean + epsilon * torch.exp(0.5*log_var)
		
	def decoder(self, z):
		y = F.relu(self.dec_fc1(z))
		y = self.dec_drop1(y)
		y = torch.sigmoid(self.dec_fc2(y))
		return y
		
	def forward(self, x, x_char, y_opt, device):
		
		mean, log_var = self.encoder(x)
		delta = 1e-8
		KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
		z = self.sample_z(mean, log_var, device)
		y = self.decoder(z)
		KL = self.beta*KL

		# Reconstruction loss
		# Assume Gaussian distribution for decoder = use Mean Squared Error (MSE)
		# Multiply by # of data points to scale woth KL divergence loss
		loss_fnc = nn.MSELoss()
		reconstruction = loss_fnc(y,y_opt)*(y.shape[0]*y.shape[1])
		reconstruction = reconstruction * (-1) * self.alpha

		# Reconstruction loss assuming Bernoulli distribution for decoder = use Binary Cross Entropy (BCE) loss
		#reconstruction = torch.sum(y_opt * torch.log(y + delta) + (1 - y_opt) * torch.log(1 - y + delta))
		#print(reconstruction)
		#print(KL)

		label_loss = 0
		y_char = x_char	# no characteristics prediction

		return [KL, reconstruction], z, y, y_char
		
	
		
		
		
		