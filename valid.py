#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:37:20 2025

@author: andreyvlasenko
"""

import os
import numpy as np
import torch
import torch.nn as nn
from vae_model import VariationalAttentionModel
from vae_utilities_cpu import load_checkpoint
import matplotlib.pyplot as plt
from noise_fun import noise_fun


# Constants
dataset_folder = "../dataset"
batch_size = 2
epochs = 30000
pretrain_epochs = 5  # Pretraining epochs for the generator
sample_rate = 16000
seq_len = 120000
learning_rate = 0.00001 * .25 * 0.5
n_channels = 2




accumulation_steps = 8 * 2 * 2
checkpoint_folder_load = "checkpoints_gan_music_chunk/"
checkpoint_folder = "checkpoints_gan_music_vae3"
music_out_folder = "music_out_gan_music_vae3"
noise_dim = 150 #seq_len  # Dimensionality of noise input to the generator
smoothing = 0.  # Label smoothing for discriminator
save_after_nepochs = 70

device = torch.device("cpu")
resume_from_checkpoint = 70  # Set this to the checkpoint path if resuming



# Model initialization


# Generator (uses AttentionModel as the base)
generator = VariationalAttentionModel(sound_channels= n_channels, seq_len=120000).to(device)


# Optimizers
g_pretrain_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


gan_criterion = nn.BCEWithLogitsLoss()  # For adversarial training

# Load checkpoint if specified
start_epoch = 1
if resume_from_checkpoint:
    generator_path = checkpoint_folder_load + "/generator_epoch_" + str(resume_from_checkpoint ) +".pt"
    start_epoch = load_checkpoint(generator_path, generator, g_optimizer)
    
noise = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim)
fake_music1, mu, logvar = generator(noise)
fake_music2, mu, logvar = generator(noise)

print("Sample 1", fake_music1[0,0,:10])
print("Sample 2", fake_music1[1,0,:10])

print("diff1", fake_music1[1,0,:10] - fake_music1[0,0,:10])
print("diff2", fake_music1[0,0,:10] - fake_music2[0,0,:10])


#%%
#plt.plot(fake_music[1,1,:].detach().numpy())


