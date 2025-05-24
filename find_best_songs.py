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
from torch.utils.data import DataLoader
from vae_model import VariationalAttentionModel, AudioDataset
from discriminator import Discriminator  # Import the discriminator model
from vae_utilities_cpu import load_checkpoint
from split_and_append_chunks import split_and_append_chunks
from noise_fun import noise_fun
import matplotlib.pyplot as plt


# Constants

batch_size = 8
epochs = 30000
pretrain_epochs = 8  # Pretraining epochs for the generator
sample_rate = 16000
seq_len = 120000
learning_rate = 0.0000051 #*0.25 
learning_rate_disc = .0001 #*0.25
n_channels = 2

accumulation_steps = 8 * 2 * 4 #* 4
checkpoint_folder_load = "../../checkpoints/"
checkpoint_folder = "checkpoints_gan_music_double"
music_out_folder = "music_out_gan_music_double"
noise_dim = seq_len  # Dimensionality of noise input to the generator
smoothing = 0.05  # Label smoothing for discriminator
save_after_nepochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resume_from_checkpoint = 690  # Set this to the checkpoint path if resuming


# Generator (uses AttentionModel as the base)
generator = VariationalAttentionModel(sound_channels=n_channels, seq_len=seq_len).to(device)

# Discriminator
discriminator = Discriminator(input_dim=n_channels, n_channels=n_channels, seq_len=seq_len).to(device)

# Optimizers
g_pretrain_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_disc, betas=(0.5, 0.999))

# Loss functions
pretrain_criterion = nn.SmoothL1Loss()  # For generator pretraining
gan_criterion = nn.BCEWithLogitsLoss()  # For adversarial training

# Load checkpoint if specified

generator_path = checkpoint_folder_load + "/generator_epoch_" + str(resume_from_checkpoint) + ".pt"
start_epoch = load_checkpoint(generator_path, generator, g_optimizer)
discriminator_path = checkpoint_folder_load + "/discriminator_epoch_" + str(resume_from_checkpoint) + ".pt"
start_epoch = load_checkpoint(discriminator_path, discriminator, d_optimizer)

# Pretrain the generator

best_songs = []

for i in range(0, 1000):
    print('i = ', i)
    with torch.no_grad():
        noise = noise_fun(batch_size=1, n_channels=n_channels, seq_len=noise_dim, device=device)
        fake_music, _, _ = generator(noise)
        d_fake_logits = discriminator(fake_music)
        # Use ONES as label: we want fakes that look real to D!
        d_loss_fake = gan_criterion(d_fake_logits, torch.ones_like(d_fake_logits))
        score = d_loss_fake.item() if hasattr(d_loss_fake, "item") else float(d_loss_fake)
        best_songs.append((score, fake_music.cpu()))

best_songs.sort(key=lambda x: x[0])
top_10 = best_songs[:10]
top_10_songs = [sample for score, sample in top_10]
top_10_scores = [score for score, sample in top_10]

print("Top 10 d_loss_fake scores (lowest = most realistic):")
for i, score in enumerate(top_10_scores, 1):
    print(f"{i}: {score:.6f}")

os.makedirs("best songs", exist_ok=True)
for idx, (score, sample) in enumerate(top_10, 1):
    arr = sample.squeeze().numpy()
    npy_path = os.path.join("best songs", f"song_{idx}.npy")
    np.save(npy_path, arr)
    plt.plot(arr[0,:])
    plt.title(f'Sample {idx}, loss={score:.6f}')
    plt.show()
    # print(f"Saved: {npy_path}")