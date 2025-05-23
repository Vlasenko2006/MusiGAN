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
from vae_train_gan import train_vae_gan, pretrain_generator, pretrain_discriminator  # Updated GAN training functions
from utilities import load_checkpoint, save_checkpoint
from split_and_append_chunks import split_and_append_chunks
from load_decoder_weights import load_decoder_weights

# Constants
dataset_folder = "../dataset"
batch_size = 8
epochs = 30000
pretrain_epochs = 8  # Pretraining epochs for the generator
sample_rate = 16000
seq_len = 120000
learning_rate = 0.0000051 
learning_rate_disc = .0001
n_channels = 2

accumulation_steps = 8 * 2 * 4
checkpoint_folder_load = "checkpoints_gan_music_chunk_100"
checkpoint_folder = "checkpoints_gan_music_chunk_100"
music_out_folder = "music_out_gan_music_chunk_100"
noise_dim = seq_len  # Dimensionality of noise input to the generator
smoothing = 0.05  # Label smoothing for discriminator
save_after_nepochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resume_from_checkpoint = None  # Set this to the checkpoint path if resuming

# Load datasets
data_sample1 = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
data_sample2 = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)
data_sample1 = split_and_append_chunks(data_sample1)
data_sample2 = split_and_append_chunks(data_sample2)

dataset = np.append(data_sample1, data_sample2, axis=0)
print("dataset.shape = ", dataset.shape)

audio_dataset = AudioDataset(dataset)
train_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)


# Model initialization
input_dim = dataset[0].shape[-1]  # Infer input dimension from data
print("input_dim = ", input_dim)

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
start_epoch = 1
if resume_from_checkpoint:
    generator_path = checkpoint_folder_load + "/generator_epoch_" + str(resume_from_checkpoint) + ".pt"
    start_epoch = load_checkpoint(generator_path, generator, g_optimizer)
    discriminator_path = checkpoint_folder_load + "/discriminator_epoch_" + str(resume_from_checkpoint) + ".pt"
    start_epoch = load_checkpoint(discriminator_path, discriminator, d_optimizer)

# Pretrain the generator
load_decoder_weights(generator, "../Lets_Rock/checkpoints_trans5/model_epoch_20.pt")
if start_epoch == 1:
    print("Pretraining the generator...")
    pretrain_generator(generator, 
                       train_loader, 
                       g_optimizer, 
                       pretrain_criterion, 
                       device, 
                       noise_dim, 
                       n_channels,
                       pretrain_epochs)
    pretrain_discriminator(discriminator,
                           generator,
                           train_loader,
                           g_optimizer,
                           device,
                           noise_dim,
                           n_channels=2,
                           pretrain_epochs=2,
                           smoothing=0.1)



# Train GAN
print("Starting GAN training...")
train_vae_gan(generator, 
          discriminator, 
          g_optimizer, 
          d_optimizer, 
          train_loader, 
          start_epoch, 
          epochs, 
          device, 
          noise_dim,
          n_channels, 
          sample_rate, 
          checkpoint_folder, 
          music_out_folder, 
          accumulation_steps=accumulation_steps, 
          smoothing=smoothing, 
          save_after_nepochs=save_after_nepochs)
