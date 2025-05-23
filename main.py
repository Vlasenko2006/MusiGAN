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
from model import AttentionModel
from smart_discriminator import Discriminator  # Import the discriminator model
from train_gan import train_gan, pretrain_generator  # Updated GAN training functions
from utilities import load_checkpoint, save_checkpoint
from split_and_append_chunks import split_and_append_chunks
from load_decoder_weights import load_decoder_weights

# Constants
dataset_folder = "../dataset"
batch_size = 8
epochs = 30000
pretrain_epochs = 3  # Pretraining epochs for the generator
sample_rate = 16000
seq_len = 120000
learning_rate = 0.004
n_channels = 2




accumulation_steps = 4 *2 *4
ref_checkpoint_folder = "checkpoints_gan_music_freese2"
checkpoint_folder = "checkpoints_gan_music_freese2"
music_out_folder = "music_out_gan_music_freese2"
noise_dim = seq_len  # Dimensionality of noise input to the generator
smoothing = 0.05  # Label smoothing for discriminator
save_after_nepochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resume_from_checkpoint = None #150  # Set this to the checkpoint path if resuming

# Load datasets
data_sample1 = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
data_sample2 = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)
data_sample1 = split_and_append_chunks(data_sample1)
data_sample2 = split_and_append_chunks(data_sample2)

dataset = np.append(data_sample1,data_sample2, axis = 0)

print("dataset.shape = ", dataset.shape)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
input_dim = dataset[0][0].shape[-1]  # Infer input dimension from data
print("input_dim = ", input_dim)

# Generator (uses AttentionModel as the base)
generator = AttentionModel(sound_channels= n_channels, seq_len=120000).to(device)

# Discriminator
discriminator = Discriminator(input_dim=n_channels, n_channels=n_channels, seq_len=120000).to(device)

# Optimizers
g_pretrain_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss functions
pretrain_criterion = nn.SmoothL1Loss()  # For generator pretraining
gan_criterion = nn.BCEWithLogitsLoss()  # For adversarial training

# Load checkpoint if specified
start_epoch = 1
if resume_from_checkpoint:
    generator_path = ref_checkpoint_folder + "/generator_epoch_" + str(resume_from_checkpoint ) +".pt"
    start_epoch = load_checkpoint(generator_path, generator, g_optimizer)
    discriminator_path = ref_checkpoint_folder + "/discriminator_epoch_" + str(resume_from_checkpoint ) + ".pt"
    start_epoch = load_checkpoint(discriminator_path, discriminator, d_optimizer)



load_decoder_weights(generator, "../Lets_Rock/checkpoints_trans5/model_epoch_20.pt")
# Pretrain the generator
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

# Train GAN
#load_decoder_weights(generator, "../Lets_Rock/checkpoints_trans5/model_epoch_20.pt")
print("Starting GAN training...")
train_gan(generator, 
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
