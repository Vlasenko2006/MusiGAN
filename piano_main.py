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
from piano_model import VariationalAttentionModel, AudioDataset
from discriminator import Discriminator_with_mdisc # Import the discriminator model
from piano_train import train_vae_gan  # Updated GAN training functions
from utilities import load_checkpoint
from split_and_append_chunks import split_and_append_chunks
from update_checkpoint import update_checkpoint
from piano_pretrain import pretrain_generator, pretrain_discriminator
from config_reader import load_config_from_yaml



# Load configuration at the start of your script
yaml_filepath = "config/config.yaml"
config = load_config_from_yaml(yaml_filepath)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load datasets
data_sample1 = np.load(os.path.join(config["dataset_folder"], "training_set.npy"), allow_pickle=True)
data_sample2 = np.load(os.path.join(config["dataset_folder"], "validation_set.npy"), allow_pickle=True)
data_sample1 = data_sample1/np.max(data_sample1)
data_sample2 = data_sample2/np.max(data_sample2)
data_sample1 = split_and_append_chunks(data_sample1)
data_sample2 = split_and_append_chunks(data_sample2)

dataset = np.append(data_sample1, data_sample2, axis=0)

print("dataset.shape = ", dataset.shape)

audio_dataset = AudioDataset(dataset)
train_loader = DataLoader(audio_dataset, batch_size=config["batch_size"], shuffle=True)


# Model initialization
input_dim = dataset[0].shape[-1]  # Infer input dimension from data
print("input_dim = ", input_dim)


# Generator (uses AttentionModel as the base)
generator = VariationalAttentionModel(**config["model"]).to(device)

# Discriminator
discriminator = Discriminator_with_mdisc(input_dim = config["model"]["sound_channels"],
                                         n_channels = config["model"]["sound_channels"], # FIXME!!! Possible duplication
                                         seq_len = config["model"]["seq_len"]
                                         ).to(device)

# Optimizers
g_pretrain_optimizer = torch.optim.Adam(generator.parameters(),
                                        lr=1e-7,
                                        betas=(0.5, 0.999))

g_optimizer = torch.optim.AdamW(generator.parameters(), 
                                lr=config["learning_rate"],
                                betas=(0.5, 0.999))

d_optimizer = torch.optim.AdamW(discriminator.parameters(), 
                                lr=config["learning_rate"],
                                betas=(0.5, 0.999))

# Loss functions
pretrain_criterion = nn.SmoothL1Loss()  # For generator pretraining
gan_criterion = nn.BCEWithLogitsLoss()  # For adversarial training


# Load checkpoint if specified
start_epoch = 1
if config["resume_from_checkpoint"]:
    resume_epoch = config["resume_from_checkpoint"]
    generator_path = os.path.join(
        config["checkpoint_folder_load"], 
        f"generator_epoch_{resume_epoch}.pt"
    )
    
    start_epoch = load_checkpoint(generator_path, generator, g_optimizer)
    
    discriminator_path = os.path.join(
        config["checkpoint_folder_load"],
        f"discriminator_epoch_{resume_epoch}.pt"
    )

    if config["update_discriminator"]:
        start_epoch = update_checkpoint(discriminator_path, discriminator, d_optimizer)
        pretrain_discriminator(discriminator,
                           generator,
                           train_loader,
                           g_optimizer,
                           device,
                           config["noise_dim"],
                           n_channels=config["model"]["sound_channels"],
                           pretrain_epochs=config["pretrain_epochs_d"] ,
                           smoothing=0.0)
    else:
        start_epoch = load_checkpoint(discriminator_path,
                                      discriminator,
                                      d_optimizer)

# Pretrain the generator

if start_epoch == 1:
    #load_decoder_weights(generator, "../Lets_Rock/checkpoints_trans5/model_epoch_20.pt")
    print("Pretraining the generator...")
    pretrain_generator(generator, 
                       train_loader, 
                       g_optimizer, 
                       pretrain_criterion, 
                       device, 
                       config["noise_dim"], 
                       config["model"]["sound_channels"],
                       pretrain_epochs = config["pretrain_epochs_g"] )

    pretrain_discriminator(discriminator,
                           generator,
                           train_loader,
                           g_optimizer,
                           device,
                           config["noise_dim"],
                           n_channels=config["model"]["sound_channels"],
                           pretrain_epochs = config["pretrain_epochs_d"] ,
                           smoothing=0.0)



# Train GAN
print("Starting GAN training...")
train_vae_gan(generator, 
          discriminator, 
          g_optimizer, 
          d_optimizer, 
          train_loader, 
          start_epoch, 
          config['epochs'],     # mind using **kwargs
          device, 
          config["noise_dim"],
          config["model"]["sound_channels"], 
          config["sample_rate"], 
          config['checkpoint_folder'], 
          config["music_out_folder"], 
          accumulation_steps=config["accumulation_steps"], 
          smoothing=config["smoothing"], 
          save_after_nepochs=config["save_after_nepochs"],
          batch_size = config['batch_size'],
          loss_trackers = config["loss_trackers"],
          loss_weights = config["loss_weights"]
          )
