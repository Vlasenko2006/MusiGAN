#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:59:41 2025

@author: andrey
"""

import torch
from tqdm import tqdm
from utilities import save_checkpoint, save_sample_as_numpy
from noise_fun import noise_fun

# Helper function to freeze parameters
def freeze_decoder_weights(model):
    """
    Freeze the weights of the decoder in the encoder_decoder model.

    Args:
        model: The encoder_decoder model.
    """
    for name, param in model.encoder_decoder.named_parameters():
        # Check if the parameter belongs to the decoder
        if "decoder" in name:
            param.requires_grad = False


# Function to train GAN for music generation
def train_gan(generator,
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
              accumulation_steps=4,
              smoothing=0.1,
              save_after_nepochs=10):
    """
    Train GAN for music generation.

    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        g_optimizer: Optimizer for the generator.
        d_optimizer: Optimizer for the discriminator.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        start_epoch: Starting epoch for training.
        epochs: Total number of epochs.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        noise_dim: Dimensionality of the noise input to the generator.
        sample_rate: Sampling rate of the music.
        checkpoint_folder: Folder to save model checkpoints.
        music_out_folder: Folder to save generated music samples.
        accumulation_steps: Number of steps for gradient accumulation.
        smoothing: Label smoothing factor for discriminator training.
        save_after_nepochs: Save results after this number of epochs.
    """
    generator.to(device)
    discriminator.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for stability

    freeze_decoder_weights(generator)
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Training phase
        generator.train()
        discriminator.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        for batch_idx, real_music in enumerate(tqdm(train_loader, desc="Training")):
            batch_size = real_music.size(0)
            real_music = real_music.to(device).float()  # Ensure input is in Float precision

            # Add Gaussian noise to real music for discriminator robustness
            noisy_real_music = real_music + 0.05 * torch.randn_like(real_music)

            # Generate fake music
            noise = noise_fun(batch_size = batch_size, n_channels = n_channels, seq_len = noise_dim, device =device) #torch.randn(batch_size, n_channels, noise_dim).to(device).float()
            fake_music = generator(noise)
           

            # Train Discriminator
            real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(device).float()

            d_real_logits = discriminator(noisy_real_music.float())
            d_fake_logits = discriminator(fake_music.detach().float())

            d_loss_real = criterion(d_real_logits, real_labels)
            d_loss_fake = criterion(d_fake_logits, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / accumulation_steps  # Scale loss for accumulation

            d_loss.backward()  # Accumulate gradients

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                d_optimizer.step()
                d_optimizer.zero_grad()

            # Train Generator
            for _ in range(2):  # Train the generator more frequently
                noise = torch.randn(batch_size, n_channels, noise_dim).to(device).float()
                fake_music = generator(noise)

                g_loss = criterion(discriminator(fake_music), real_labels) / accumulation_steps
                g_loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()

            epoch_loss_d += d_loss.item() * accumulation_steps  # Scale back loss
            epoch_loss_g += g_loss.item() * accumulation_steps

        # Log epoch losses
        print(f"Epoch {epoch}/{epochs} - D Loss: {epoch_loss_d/len(train_loader):.8f}, G Loss: {epoch_loss_g/len(train_loader):.8f}")

        # Save generated music and models
        if (epoch % save_after_nepochs == 0):
            print(" ==== saving music samples and models ==== ")
                                 
            save_sample_as_numpy(generator, device, music_out_folder, epoch, noise_dim, n_channels= n_channels, num_samples=10, prefix='')
            save_checkpoint(generator, g_optimizer, epoch, checkpoint_folder, "generator")
            save_checkpoint(discriminator, d_optimizer, epoch, checkpoint_folder, "discriminator")


# Function to pretrain the generator
def pretrain_generator(generator,
                       train_loader,
                       optimizer,
                       criterion, 
                       device,
                       noise_dim,
                       n_channels = 2,
                       pretrain_epochs=20):
    """
    Pretrain the generator with Smooth L1 loss for better initialization.

    Args:
        generator: The generator model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for the generator.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        noise_dim: Dimensionality of the noise input to the generator.
        pretrain_epochs: Number of epochs for pretraining.
    """
    generator.to(device)
    criterion = torch.nn.SmoothL1Loss()
    freeze_decoder_weights(generator)
    for epoch in range(pretrain_epochs):
        print(f"Pretraining Generator Epoch {epoch + 1}/{pretrain_epochs}")
        generator.train()
        epoch_loss = 0.0

        for real_music in tqdm(train_loader, desc="Pretraining"):
            batch_size = real_music.size(0)
            real_music = real_music.to(device).float()  # Ensure input is in Float precision

            # Generate random noise
            noise = noise_fun(batch_size = batch_size, n_channels = n_channels, seq_len = noise_dim, device =device) #torch.randn(batch_size, n_channels, noise_dim).to(device).float()

            # Generate fake music
            fake_music = generator(noise)

            # Calculate Smooth L1 loss between fake and real music
            loss = criterion(fake_music, real_music)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Pretraining Generator Epoch {epoch + 1}/{pretrain_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")
