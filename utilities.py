#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:05 2025

@author: andrey
"""

import os
import numpy as np
import torch
from noise_fun import noise_fun

# Generate and save music samples as NumPy files
def save_sample_as_numpy(generator, device, music_out_folder, epoch, noise_dim, n_channels=2, num_samples=10, prefix=''):
    """
    Generate and save music samples using the generator model.

    Args:
        generator: The generator model.
        device: Device to use for computation (e.g., 'cuda' or 'cpu').
        music_out_folder: Folder to save the generated music samples.
        epoch: Current epoch number (used for naming the files).
        noise_dim: Dimensionality of the noise input to the generator.
        n_channels: Number of audio channels (default is 2 for stereo audio).
        num_samples: Number of samples to generate (default is 10).
        prefix: Prefix to add to the saved file names (default is '').
    """
    generator.eval()  # Set the generator to evaluation mode
    os.makedirs(music_out_folder, exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise
            noise = noise_fun(batch_size = 1, n_channels = n_channels, seq_len = noise_dim, device =device) #torch.randn(1, n_channels, noise_dim).to(device)  # Batch size = 1

            # Generate music
            generated_music = generator(noise)

            # Convert to NumPy and save
            generated_np = generated_music.cpu().numpy()[0]  # Take the first sample
            np.save(os.path.join(music_out_folder, prefix + f"sample_{i + 1}_epoch_{epoch}.npy"), generated_np)

            print(f"Saved generated music sample {i + 1} as NumPy file for epoch {epoch}.")

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_folder, model_name="model"):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number.
        checkpoint_folder: Folder to save the checkpoint file.
        model_name: Name of the model (default is 'model').
    """
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"{model_name}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Load model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        start_epoch: The epoch number to resume training from.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from epoch 1.")
        return 1  # Start from the first epoch if no checkpoint is found
