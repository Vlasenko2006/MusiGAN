#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:05 2025

@author: andrey
"""

import os
import numpy as np
import torch


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
