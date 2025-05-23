#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:05 2025

@author: andrey
"""


import torch
import torch.nn as nn


def compute_chunk_values(input_sequence, num_chunks, reduction="max"):
    """
    Splits the input sequence into N chunks along the seq_len dimension, computes either the maximal or mean value
    for each chunk, and assembles a new sequence of these values.

    Args:
        input_sequence (torch.Tensor): The input sequence (3D tensor) with dimensions [batch_size, seq_len, input_dim].
        num_chunks (int): The number of chunks to split the seq_len dimension into.
        reduction (str): The reduction method to apply to each chunk. 
                         Choose between "max" (default) or "mean".

    Returns:
        torch.Tensor: The new sequence of reduced values with dimensions [batch_size, new_seq_len, input_dim].
                      The new_seq_len is equal to `min(num_chunks, seq_len)`.
    """
    if reduction not in ["max", "mean"]:
        raise ValueError("Reduction must be either 'max' or 'mean'.")

    # Ensure input_sequence is a 3D tensor
    if input_sequence.ndim != 3:
        raise ValueError("Input sequence must be a 3D tensor with dimensions [batch_size, seq_len, input_dim].")

    batch_size, seq_len, input_dim = input_sequence.shape

    # Adjust num_chunks if seq_len is smaller than num_chunks
    num_chunks = min(num_chunks, seq_len)

    # Compute the size of each chunk
    chunk_size = seq_len // num_chunks

    # Handle remaining elements that don't fit evenly into chunks
    leftover = seq_len % num_chunks

    # Split the input sequence into chunks
    reshaped = input_sequence[:, :chunk_size * num_chunks, :].view(batch_size, num_chunks, chunk_size, input_dim)

    # Add leftover elements to the last chunk, if any
    if leftover > 0:
        leftover_chunk = input_sequence[:, -leftover:, :].mean(dim=1, keepdim=True)  # Handle leftover elements
        reshaped = torch.cat([reshaped, leftover_chunk.unsqueeze(1)], dim=1)  # Add to reshaped tensor

    # Compute the reduced value for each chunk along the chunk_size dimension
    if reduction == "max":
        reduced_values, _ = torch.max(reshaped, dim=2)  # Max value along each chunk
    elif reduction == "mean":
        reduced_values = torch.mean(reshaped, dim=2)  # Mean value along each chunk

    # The output will have dimensions [batch_size, num_chunks, input_dim]
    return reduced_values






def vae_loss_function(reconstructed, inputs, mu, logvar, criterion=nn.MSELoss()):
    """
    Compute the VAE loss as the sum of the reconstruction loss and the KL-divergence.
    """
    # Reconstruction loss
    reconstruction_loss = criterion(reconstructed, inputs)

    # KL-divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / inputs.size(0)  # Normalize by batch size

    # Total loss (with optional weighting for the KL term)
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss
