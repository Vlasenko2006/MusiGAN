#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 07:40:52 2025

@author: andrey
"""

import re
import matplotlib.pyplot as plt

def parse_gan_log(log_text):
    """
    Parse GAN training log and extract epoch, D Loss, G Loss, Mean Recon.
    Returns lists of epochs, d_losses, g_losses, mean_recons.
    """
    # Regex to match lines like:
    # "Epoch 690/30000 - D Loss: 0.28733735, G Loss: 2.27755853, Mean Recon: 0.50223107"
    pattern = re.compile(
        r"Epoch (\d+)/\d+ - D Loss: ([0-9.]+), G Loss: ([0-9.]+), Mean Recon: ([0-9.]+)"
    )
    epochs, d_losses, g_losses, mean_recons = [], [], [], []
    for match in pattern.finditer(log_text):
        epoch = int(match.group(1))
        d_loss = float(match.group(2))
        g_loss = float(match.group(3))
        mean_recon = float(match.group(4))
        epochs.append(epoch)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        mean_recons.append(mean_recon)
    return epochs, d_losses, g_losses, mean_recons

def plot_gan_losses(epochs, d_losses, g_losses, mean_recons):
    """
    Plot D Loss, G Loss, and Mean Recon over epochs.
    """
    plt.figure(figsize=(12,6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, d_losses, label="D Loss", color='tab:blue')
    plt.plot(epochs, g_losses, label="G Loss", color='tab:orange')
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Losses")

    plt.subplot(2, 1, 2)
    plt.plot(epochs, mean_recons, label="Mean Recon", color='tab:green')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Recon")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  #  Example: Read from a file named 'training.log'
    with open("training.log") as f:
        log_text = f.read()

 
    epochs, d_losses, g_losses, mean_recons = parse_gan_log(log_text)
    plot_gan_losses(epochs, d_losses, g_losses, mean_recons)