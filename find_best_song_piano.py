#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select best GAN-generated music samples using discriminator score.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from piano_model import VariationalAttentionModel
from discriminator import Discriminator_with_mdisc
from vae_utilities_cpu import load_checkpoint
from noise_fun import noise_fun
import torch.nn.functional as F




def create_plot_with_two_subfigures(arr1, arr2, idx, score, path_results):
    """
    Create a plot with 2 subfigures.

    Args:
        arr (numpy.ndarray): Array containing data to be plotted (shape: [channels, samples]).
        idx (int): Sample index to annotate in the title.
        score (float): Score value to annotate in the title.
        path_results (str): File path to save the generated plot.
    """
    plt.figure(figsize=(12, 4))  # Adjust figsize for better readability

    # Subfigure 1: Plot the first channel
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(arr1)  # Plot the first channel
    plt.title(f'Sample {idx} - Estimated, score={score:.6f}')

    # Subfigure 2: Plot the second channel
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(arr2)  # Plot the second channel
    plt.title(f'Sample {idx} - True, score={score:.6f}')

    # Save the figure
    plt.tight_layout()
    plt.savefig(path_results)
    print(f"Plot saved for label_{idx}")



# User parameters
num_samples_to_generate = 1000
selection_batch_size = 16
top_k = 25
seq_len = 120000
n_channels = 2
seq_len = 16500
latent_seq_len = 500
use_gaussians = True

noise_dim = 3000 * 4 * 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoints and epoch
checkpoints = {'checkpoint_piano_hards4': 21 } #651 149
checkpoint = 'checkpoint_piano_hards4'
resume_from_checkpoint = checkpoints[checkpoint]

# Output directory
outdir = f"music_out/{checkpoint}/"
valid_set = np.load('../dataset_piano/validation_set.npy')

os.makedirs(outdir, exist_ok=True)


# Load models
generator = VariationalAttentionModel(sound_channels=n_channels, seq_len=seq_len, latent_seq_len = latent_seq_len, use_gaussians = use_gaussians).to(device)
discriminator = Discriminator_with_mdisc(input_dim=n_channels, n_channels=n_channels, seq_len=seq_len).to(device)
generator_path = os.path.join(f"{checkpoint}/", f"generator_epoch_{resume_from_checkpoint}.pt")
discriminator_path = os.path.join(f"{checkpoint}/", f"discriminator_epoch_{resume_from_checkpoint}.pt")
g_optimizer = torch.optim.Adam(generator.parameters())
d_optimizer = torch.optim.Adam(discriminator.parameters())

load_checkpoint(generator_path, generator, g_optimizer)
load_checkpoint(discriminator_path, discriminator,d_optimizer)
generator.eval()
discriminator.eval()



# Collect samples and scores
best_songs = []
with torch.no_grad():
    total = 0
    while total < num_samples_to_generate:
        batch = min(selection_batch_size, num_samples_to_generate - total)
        noise = noise_fun(batch_size=batch, n_channels=n_channels, seq_len=noise_dim, device=device)
        (_, _, fake_music)= generator(noise)
        d_fake_logits = discriminator(fake_music)
        # For each sample in batch, store (score, sample)
        for i in range(batch):
            score = F.sigmoid(d_fake_logits[i] if d_fake_logits.ndim > 0 else d_fake_logits).item()
            best_songs.append((score, fake_music[i].cpu()))
        total += batch
        print(f"Generated {total}/{num_samples_to_generate} samples", end='\r')

# Sort and select top_k
best_songs.sort(key=lambda x: x[0])
top_songs = best_songs[:top_k]
worst_songs = best_songs[-top_k:]



#%%
# Save top and worst samples
def save_samples(songs, label):
    for idx, (score, sample) in enumerate(songs, 1):
        arr = sample.numpy()
        npy_path = os.path.join(outdir, f"{label}_song_{idx}.npy")
        path_results = os.path.join( outdir, f"{label}_song_{idx}_plot.png")
        np.save(npy_path, arr)
        create_plot_with_two_subfigures(arr[0, :], valid_set[idx,1,1,:], idx, score, path_results)

print('music saved in', outdir)
arr = save_samples(top_songs, "best")
arr = save_samples(worst_songs, "worst")
