#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for comparing generator outputs at two checkpoints.
@author: andreyvlasenko
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from noise_fun import noise_fun
from vae_model import VariationalAttentionModel
from vae_utilities_cpu import load_checkpoint
from scipy.stats import pearsonr, wasserstein_distance, entropy

sample_rate = 16000
seq_len = 120000
n_channels = 2
learning_rate = 0.0000051
checkpoint_folder_load = "../../checkpoints/"
noise_dim = seq_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resume_from_epoch = [540, 690]

# Prepare generator
generator = VariationalAttentionModel(sound_channels=n_channels, seq_len=seq_len).to(device)
generator.eval()
optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Generate noise (fixed seed for reproducibility)
torch.manual_seed(42)
noise_g = noise_fun(batch_size=1, n_channels=n_channels, seq_len=noise_dim, device=device)

outputs = []
mus = []
logvars = []

for epoch in resume_from_epoch:
    checkpoint_path = os.path.join(checkpoint_folder_load, f"generator_epoch_{epoch}.pt")
    load_checkpoint(checkpoint_path, generator, optimizer)
    with torch.no_grad():
        fake_music_g, mu_g, logvar_g = generator(noise_g)
        outputs.append(fake_music_g.cpu().numpy().squeeze())
        mus.append(mu_g.cpu().numpy())
        logvars.append(logvar_g.cpu().numpy())

# Convert to numpy arrays of shape (n_channels, seq_len)
music_a = outputs[0]
music_b = outputs[1]

# Flatten for channelwise analysis
music_a_flat = music_a.reshape(-1)
music_b_flat = music_b.reshape(-1)

print(f"\nGenerated music shapes: {music_a.shape}, {music_b.shape}")

# Pearson correlation coefficient
corr = np.corrcoef(music_a_flat, music_b_flat)[0, 1]
print(f"Pearson correlation between checkpoint {resume_from_epoch[0]} and {resume_from_epoch[1]}: {corr:.4f}")

# Wasserstein (Earth Mover's) distance (per channel)
wd = []
for ch in range(n_channels):
    wd_val = wasserstein_distance(music_a[ch], music_b[ch])
    wd.append(wd_val)
    print(f"Wasserstein distance (channel {ch}): {wd_val:.4f}")

# Jensen-Shannon divergence (histogram-based, per channel)
jsd = []
for ch in range(n_channels):
    hist_a, bins = np.histogram(music_a[ch], bins=100, density=True)
    hist_b, _ = np.histogram(music_b[ch], bins=bins, density=True)
    m = 0.5 * (hist_a + hist_b)
    # Add small value to avoid log(0)
    js_a = entropy(hist_a + 1e-8, m + 1e-8)
    js_b = entropy(hist_b + 1e-8, m + 1e-8)
    js = 0.5 * (js_a + js_b)
    jsd.append(js)
    print(f"Jensen-Shannon divergence (channel {ch}): {js:.4f}")

# Statistics
for idx, music in enumerate([music_a, music_b]):
    print(f"\nCheckpoint {resume_from_epoch[idx]} statistics:")
    for ch in range(n_channels):
        print(
            f"  Channel {ch}: mean={music[ch].mean():.4f}, std={music[ch].std():.4f}, min={music[ch].min():.4f}, max={music[ch].max():.4f}"
        )

# Visualization
plt.figure(figsize=(12, 6))
for ch in range(n_channels):
    plt.subplot(n_channels, 2, ch * 2 + 1)
    plt.plot(music_a[ch][:1000], label=f'Checkpoint {resume_from_epoch[0]}')
    plt.plot(music_b[ch][:1000], label=f'Checkpoint {resume_from_epoch[1]}', alpha=0.7)
    plt.title(f'Waveforms (Channel {ch})')
    plt.legend()
    plt.subplot(n_channels, 2, ch * 2 + 2)
    plt.hist(music_a[ch], bins=100, alpha=0.5, label=f'Checkpoint {resume_from_epoch[0]}', density=True)
    plt.hist(music_b[ch], bins=100, alpha=0.5, label=f'Checkpoint {resume_from_epoch[1]}', density=True)
    plt.title(f'Histograms (Channel {ch})')
    plt.legend()
plt.tight_layout()
plt.show()