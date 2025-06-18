#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select best GAN-generated music samples using discriminator score.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from vae_model_spectral import VariationalAttentionModel
from discriminator import Discriminator_with_mdisc
from vae_utilities_cpu import load_checkpoint
from noise_fun import noise_fun

from scipy.signal import find_peaks, iirnotch, filtfilt


def visualize_and_filter_parasites(
    signal,
    path_results,
    fs,
    music_band=(400, 700),
    q_factor=30.0,
    peak_height_ratio=0.1,
    show_plots=True,
    return_filtered=True
):
    """
    Visualize FFT spectrum, detect parasitic frequencies outside music_band, and filter them out.

    Parameters:
    - signal: np.ndarray, 1D array of audio samples
    - fs: int, sample rate in Hz
    - music_band: tuple (low, high), frequencies considered as "wanted" (Hz)
    - q_factor: float, notch filter quality factor (higher = narrower notch)
    - peak_height_ratio: float, relative height threshold for peak detection
    - show_plots: bool, whether to display plots
    - return_filtered: bool, whether to return the filtered signal

    Returns:
    - If return_filtered: filtered_signal, removed_freqs
      else: removed_freqs
    """

    # FFT
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, 1/fs)
    fft_magnitude = np.abs(fft_vals)

    # Peak detection for significant frequencies
    peaks, props = find_peaks(fft_magnitude, height=np.max(fft_magnitude)*peak_height_ratio)
    peak_freqs = fft_freqs[peaks]

    # Identify parasitic frequencies (outside music_band)
    parasitic_freqs = [f for f in peak_freqs if f < music_band[0] or f > music_band[1]]

    if show_plots:
        plt.figure(figsize=(10,4))
        plt.plot(fft_freqs, fft_magnitude, label='Original')
        plt.scatter(peak_freqs, fft_magnitude[peaks], color='r', label='Detected Peaks')
        for f in parasitic_freqs:
            plt.axvline(f, color='orange', linestyle='--', alpha=0.5)
        plt.title('FFT Spectrum (Detected Peaks & Parasitics)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, fs // 2)
        plt.legend()
        plt.grid(True)
        plt.savefig(path_results)
        #plt.show()

    # filtered_signal = signal.copy()
    # for f0 in parasitic_freqs:
    #     b, a = iirnotch(f0, q_factor, fs)
    #     filtered_signal = filtfilt(b, a, filtered_signal)

    # if show_plots:
    #     fft_filtered = np.fft.rfft(filtered_signal)
    #     plt.figure(figsize=(10,4))
    #     plt.plot(fft_freqs, fft_magnitude, label='Original')
    #     plt.plot(fft_freqs, np.abs(fft_filtered), label='Filtered')
    #     for f in parasitic_freqs:
    #         plt.axvline(f, color='orange', linestyle='--', alpha=0.4)
    #     plt.title('FFT Spectrum Before and After Notch Filtering')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Magnitude')
    #     plt.xlim(0, fs // 2)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # if return_filtered:
    #     return filtered_signal, parasitic_freqs
    # else:
   


# User parameters
num_samples_to_generate = 10
selection_batch_size = 16
top_k = 5
seq_len = 120000
n_channels = 2
noise_dim = seq_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoints and epoch
checkpoints = {'checkpoint_new9': 238}
checkpoint = 'checkpoint_new9'
resume_from_checkpoint = checkpoints[checkpoint]

# Output directory
outdir = f"music_out/{checkpoint}/"
os.makedirs(outdir, exist_ok=True)

# Load models
generator = VariationalAttentionModel(sound_channels=n_channels, seq_len=seq_len).to(device)
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
        (_, _,_, _, _, _,_, _, _, _, fake_music)= generator(noise)
        d_fake_logits = discriminator(fake_music)
        # For each sample in batch, store (score, sample)
        for i in range(batch):
            score = d_fake_logits[i].item() if d_fake_logits.ndim > 0 else d_fake_logits.item()
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
        path_results_w = os.path.join( outdir, f"{label}_song_{idx}_w_plot.png")
        np.save(npy_path, arr)
        plt.figure(figsize=(10, 2))
        plt.plot(arr[0, :])  # Plot the first channel
        plt.title(f'Sample {idx}, score={score:.6f}')
        plt.savefig(path_results)
        visualize_and_filter_parasites(
            arr[0,:],
            path_results_w,
            fs = 12000,
            music_band=(0, 8000),
            q_factor=30.0,
            peak_height_ratio=0.2,
            show_plots=True,
            return_filtered=False
        )
        print(f"label_{idx}")
        #plt.close()

print('music saved in', outdir)
arr = save_samples(top_songs, "best")
arr = save_samples(worst_songs, "worst")
