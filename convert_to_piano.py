#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:05:35 2025

@author: andreyvlasenko
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

ndev = 25
keys = list(range(27, 60))
def piano_key_intervals(keys: List[int], ndev: float) -> Dict[int, Tuple[float, float]]:
    intervals = {}
    for n in keys:
        freq = 2 ** ((n - 49) / 12) * 440
        intervals[n] = (freq - ndev, freq + ndev)
    return intervals

def split_piano_keys_loudness(
    signals: torch.Tensor,
    fs: int = 12000,
    keys: List[int] = keys,
    ndev: float = ndev,
    min_note_duration: float = 0.05,
    loudness_threshold_ratio: float = 0.05
) -> torch.Tensor:
    batch, n_channels, seq_len = signals.shape
    interval_len = int(round(min_note_duration * fs))
    n_intervals = int(np.ceil(seq_len / interval_len))
    n_keys = len(keys)

    pad_len = n_intervals * interval_len - seq_len
    if pad_len > 0:
        signals = torch.nn.functional.pad(signals, (0, pad_len))

    bands = piano_key_intervals(keys, ndev)
    fft_len = signals.shape[-1]
    signals_np = signals.detach().cpu().numpy()
    fft_vals = np.fft.rfft(signals_np, n=fft_len, axis=-1)
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)

    key_band_signals = []
    for idx, key in enumerate(keys):
        # --- Gaussian frequency mask ---
        freq_center = (bands[key][0] + bands[key][1]) / 2
        std = 0.5 * ndev
        # Gaussian mask centered at freq_center
        gauss_mask = np.exp(-0.5* ((freqs - freq_center) / std) ** 2)
        gauss_mask = gauss_mask.reshape(1, 1, -1)
        band_fft = fft_vals * gauss_mask
        band_signal = np.fft.irfft(band_fft, n=fft_len, axis=-1)
        key_band_signals.append(band_signal)
    key_band_signals = np.stack(key_band_signals, axis=0)

    loudness = []
    for t in range(n_intervals):
        start = t * interval_len
        end = start + interval_len
        seg = key_band_signals[..., start:end]
        rms = np.sqrt(np.mean(seg**2, axis=-1))
        loudness.append(rms)
    loudness = np.stack(loudness, axis=0)
    loudness = np.transpose(loudness, (2, 3, 1, 0))

    max_loudness = loudness.max()
    threshold = loudness_threshold_ratio * max_loudness
    loudness[loudness < threshold] = 0.0

    loudness_tensor = torch.from_numpy(loudness).float().to(signals.device)
    return loudness_tensor

def from_loudness_to_signal(
    loudness_tensor: torch.Tensor,
    fs: int = 12000,
    keys: List[int] = keys,
    ndev: float = ndev,
    min_note_duration: float = 0.05,
    seq_len: int = 120000
) -> torch.Tensor:
    device = loudness_tensor.device
    batch, n_channels, n_keys, n_intervals = loudness_tensor.shape
    interval_len = int(round(min_note_duration * fs))
    total_len = n_intervals * interval_len

    t = torch.arange(interval_len, device=device).float() / fs

    bands = piano_key_intervals(keys, ndev)
    key_freqs = torch.tensor([((fmin + fmax) / 2) for (fmin, fmax) in bands.values()], device=device).float()

    signals = torch.zeros((batch, n_channels, total_len), device=device)

    for interval_idx in range(n_intervals):
        loudness = loudness_tensor[:, :, :, interval_idx]
        phases = 2 * np.pi * key_freqs[:, None] * t[None, :]
        sinusoids = torch.sin(phases)
        loudness_exp = loudness.unsqueeze(-1)
        note_waves = loudness_exp * sinusoids
        interval_wave = note_waves.sum(dim=2)
        start = interval_idx * interval_len
        end = start + interval_len
        signals[:, :, start:end] += interval_wave

    if total_len > seq_len:
        signals = signals[:, :, :seq_len]
    return signals.float()


# Example usage:
# python piano_loudness_process.py input.npy output.npy

input_path = "../dataset/validation_set.npy"
output_path = "../dataset_piano/validation_set.npy"

# Load numpy array (shape [batch, n_channels, seq_len])
np_signal = np.load(input_path)
signals = torch.from_numpy(np_signal).float()
#signals = signals[:10,...]

fs = 12000
min_note_duration = 0.02
loudness_threshold_ratio = 0.025
notes = int((keys[-1] - keys[0] + 1 ) * 10 / min_note_duration)

loudness_tensor = torch.zeros(len(signals),2,2, notes)

# Convert to loudness tensor
print("Processing the data")
for i in range(0,2):
    aux = split_piano_keys_loudness(
        signals[:,i,:,:].squeeze(), fs=fs, keys=keys, ndev=ndev,
        min_note_duration=min_note_duration, loudness_threshold_ratio=loudness_threshold_ratio
    )
    print("aux.shape = ", aux.shape)
    print("len(signals),1,2, notes = [", len(signals),1,2, notes,"]")
    print("loudness_tensor[:,i,:,:] = ", loudness_tensor[:,i,:,:].shape)
    print("aux.reshape(len(signals),1,2, notes) = ", aux.reshape(len(signals),1,2, notes).shape)   
    loudness_tensor[:,i,:,:] = aux.reshape(len(signals),2, notes)
    print("% = ", 50*(i+1))     
    
    
#     loudness_tensor[:,i,:,:] = split_piano_keys_loudness(
#         signals[:,i,:,:].squeeze(), fs=fs, keys=keys, ndev=ndev,
#         min_note_duration=min_note_duration, loudness_threshold_ratio=loudness_threshold_ratio
#     )

loudness = np.asarray(loudness_tensor)
np.save(output_path,loudness)
print(" Data conversion complete")

# # Convert back to signal
# recon_signal = from_loudness_to_signal(
#     loudness_tensor, fs=fs, keys=keys, ndev=ndev,
#     min_note_duration=min_note_duration, seq_len=signals.shape[-1]
# )

# # Save the result as numpy array
# for i in range(0,len(signals)):
#     output_path_music = output_path + "sample_" + str(i) + ".npy"
#     np.save(output_path_music, recon_signal[i,...].cpu().numpy().squeeze())
# 	

