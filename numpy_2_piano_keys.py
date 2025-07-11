#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:53:35 2025

@author: andreyvlasenko
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:05:35 2025

@author: andreyvlasenko
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os

def piano_key_intervals(keys: List[int], ndev: float) -> Dict[int, Tuple[float, float]]:
    intervals = {}
    for n in keys:
        freq = 2 ** ((n - 49) / 12) * 440
        intervals[n] = (freq - ndev, freq + ndev)
    return intervals

def split_piano_keys_loudness(
    signals: torch.Tensor, 
    fs: int,
    keys: List[int],
    ndev: float,
    min_note_duration: float = 0.05,
    loudness_threshold_ratio: float = 0.05
) -> torch.Tensor:
    """Splits signals into frequency bands for piano keys and computes loudness."""
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
        freq_center = (bands[key][0] + bands[key][1]) / 2
        std = 0.5 * ndev
        gauss_mask = np.exp(-0.5 * ((freqs - freq_center) / std) ** 2)
        gauss_mask = gauss_mask.reshape(1, 1, -1)
        band_fft = fft_vals * gauss_mask
        band_signal = np.fft.irfft(band_fft, n=fft_len, axis=-1)
        key_band_signals.append(band_signal)
    key_band_signals = np.stack(key_band_signals, axis=0)  # [n_keys, batch, n_channels, seq_len]

    loudness = []
    for t in range(n_intervals):
        start = t * interval_len
        end = start + interval_len
        seg = key_band_signals[..., start:end]
        rms = np.sqrt(np.mean(seg**2, axis=-1))  # [n_keys, batch, n_channels]
        loudness.append(rms)
    loudness = np.stack(loudness, axis=0)  # [n_intervals, n_keys, batch, n_channels]
    loudness = np.transpose(loudness, (2, 3, 1, 0))  # [batch, n_channels, n_keys, n_intervals]

    max_loudness = loudness.max()
    threshold = loudness_threshold_ratio * max_loudness
    loudness[loudness < threshold] = 0.0

    loudness_tensor = torch.from_numpy(loudness).float().to(signals.device)
    return loudness_tensor

def from_loudness_to_signal(
    loudness_tensor: torch.Tensor,
    fs: int,
    keys: List[int],
    ndev: float,
    min_note_duration: float = 0.05,
    seq_len: int = 120000
) -> torch.Tensor:
    """Reconstructs signal from per-key loudness tensor."""
    device = loudness_tensor.device
    batch, n_channels, n_keys, n_intervals = loudness_tensor.shape
    interval_len = int(round(min_note_duration * fs))
    total_len = n_intervals * interval_len

    t = torch.arange(interval_len, device=device).float() / fs

    bands = piano_key_intervals(keys, ndev)
    key_freqs = torch.tensor([((fmin + fmax) / 2) for (fmin, fmax) in bands.values()], device=device).float()

    signals = torch.zeros((batch, n_channels, total_len), device=device)

    for interval_idx in range(n_intervals):
        loudness = loudness_tensor[:, :, :, interval_idx]  # [batch, n_channels, n_keys]
        phases = 2 * np.pi * key_freqs[None, None, :, None] * t[None, None, None, :]  # [1,1,n_keys,interval_len]
        sinusoids = torch.sin(phases)  # [1,1,n_keys,interval_len]
        loudness_exp = loudness.unsqueeze(-1)  # [batch, n_channels, n_keys, 1]
        note_waves = loudness_exp * sinusoids  # [batch, n_channels, n_keys, interval_len]
        interval_wave = note_waves.sum(dim=2)  # [batch, n_channels, interval_len]
        start = interval_idx * interval_len
        end = start + interval_len
        signals[:, :, start:end] += interval_wave

    if total_len > seq_len:
        signals = signals[:, :, :seq_len]
    return signals.float()


def main():
    # Example usage:
    input_path = "../validation_set.npy"
    output_path = "GAN_output/"
    save_piano_keys =True  # Do we save the training or validation sets?
    ndev = 25    # We assume that each piano's note has a gaussian bell-shaped frequences ndev is their standart deviation
    keys = list(range(27, 60))
    fs = 12000
    min_note_duration = 0.02  # loudness of all notes withing this interval is constant
    loudness_threshold_ratio = 0.025 # anything below the threshold is zeroed

    # Load numpy array (shape [batch, n_channels, seq_len])
    np_signal = np.load(input_path)
    if np_signal.ndim == 3:
        signals = torch.from_numpy(np_signal).float()
    elif np_signal.ndim == 4:
        signals = torch.from_numpy(np_signal).float()
        # Handle shape [batch, channels, 1, seq_len] or similar
        signals = signals.squeeze(2)
    else:
        raise ValueError(f"Unexpected input npy shape: {np_signal.shape}")

    signals = signals[:10, 0, :, :].squeeze()  # [batch, n_channels, seq_len]
    print("signals.shape = ", signals.shape)

    # Convert to loudness tensor
    loudness_tensor = split_piano_keys_loudness(
        signals, fs=fs, keys=keys, ndev=ndev,
        min_note_duration=min_note_duration, loudness_threshold_ratio=loudness_threshold_ratio
    )

    if save_piano_keys:
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, 'piano_keys.npy'), loudness_tensor.cpu().numpy().squeeze())
    

    # Convert back to signal for testing 
    recon_signal = from_loudness_to_signal(
        loudness_tensor, fs=fs, keys=keys, ndev=ndev,
        min_note_duration=min_note_duration, seq_len=signals.shape[-1]
    )

    # Save the result as numpy array
    os.makedirs(output_path, exist_ok=True)
    for i in range(0, 2):    # This is just for music quality testing !!!! Convert saved .npy arrays back to mp3 and hear the music
        output_path_music = os.path.join(output_path, f"sample_{i}.npy")
        print(output_path_music)
        np.save(output_path_music, recon_signal[i, ...].cpu().numpy().squeeze())


if __name__ == "__main__":
    main()