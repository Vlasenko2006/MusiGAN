#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:11:31 2025

@author: andreyvlasenko
"""

import torch

def split_signal_frequency_bands(signals, fs=12000):
    """
    Splits batched signals into defined frequency bands, returns each band back in time domain.

    Args:
        signals: torch.Tensor, shape [batch_size, n_channels, seq_length] (float32/float64, can be on GPU)
        fs: int, sample rate in Hz (default 12kHz)
    Returns:
        band_signals: dict, keys are band names, values are torch.Tensor of same shape as input (same device & dtype)
    """
    # Define frequency bands (Hz)
    bands = {
        "Super_Ultra_Low":   (0,    200),
        "Ultra_Low":   (200,    500),
        "Low":         (500,  1000),
        "Low_Medium":  (1000, 1500),
        "Medium":      (1500, 2000),
        "High":        (2000, 3000),
        "Ultra_High":  (3000, 6000)
    }
    seq_length = signals.shape[-1]
    device = signals.device
    dtype = signals.dtype

    # FFT along the last axis (signal length)
    fft_vals = torch.fft.rfft(signals, n=seq_length, dim=-1)  # shape: [B, C, F]

    # Get frequency bins (numpy is ok for freq calculation, as it's not differentiable & doesn't require gradients)
    freqs = torch.fft.rfftfreq(seq_length, d=1/fs, device=device, dtype=dtype)  # shape: [F]

    band_signals = {}
    for band_name, (f_min, f_max) in bands.items():
        # Create a mask for the frequency range
        freq_mask = (freqs >= f_min) & (freqs < f_max)  # shape: [F]
        # Broadcast mask to signal shape
        mask = freq_mask.reshape((1, 1, -1))    # [1, 1, F]
        # Zero out other frequencies
        band_fft = torch.zeros_like(fft_vals)
        band_fft = torch.where(mask, fft_vals, band_fft)
        # Inverse FFT to get time domain signal
        band_signal = torch.fft.irfft(band_fft, n=seq_length, dim=-1)
        band_signals[band_name] = band_signal

    return band_signals


def merge_band_signals(band_signals, fs=12000):
    """
    Takes a dict of band signals (each in time domain), applies FFT to each, sums their FFTs,
    then does an inverse FFT to reconstruct the merged time-domain signal.

    Args:
        band_signals: dict of torch.Tensor, each shape [batch_size, n_channels, seq_length]
        fs: int, sample rate (not strictly needed unless you want to check freq alignment)

    Returns:
        signal: torch.Tensor, merged signal in time domain, shape [batch_size, n_channels, seq_length]
    """
    bands = list(band_signals.keys())
    ref_signal = band_signals[bands[0]]
    batch_size, n_channels, seq_length = ref_signal.shape
    device = ref_signal.device
    dtype = ref_signal.dtype

    # Initialize sum of FFTs
    summed_fft = torch.zeros((batch_size, n_channels, seq_length // 2 + 1), dtype=torch.complex64 if dtype==torch.float32 else torch.complex128, device=device)

    # FFT for each band and sum
    for band in bands:
        fft_vals = torch.fft.rfft(band_signals[band], n=seq_length, dim=-1)
        summed_fft = summed_fft + fft_vals

    # Inverse FFT to get the merged signal
    merged_signal = torch.fft.irfft(summed_fft, n=seq_length, dim=-1)

    return merged_signal