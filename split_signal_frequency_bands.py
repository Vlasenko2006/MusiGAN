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
        "Low_Middle":  (1000, 1500),
        "Middle":      (1500, 2000),
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


def merge_band_signals(fband_signals, fs=12000):
    """
    Args:
        fband_signals: dict[str, torch.Tensor], each value shape [batch, channels, freq_bins] (FFT domain)
        fs: int, sample rate in Hz

    Returns:
        merged_signal: torch.Tensor, [batch, channels, seq_length] (time domain)
    """
    # Define the frequency bands (Hz)
    bands = {
        "Super_Ultra_Low": (0, 200),
        "Ultra_Low": (200, 500),
        "Low": (500, 1000),
        "Low_Middle": (1000, 1500),
        "Middle": (1500, 2000),
        "High": (2000, 3000),
        "Ultra_High": (3000, 6000)
    }

    # Get reference dimensions
    ref_band = next(iter(fband_signals.values()))
    batch_size, n_channels, n_freq_bins = ref_band.shape
    seq_length = (n_freq_bins - 1) * 2
    device = ref_band.device
    dtype = ref_band.dtype

    # Compute bin frequencies
    freqs = torch.fft.rfftfreq(seq_length, d=1.0/fs).to(device=device)

    # Initialize sum of FFTs
    summed_fft = torch.zeros((batch_size, n_channels, n_freq_bins), dtype=ref_band.dtype, device=device)

    for band_name, (f_min, f_max) in bands.items():
        band_fft = fband_signals[band_name]
        # Make a mask for valid frequencies
        mask = (freqs >= f_min) & (freqs < f_max)
        mask = mask.reshape((1, 1, -1))
        band_fft_clamped = torch.where(mask, band_fft, torch.zeros_like(band_fft))
        summed_fft = summed_fft + band_fft_clamped

    # Inverse FFT to get merged signal
    merged_signal = torch.fft.irfft(summed_fft, n=seq_length, dim=-1)
    return merged_signal