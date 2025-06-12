#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:23:42 2025

@author: andrey
"""


import torch
import torch.nn as nn
import scipy.signal
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MelSpectrogram

def pitch_consistency_loss(waveform, sample_rate, weight=1.0):
    """
    Penalizes inconsistencies in pitch to enforce melodic structure.
    
    Args:
        waveform (torch.Tensor): Generated waveform (batch_size, channels, samples).
        sample_rate (int): Sampling rate of the audio.
        weight (float): Weight to scale the loss.

    Returns:
        torch.Tensor: Loss value enforcing pitch consistency.
    """
    # Compute pitch using a pitch detection algorithm
    pitch_values = compute_pitch(waveform, sample_rate)
    
    # Penalize large differences in consecutive pitch values to enforce smooth melodic transitions
    pitch_diff = torch.abs(pitch_values[:, 1:] - pitch_values[:, :-1])
    
    # Compute loss
    loss = torch.mean(pitch_diff)
    return weight * loss



def spectral_outlier_discriminator_loss(real_waveform, fake_waveform, sample_rate, threshold=10.0, weight=1.0):
    """
    Computes a discriminator loss that penalizes spectral outliers in fake music and encourages correct classification.

    Args:
        real_waveform (torch.Tensor): Waveform of real music (batch_size, channels, samples).
        fake_waveform (torch.Tensor): Waveform of fake music (batch_size, channels, samples).
        sample_rate (int): Sampling rate of the audio.
        threshold (float): Threshold for spectral outliers.
        weight (float): Weight for the loss.

    Returns:
        torch.Tensor: Discriminator loss for penalizing spectral outliers in fake music.
    """
    # Compute STFT for fake waveforms
    fake_stft = torch.stft(fake_waveform.mean(dim=1), n_fft=2048, hop_length=512, return_complex=True)
    fake_magnitude_spectrum = fake_stft.abs()

    # Compute mean spectrum and identify outliers
    fake_mean_spectrum = fake_magnitude_spectrum.mean(dim=-1, keepdim=True)
    fake_outliers = torch.clamp(fake_magnitude_spectrum / fake_mean_spectrum - threshold, min=0.0)

    # Penalize fake music with spectral outliers
    fake_outlier_penalty = torch.mean(fake_outliers)

    # Return the positive loss
    return weight * fake_outlier_penalty


def spectral_regularization_loss(waveform, sample_rate, threshold=10.0, weight=1.0):
    # Compute STFT
    stft = torch.stft(waveform.mean(dim=1), n_fft=2048, hop_length=512, return_complex=True)
    magnitude_spectrum = stft.abs()

    # Compute mean and threshold
    mean_spectrum = magnitude_spectrum.mean(dim=-1, keepdim=True)
    outliers = torch.clamp(magnitude_spectrum / mean_spectrum - threshold, min=0.0)

    # Penalize outliers
    loss = torch.mean(outliers)
    return weight * loss





def combined_perceptual_loss(fake, real, sample_rate, scales=[512, 1024, 2048], mel_weight=1.0, complex_weight=1.0, multi_scale_weight=1.0):
    """
    Combines multi-scale STFT loss, complex STFT loss, and Mel-spectrogram loss into one perceptual loss.
    Args:
        fake: Generated audio tensor [batch, channels, seq_len].
        real: Real audio tensor [batch, channels, seq_len].
        sample_rate: Sampling rate of the audio.
        scales: List of FFT sizes for multi-scale STFT loss.
        mel_weight: Weight for the Mel-spectrogram loss.
        complex_weight: Weight for the complex STFT loss.
        multi_scale_weight: Weight for the multi-scale STFT loss.
    Returns:
        combined_loss: Scalar tensor representing the combined perceptual loss.
    """
    # Define Mel-spectrogram transformation
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=512).to(fake.device)
    amplitude_to_db = AmplitudeToDB().to(fake.device)

    # Compute Mel-spectrogram loss
    mel_fake = amplitude_to_db(mel_transform(fake.mean(dim=1)))
    mel_real = amplitude_to_db(mel_transform(real.mean(dim=1)))
    mel_loss = torch.nn.functional.l1_loss(mel_fake, mel_real)

    # Compute complex STFT loss
    fake_stft = torch.stft(fake.mean(dim=1), n_fft=2048, hop_length=512, return_complex=True)
    real_stft = torch.stft(real.mean(dim=1), n_fft=2048, hop_length=512, return_complex=True)
    complex_loss = torch.nn.functional.l1_loss(fake_stft, real_stft)

    # Compute multi-scale STFT loss
    multi_scale_loss = 0.0
    for scale in scales:
        fake_stft_scale = torch.stft(fake.mean(dim=1), n_fft=scale, hop_length=scale // 4, return_complex=True)
        real_stft_scale = torch.stft(real.mean(dim=1), n_fft=scale, hop_length=scale // 4, return_complex=True)
        multi_scale_loss += torch.nn.functional.l1_loss(fake_stft_scale.abs(), real_stft_scale.abs())

    # Combine losses with weights
    combined_loss = (
        mel_weight * mel_loss +
        complex_weight * complex_loss +
        multi_scale_weight * multi_scale_loss
    )
    return combined_loss





def rhythm_enforcement_loss(waveform, beats, large_duration_range, short_duration_range, weight=1.0):
    """
    Enforces rhythmic patterns with variability in beat durations, allowing diversity in generated music.
    Args:
        waveform: Tensor of shape [batch, channels, seq_len], generated audio waveform.
        beats: Tensor of detected beat positions (in samples).
        large_duration_range: Tuple (min, max), allowable range for large beat durations.
        short_duration_range: Tuple (min, max), allowable range for short beat durations.
        weight: Loss scaling factor.
    Returns:
        rhythm_loss: Tensor, rhythmic enforcement loss with variability.
    """
    batch_size = beats.size(0)
   # seq_len = waveform.size(-1)
    rhythm_loss = 0.0

    for b in range(batch_size):
        for i in range(beats.size(1) - 3):
            # Get beat positions and ensure indices are within bounds
            large_beat_idx = beats[b, i].long().item()
            short_beat_idx_1 = beats[b, i + 1].long().item()
            short_beat_idx_2 = beats[b, i + 2].long().item()
            next_large_beat_idx = beats[b, i + 3].long().item()

            if next_large_beat_idx <= short_beat_idx_2:
                continue  # Skip invalid sequences

            # Enforce distances between beats with variability
            large_to_short_1_dist = torch.tensor(abs(short_beat_idx_1 - large_beat_idx), dtype=torch.float32)
            short_1_to_short_2_dist = torch.tensor(abs(short_beat_idx_2 - short_beat_idx_1), dtype=torch.float32)
            short_2_to_large_dist = torch.tensor(abs(next_large_beat_idx - short_beat_idx_2), dtype=torch.float32)

            large_duration_penalty = 0.0
            if not (large_duration_range[0] <= large_to_short_1_dist <= large_duration_range[1]):
                large_duration_penalty = (large_to_short_1_dist - torch.clamp(large_to_short_1_dist, min=large_duration_range[0], max=large_duration_range[1])) ** 2

            short_duration_penalty_1 = 0.0
            if not (short_duration_range[0] <= short_1_to_short_2_dist <= short_duration_range[1]):
                short_duration_penalty_1 = (short_1_to_short_2_dist - torch.clamp(short_1_to_short_2_dist, min=short_duration_range[0], max=short_duration_range[1])) ** 2

            short_duration_penalty_2 = 0.0
            if not (large_duration_range[0] <= short_2_to_large_dist <= large_duration_range[1]):
                short_duration_penalty_2 = (short_2_to_large_dist - torch.clamp(short_2_to_large_dist, min=large_duration_range[0], max=large_duration_range[1])) ** 2

            rhythm_loss += large_duration_penalty + short_duration_penalty_1 + short_duration_penalty_2

    return weight * rhythm_loss / batch_size




def rhythm_detection_penalty(real_beats, fake_beats, large_duration_range, short_duration_range, weight=1.0):
    """
    Encourages the discriminator to detect rhythmic sequences with variability in generated music.
    Args:
        real_beats: Tensor of detected beat positions in real music (in samples).
        fake_beats: Tensor of detected beat positions in generated music (in samples).
        large_duration_range: Tuple (min, max), allowable range for large beat durations.
        short_duration_range: Tuple (min, max), allowable range for short beat durations.
        weight: Loss scaling factor.
    Returns:
        detection_penalty: Tensor, rhythmic sequence detection penalty with variability.
    """
    batch_size = real_beats.size(0)
    detection_penalty = 0.0

    for b in range(batch_size):
        for i in range(real_beats.size(1) - 3):
            # Get real and fake beat positions
            real_large_beat_idx = real_beats[b, i].long().item()
            real_short_beat_idx_1 = real_beats[b, i + 1].long().item()
            real_short_beat_idx_2 = real_beats[b, i + 2].long().item()
            real_next_large_beat_idx = real_beats[b, i + 3].long().item()

            fake_large_beat_idx = fake_beats[b, i].long().item()
            fake_short_beat_idx_1 = fake_beats[b, i + 1].long().item()
            fake_short_beat_idx_2 = fake_beats[b, i + 2].long().item()
            fake_next_large_beat_idx = fake_beats[b, i + 3].long().item()

            if real_next_large_beat_idx <= real_short_beat_idx_2 or fake_next_large_beat_idx <= fake_short_beat_idx_2:
                continue  # Skip invalid sequences

            # Penalize discrepancies in rhythmic pattern with variability
            real_rhythm_distances = torch.tensor([
                abs(real_short_beat_idx_1 - real_large_beat_idx),
                abs(real_short_beat_idx_2 - real_short_beat_idx_1),
                abs(real_next_large_beat_idx - real_short_beat_idx_2)
            ], dtype=torch.float32)
            fake_rhythm_distances = torch.tensor([
                abs(fake_short_beat_idx_1 - fake_large_beat_idx),
                abs(fake_short_beat_idx_2 - fake_short_beat_idx_1),
                abs(fake_next_large_beat_idx - fake_short_beat_idx_2)
            ], dtype=torch.float32)

            large_duration_penalty = torch.mean((real_rhythm_distances - torch.clamp(fake_rhythm_distances, min=large_duration_range[0], max=large_duration_range[1])) ** 2)
            short_duration_penalty = torch.mean((fake_rhythm_distances - torch.clamp(real_rhythm_distances, min=short_duration_range[0], max=short_duration_range[1])) ** 2)

            detection_penalty += large_duration_penalty + short_duration_penalty

    return weight * detection_penalty / batch_size





def silence_loss(waveform, beats, weight=1.0):
    """
    Penalizes non-zero values in quiet intervals (regions of low loudness).
    Args:
        waveform: Tensor of shape [batch, channels, seq_len], generated waveform.
        beats: Tensor of detected beat positions (in samples).
        weight: Loss scaling factor.
    Returns:
        silence_loss: Tensor, silence regularity loss.
    """
    batch_size, seq_len = waveform.shape[0], waveform.shape[-1]

    # Create silence mask for all batches at once
    silence_mask = torch.ones_like(waveform, dtype=torch.bool)

    # Vectorized masking of beat intervals
    for b in range(batch_size):
        beat_indices = beats[b].long().clamp(max=seq_len - 1)  # Ensure indices are within bounds
        for i in range(beat_indices.size(0) - 1):
            silence_mask[b, :, beat_indices[i]:beat_indices[i + 1]] = False

    # Define silence regions as parts where absolute waveform amplitude is less than 0.5 * std
    std_per_channel = waveform.std(dim=-1, keepdim=True)  # Compute std along the sequence dimension
    silence_condition = waveform.abs() < 0.5 * std_per_channel
    silence_mask &= silence_condition

    # Penalize non-zero values in silence regions
    silence_loss = torch.mean(torch.abs(waveform[silence_mask]))
    return weight * silence_loss



def beat_timing_loss(real_beats, fake_beats, weight=1.0):
    dtw_distance = torch.mean(torch.abs(real_beats - fake_beats))  # Dynamic Time Warping (DTW)
    return weight * dtw_distance




def compute_beats(waveform, sample_rate, max_beats=100):
    """
    Compute the beat positions for a waveform using torchaudio.
    Args:
        waveform: Tensor of shape [batch, channels, seq_len]
        sample_rate: Sampling rate of the waveform
        max_beats: Maximum number of beats to return per batch sample (for padding)
    Returns:
        beats: Tensor of shape [batch, max_beats], representing beat positions in seconds
    """
    batch_size, channels, seq_len = waveform.shape

    # Convert to mono if stereo
    mono_waveform = waveform.mean(dim=1)  # Average across channels

    # Compute onset strength
    spectrogram_transform = Spectrogram(n_fft=2048, hop_length=512, power=2.0).to(waveform.device)
    amplitude_to_db = AmplitudeToDB().to(waveform.device)
    spectrogram = amplitude_to_db(spectrogram_transform(mono_waveform))

    # Validate the spectrogram
    if spectrogram.size(-1) <= 1:
        raise ValueError("Spectrogram has insufficient size for beat detection. Check input waveform.")

    onset_strength = torch.mean(spectrogram, dim=1)  # Aggregate across frequency bins

    # Validate onset_strength shape
    if onset_strength.size(1) <= 1:
        raise ValueError("onset_strength has insufficient size for peak detection. Check spectrogram computation.")

    # Normalize onset strength
    onset_strength = (onset_strength - onset_strength.min(dim=-1, keepdim=True)[0]) / (
            onset_strength.max(dim=-1, keepdim=True)[0] + 1e-8)

    # Compute threshold for peak detection
    threshold = onset_strength.mean(dim=-1, keepdim=True) + 0.5 * onset_strength.std(dim=-1, keepdim=True)

    # Ensure threshold matches onset_strength shape
    threshold = threshold.expand_as(onset_strength)

    # Detect peaks in onset strength
    peaks = (onset_strength[:, 1:] > onset_strength[:, :-1]) & (onset_strength[:, 1:] > threshold[:, 1:])

    # Compute beat positions
    beat_positions = peaks.nonzero(as_tuple=False)
    if beat_positions.numel() == 0:
        return torch.zeros((batch_size, max_beats), device=waveform.device)

    # Scale beat positions to time in seconds
    beat_positions[:, 1] = beat_positions[:, 1] * (512 / sample_rate)

    # Pad or truncate beats
    beats = torch.zeros((batch_size, max_beats), device=waveform.device)
    for b in range(batch_size):
        batch_beat_positions = beat_positions[beat_positions[:, 0] == b][:, 1]
        beats[b, :min(max_beats, batch_beat_positions.size(0))] = batch_beat_positions[:max_beats]

    return beats



def rythm_and_beats_loss(real_data, fake_data, sample_rate=12000, rhythm_weight=0.01, pitch_weight=0.01, btw_weight=1.0):
    """
    Compute the rhythm and pitch loss between real and fake audio data.
    Args:
        real_data: Tensor of shape [batch, channels, seq_len], real audio waveform
        fake_data: Tensor of shape [batch, channels, seq_len], generated audio waveform
        sample_rate: Sampling rate of the audio
        rhythm_weight: Weight for the rhythm loss
        pitch_weight: Weight for the pitch loss
    Returns:
        rythm_and_beats_loss: Combined rhythm and pitch loss
    """
    # Compute rhythm (beats)
    real_beats = compute_beats(real_data, sample_rate)
    fake_beats = compute_beats(fake_data, sample_rate)

    # Normalize beats before calculating loss
    real_beats_norm = (real_beats - real_beats.mean(dim=1, keepdim=True)) / (real_beats.std(dim=1, keepdim=True) + 1e-8)
    fake_beats_norm = (fake_beats - fake_beats.mean(dim=1, keepdim=True)) / (fake_beats.std(dim=1, keepdim=True) + 1e-8)

    # Mask padded values (assume zeros are used for padding)
    mask = (real_beats != 0) & (fake_beats != 0)
    rhythm_loss = torch.mean(torch.abs(real_beats_norm[mask] - fake_beats_norm[mask]))

    # Compute pitch
    real_pitch = compute_pitch(real_data, sample_rate)
    fake_pitch = compute_pitch(fake_data, sample_rate)

    # Normalize pitch before calculating loss
    real_pitch_norm = (real_pitch - real_pitch.mean(dim=-1, keepdim=True)) / (real_pitch.std(dim=-1, keepdim=True) + 1e-8)
    fake_pitch_norm = (fake_pitch - fake_pitch.mean(dim=-1, keepdim=True)) / (fake_pitch.std(dim=-1, keepdim=True) + 1e-8)

    pitch_loss = torch.mean(torch.abs(real_pitch_norm - fake_pitch_norm))

    # Beat timing loss
    dtw_loss = torch.mean(torch.abs(real_beats_norm - fake_beats_norm)) * btw_weight

    # Combine losses
    rythm_and_beats_loss = rhythm_weight * rhythm_loss + pitch_weight * pitch_loss + dtw_loss
    return rythm_and_beats_loss




def compute_pitch(waveform, sample_rate, n_fft=2048, hop_length=512):
    batch_size, channels, seq_len = waveform.shape
    mono_waveform = waveform.mean(dim=1)  # Convert to mono
    stft = torch.stft(mono_waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitudes = stft.abs()
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).to(waveform.device)
    pitch_indices = magnitudes.argmax(dim=1)
    pitch = freqs[pitch_indices]

    # Introduce random pitch modulation
    pitch_variation = torch.randn_like(pitch) * 0.1  # Add noise
    return pitch + pitch_variation


def beat_duration_loss(waveform, sample_rate, min_duration, max_duration, weight=1.0):
    """
    Penalizes beat durations that fall outside the desired range.
    Args:
        waveform: Tensor of shape [batch, channels, seq_len], generated waveform.
        sample_rate: Sampling rate of the audio.
        min_duration: Minimum allowable beat duration (in seconds).
        max_duration: Maximum allowable beat duration (in seconds).
        weight: Loss scaling factor.
    Returns:
        duration_loss: Tensor, beat duration loss.
    """
    # Compute beats
    beats = compute_beats(waveform, sample_rate)
    beat_durations = beats[:, 1:] - beats[:, :-1]  # Time differences between consecutive beats

    # Mask valid durations
    valid_mask = (beat_durations >= min_duration) & (beat_durations <= max_duration)

    # Penalize durations outside the range
    duration_loss = torch.mean((~valid_mask) * beat_durations**2)  # Quadratic penalty for invalid durations
    return weight * duration_loss

 

def alternation_loss(beats, waveform, weight=1.0):
    """
    Penalizes deviations from the beat-quiet-beat alternation pattern.
    Args:
        beats: Tensor of detected beat positions (in samples).
        waveform: Tensor of shape [batch, channels, seq_len], generated waveform.
        weight: Loss scaling factor.
    Returns:
        alternation_loss: Tensor, alternation loss.
    """
    batch_size = beats.size(0)
    seq_len = waveform.size(-1)
    alternation_loss = 0.0

    for b in range(batch_size):
        for i in range(beats.size(1) - 1):
            # Convert beat positions to integers and ensure indices are in bounds
            start_idx = min(beats[b, i].long().item(), seq_len - 1)
            end_idx = min(beats[b, i+1].long().item(), seq_len - 1)

            if start_idx >= end_idx:
                continue  # Skip invalid intervals

            interval = waveform[b, :, start_idx:end_idx]
            
            if i % 2 == 0:  # Beat interval
                alternation_loss += torch.mean((interval.abs() - interval.abs().mean())**2)
            else:  # Quiet interval
                alternation_loss += torch.mean(interval.abs()**2)

    return weight * alternation_loss


def smoothing_loss(waveform, weight=1.0):
    # waveform: [batch, channels, seq_len]
    diff = waveform[..., 1:] - waveform[..., :-1]
    return weight * diff.abs().mean()

def smoothing_loss2(waveform, weight=1.0):
    # waveform: [batch, channels, seq_len]
    diff1 = waveform[..., 1:-1] - waveform[..., :-2]
    diff2 = waveform[..., 2:] - waveform[..., 1:-1]
    
    diff = diff2 - diff1
    return weight * diff.abs().mean()


def stft_loss(fake, real, nchunks=100, n_fft=1024 * 2, hop_length=256, weight=1.0): #1024
    """
    Computes cumulative STFT perceptual loss by splitting the sequence into nchunks.
    Args:
        fake, real: [batch, channels, seq_len]
        nchunks: number of chunks to split the sequence into
        n_fft, hop_length: STFT parameters
        weight: loss scaling factor
    Returns:
        cumulative_loss: sum of per-chunk losses
    """
    batch, channels, seq_len = fake.shape
    chunk_len = seq_len // nchunks
    cumulative_loss = 0.0

    for i in range(nchunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < nchunks - 1 else seq_len
        fake_chunk = fake[..., start:end]
        real_chunk = real[..., start:end]
        # Flatten batch and channel for stft
        fake_chunk = fake_chunk.contiguous().view(-1, fake_chunk.shape[-1])
        real_chunk = real_chunk.contiguous().view(-1, real_chunk.shape[-1])

        stft_fake = torch.stft(fake_chunk, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        stft_real = torch.stft(real_chunk, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        mag_fake = stft_fake.abs()
        mag_real = stft_real.abs()
        loss = torch.nn.functional.l1_loss(mag_fake, mag_real)
        cumulative_loss += loss

    return weight * cumulative_loss


def freeze_decoder_weights(model):
    for name, param in model.variational_encoder_decoder.named_parameters():
        if "decoder" in name:
            param.requires_grad = False

def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False

def denormalize_waveform(waveform, mean, std):
    return waveform * std + mean

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



def compute_chunkwise_stats_loss(fake_music, real_music, num_chunks=1000, lambda_mean=0.15, lambda_std=0.15, lambda_max=0.075):  # num_chunks = 200
    batch_size, n_channels, seq_len = real_music.size()
    chunk_len = seq_len // num_chunks
    local_mean_loss = 0
    local_std_loss = 0
    local_max_loss = 0
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < num_chunks - 1 else seq_len
        real_chunk = real_music[..., start:end]
        fake_chunk = fake_music[..., start:end]
        mean_real = real_chunk.mean(dim=-1, keepdim=True)
        mean_fake = fake_chunk.mean(dim=-1, keepdim=True)
        std_real = real_chunk.std(dim=-1, keepdim=True)
        std_fake = fake_chunk.std(dim=-1, keepdim=True)
        max_real = real_chunk.abs().amax(dim=-1, keepdim=True)
        max_fake = fake_chunk.abs().amax(dim=-1, keepdim=True)
        local_mean_loss += torch.mean((mean_fake - mean_real) ** 2)
        local_std_loss  += torch.mean((std_fake - std_real) ** 2)
        local_max_loss  += torch.mean((max_fake - max_real) ** 2)
    # Average over chunks
    local_mean_loss /= num_chunks
    local_std_loss  /= num_chunks
    local_max_loss  /= num_chunks
    return (
        lambda_mean * local_mean_loss +
        lambda_std * local_std_loss +
        lambda_max * local_max_loss
    )



def penalize_peaks_loss(signals, fs=12000, music_band=(0, 8000), peak_height_ratio=0.1, q_factor=30.0):
    """
    Differentiable penalty cost for parasitic peaks outside the desired frequency band for batched stereo signals,
    with filtering applied.

    Parameters:
    - signals: torch.Tensor, shape [batch_size, channels, signal_length], stereo signals
    - fs: int, sample rate in Hz
    - music_band: tuple (low, high), frequencies considered as "wanted" (Hz)
    - peak_height_ratio: float, relative height threshold for peak detection
    - q_factor: float, quality factor for narrowing the band (higher = narrower)

    Returns:
    - penalty: torch.Tensor, aggregated penalty cost for the batch
    """

    # FFT across the last dimension (signal length)
    fft_vals = torch.fft.rfft(signals, dim=-1)  # Compute FFT
    fft_magnitude = torch.abs(fft_vals)  # Magnitude of FFT
    
    # Compute mean magnitude within the band for normalization
    mean_magnitude = torch.mean(fft_magnitude, dim=-1, keepdim=True) + 1e-8  # Avoid division by zero

    # Define threshold for peak detection (per channel)
    threshold = torch.max(fft_magnitude, dim=-1, keepdim=True).values * peak_height_ratio

    # Detect outliers outside the desired band
    outliers = fft_magnitude[..., fft_magnitude > threshold]

    # Compute relative heights of outliers
    relative_heights = outliers / mean_magnitude

    # Aggregate penalty for each channel and batch
    channel_penalty = torch.sum(relative_heights, dim=-1, keepdim=True)  # Ensure channel_penalty retains batch and channel dimensions
    penalty = channel_penalty.mean(dim=0).mean()  # Aggregate across batches and channels

    return penalty#, outliers



def bandwise_generator_loss(fake_band_signals, real_band_signals, fband_signals, real_music, 
                            criterion_recon=None, criterion_gan=None, discriminator=None, real_labels=None, loss_weights=None):
    """
    Computes generator loss for bandwise outputs.
    Args:
        fake_band_signals: dict of time-domain tensors, keys are band names, values shape [B, C, L]
        real_band_signals: dict of time-domain tensors, same structure as fake_band_signals
        fband_signals: dict of frequency-domain tensors, keys are band names, values shape [B, C, F]
        real_music: torch.Tensor, [B, C, L] (for full waveform GAN loss)
        criterion_recon: loss function for reconstruction (default: nn.SmoothL1Loss())
        criterion_gan: GAN loss criterion (default: nn.BCEWithLogitsLoss())
        discriminator: discriminative model, must return logits for GAN loss
        real_labels: torch.Tensor, shape [B, 1]
        loss_weights: dict of weights for each loss term (optional)
    Returns:
        total_g_loss: scalar tensor (requires grad)
        loss_terms: dict of individual loss terms (for logging)
    """
    if criterion_recon is None:
        criterion_recon = nn.SmoothL1Loss()
    if criterion_gan is None:
        criterion_gan = nn.BCEWithLogitsLoss()

    total_g_loss = 0.0
    loss_terms = {}

    # 1. Bandwise reconstruction loss (sum over bands)
    recon_loss = 0.0
    for band in fake_band_signals:
        recon_loss += criterion_recon(fake_band_signals[band], real_band_signals[band])
    if loss_weights is not None and "band_recon" in loss_weights:
        recon_loss = loss_weights["band_recon"] * recon_loss
    loss_terms["band_recon"] = recon_loss
    total_g_loss += recon_loss

    # 2. Bandwise spectral (frequency domain) loss (optional, e.g., L1/L2 on F-domain)
    spectral_loss = 0.0
    for band in fband_signals:
        spectral_loss += torch.nn.functional.l1_loss(fband_signals[band], torch.fft.rfft(real_band_signals[band], n=real_band_signals[band].shape[-1], dim=-1))
    if loss_weights is not None and "band_spectral" in loss_weights:
        spectral_loss = loss_weights["band_spectral"] * spectral_loss
    loss_terms["band_spectral"] = spectral_loss
    total_g_loss += spectral_loss

    # 3. Full waveform adversarial loss (if using a full waveform discriminator)
    if discriminator is not None and real_labels is not None:
        merged_fake = sum(fake_band_signals.values())
        g_gan_loss = criterion_gan(discriminator(merged_fake), real_labels)
        if loss_weights is not None and "gan" in loss_weights:
            g_gan_loss = loss_weights["gan"] * g_gan_loss
        loss_terms["gan"] = g_gan_loss
        total_g_loss += g_gan_loss

    return total_g_loss, loss_terms

def bandwise_discriminator_loss(fake_band_signals, real_band_signals, discriminator, criterion_gan=None, real_labels=None, fake_labels=None, loss_weights=None):
    """
    Computes discriminator loss for bandwise outputs.
    Args:
        fake_band_signals: dict of time-domain tensors, keys are band names, values shape [B, C, L]
        real_band_signals: dict of time-domain tensors, same structure as fake_band_signals
        discriminator: discriminative model, must return logits for GAN loss
        criterion_gan: GAN loss criterion (default: nn.BCEWithLogitsLoss())
        real_labels: torch.Tensor, shape [B, 1]
        fake_labels: torch.Tensor, shape [B, 1]
        loss_weights: dict of weights for each loss term (optional)
    Returns:
        total_d_loss: scalar tensor (requires grad)
        loss_terms: dict of individual loss terms (for logging)
    """
    if criterion_gan is None:
        criterion_gan = nn.BCEWithLogitsLoss()

    total_d_loss = 0.0
    loss_terms = {}

    # Full waveform discriminator: merge bands (sum or other merge, here sum)
    merged_real = sum(real_band_signals.values())
    merged_fake = sum(fake_band_signals.values())

    d_real_logits = discriminator(merged_real)
    d_fake_logits = discriminator(merged_fake)

    d_loss_real = criterion_gan(d_real_logits, real_labels)
    d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    if loss_weights is not None and "gan" in loss_weights:
        d_loss = loss_weights["gan"] * d_loss

    loss_terms["gan"] = d_loss
    total_d_loss += d_loss

    return total_d_loss, loss_terms
