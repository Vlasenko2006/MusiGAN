import torch
import yaml
import os
from tqdm import tqdm
from vae_utilities import save_checkpoint, save_sample_as_numpy
from loss_functions import spectral_regularization_loss, combined_perceptual_loss, rhythm_enforcement_loss, rhythm_detection_penalty, silence_loss, beat_timing_loss, compute_beats, rythm_and_beats_loss, compute_pitch, beat_duration_loss, alternation_loss, smoothing_loss, smoothing_loss2, stft_loss, freeze_decoder_weights, denormalize_waveform, kl_divergence_loss, compute_chunkwise_stats_loss, spectral_outlier_discriminator_loss, penalize_peaks_loss

def normalize_loss_weights(generator,
                           discriminator,
                           train_loader,
                           sample_rate, 
                           batch_size,
                           n_channels,
                           noise_dim, 
                           device,
                           beat_min_duration,
                           beat_max_duration,
                           smoothing,
                           rhythm_weight,  # 0.005
                           pitch_weight,
                           btw_weight,
                           outlier_threshold,
                           comb_percept_lambda,
                           large_duration_range,
                           short_duration_range,
                           peak_height_ratio,
                           yaml_filepath=None,  # Path to YAML file for weight multipliers
                           batch_fraction=0.3):  # Fraction of batches to use for normalization
    """
    Computes average values for loss components over a subset of batches and sets their weights
    so that all costs start at a value of 1 at the beginning of training.
    Additionally, loads weight multipliers from a YAML file if provided, or creates a new YAML file with default weights.

    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        train_loader: Training data loader.
        sample_rate: Sampling rate of the audio.
        yaml_filepath: Path to YAML file containing weight multipliers.
        batch_fraction: Fraction of batches to use for normalization.

    Returns:
        loss_weights: Dictionary of normalized weights for each cost term.
    """
    generator.eval()
    discriminator.eval()

    # Initialize loss trackers
    loss_values = {
        "alternation_cost": 0.0,
        "beat_duration_cost": 0.0,
        "beat_timing_cost": 0.0,
        "cb_percept_cost": 0.0,
        "cp_cost": 0.0,
        "diversity_cost": 0.0,
        "fm_cost": 0.0,
        "freq_outlier_cost": 0.0,
        "g_cost": 0.0,
        "g_cost_kl": 0.0,
        "g_cost_reconstruction": 0.0,
        "g_cost_stats": 0.0,
        "monotony_cost": 0.0,
        "perceptual_cost": 0.0,
        "penalize_peaks_cost": 0,
        "rythm_and_beats_cost": 0.0,
        "silence_cost": 0.0,
        "smoothing_cost": 0.0,
        "smoothing_cost2": 0.0,
        "enforce_rhythm_cost": 0.0,
        "alternation_cost_d": 0.0,
        "beat_duration_cost_d": 0.0,
        "beat_duration_d": 0,
        "cb_percept_cost_d": 0.0,
        "cp_cost_d": 0,
        "d_cost_fake": 0,
        "d_cost_real": 0,
        "detection_cost_d": 0,
        "diversity_cost_d": 0,
        "freq_outlier_cost_d": 0.0,
        "silence_cost_d": 0.0
    }

    loss_weights = loss_values
    # Load weight multipliers from YAML file
    if yaml_filepath and os.path.exists(yaml_filepath):
        with open(yaml_filepath, 'r') as yaml_file:
            yaml_weights = yaml.safe_load(yaml_file)
            for loss_name, multiplier in yaml_weights.items():
                if loss_name in loss_weights:
                    loss_weights[loss_name] = multiplier
        print("Loaded YAML weights:", yaml_weights)
        print("Computed Loss Weights Before Multipliers:", loss_weights)
    else:
        print("Yaml file does not exsists, creating it")
        criterion_gan = torch.nn.BCEWithLogitsLoss()
        criterion_reconstruction = torch.nn.SmoothL1Loss()

        # Determine the number of batches to use (30% of total batches)
        total_batches = len(train_loader)
        num_batches_to_use = int(batch_fraction * total_batches)

        # Iterate through the subset of batches
        batch_count = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                if batch_count >= num_batches_to_use:
                    break  # Stop once we've processed the required fraction of batches

                real_labels = torch.ones(batch_size, 1, device=device) * (1 - smoothing)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                target_norm, target_mean, target_std = data
                real_music = target_norm.to(device).float()
                noisy_real_music = real_music + 0.0 * torch.randn_like(real_music)
                noise_g = torch.randn(batch_size, n_channels, noise_dim, device=device)

                # Forward pass
                fake_music_g, mu_g, logvar_g = generator(noise_g)
                real_features = discriminator.get_intermediate_features(real_music).detach()
                fake_features = discriminator.get_intermediate_features(fake_music_g)
                beats_real = compute_beats(noisy_real_music, sample_rate)
                beats = compute_beats(fake_music_g, sample_rate)
                fake_pitch = compute_pitch(fake_music_g, sample_rate)

                # Compute loss components
                g_cost_reconstruction = criterion_reconstruction(fake_music_g, real_music)
                g_cost_kl = kl_divergence_loss(mu_g, logvar_g) / batch_size
                g_cost = criterion_gan(discriminator(fake_music_g), real_labels)
                g_cost_stats = compute_chunkwise_stats_loss(fake_music_g, real_music)
                fm_cost = torch.mean((real_features.mean(0) - fake_features.mean(0))**2)
                smoothing_cost = smoothing_loss(fake_music_g)
                smoothing_cost2 = smoothing_loss2(fake_music_g)
                perceptual_cost = stft_loss(fake_music_g, real_music)
                rythm_and_beats_cost = rythm_and_beats_loss(real_music, fake_music_g, sample_rate=sample_rate, rhythm_weight=rhythm_weight, pitch_weight=pitch_weight, btw_weight=btw_weight)
                diversity_cost = -torch.mean(torch.var(fake_music_g, dim=0))  # Penalize batch similarity
                silence_cost = silence_loss(fake_music_g, beats)
                alternation_cost = alternation_loss(beats, fake_music_g)
                beat_duration_cost = beat_duration_loss(fake_music_g, sample_rate, min_duration=beat_min_duration, max_duration=beat_min_duration)
                monotony_cost = torch.mean((fake_pitch[1:] - fake_pitch[:-1]) ** 2)
                freq_outlier_cost = spectral_regularization_loss(fake_music_g, sample_rate, threshold=outlier_threshold)
                freq_outlier_cost_d = spectral_outlier_discriminator_loss(noisy_real_music, fake_music_g, sample_rate)
                beat_timing_cost = beat_timing_loss(beats_real, beats)

                cp_cost = combined_perceptual_loss(fake_music_g, real_music, sample_rate, scales=[512, 1024, 2048],
                                                   mel_weight=comb_percept_lambda,
                                                   complex_weight=comb_percept_lambda,
                                                   multi_scale_weight=comb_percept_lambda)
                rhythm_cost = rhythm_enforcement_loss(fake_music_g, beats, large_duration_range=large_duration_range, short_duration_range=short_duration_range)
                penalize_peaks_cost = penalize_peaks_loss(fake_music_g, fs=12000, music_band=(200, 8000), peak_height_ratio=peak_height_ratio)

                # Track loss values
                for key, value in locals().items():
                    if key in loss_values:
                        loss_values[key] += value.item()

                batch_count += 1

        # Compute averages
        average_loss_values = {loss_name: max(value / batch_count, 1e-3) for loss_name, value in loss_values.items()}

        # Normalize weights so all initial costs are 1
        loss_weights = {loss_name: .10 / max(abs(value), 1e-3) for loss_name, value in average_loss_values.items()}

        # Check if YAML file exists
        if yaml_filepath and not os.path.exists(yaml_filepath):
            # Save loss_weights to YAML file
            with open(yaml_filepath, 'w') as yaml_file:
                yaml.dump(loss_weights, yaml_file, default_flow_style=False)
            print(f"Created and saved weights to {yaml_filepath}")


    return loss_weights
