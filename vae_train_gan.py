import torch
from tqdm import tqdm
from vae_utilities import save_checkpoint, save_sample_as_numpy
from loss_functions import spectral_regularization_loss, combined_perceptual_loss, rhythm_enforcement_loss, \
    rhythm_detection_penalty, silence_loss, beat_timing_loss, compute_beats, rythm_and_beats_loss, compute_pitch, \
    beat_duration_loss, alternation_loss, smoothing_loss, smoothing_loss2, stft_loss, freeze_decoder_weights, \
    denormalize_waveform, kl_divergence_loss, compute_chunkwise_stats_loss, spectral_outlier_discriminator_loss, \
    penalize_peaks_loss, bandwise_discriminator_loss, bandwise_generator_loss

from split_signal_frequency_bands import merge_band_signals, split_signal_frequency_bands
from compute_weights import normalize_loss_weights

def train_vae_gan(generator,
                  discriminator,
                  g_optimizer, 
                  d_optimizer, 
                  train_loader, 
                  start_epoch,
                  epochs,
                  device,
                  noise_dim,
                  n_channels,
                  sample_rate, 
                  checkpoint_folder,
                  music_out_folder, 
                  accumulation_steps=4,
                  save_after_nepochs=10,
                  freeze_encoder_decoder_after=10,
                  beat_min_duration = 0.4,
                  beat_max_duration = 1.4,
                  large_duration_range=(9000, 14000), 
                  short_duration_range=(4000, 8000),
                  rhythm_weight=0.4,  # 0.005
                  pitch_weight=1.5 * 4,
                  btw_weight=0.3 * 8,
                  comb_percept_lambda = 1,
                  outlier_threshold=10.0,
                  smoothing = 0,
                  peak_height_ratio=0.3,
                  d_steps_per_g_step=7,
                  yaml_filepath=None,
                  batch_size = 8):
    
    generator.to(device)
    discriminator.to(device)

    criterion_gan = torch.nn.BCEWithLogitsLoss()
    criterion_reconstruction = torch.nn.SmoothL1Loss()

    # Initialize dictionaries to track loss values
    loss_trackers = {
        "g_loss_reconstruction": [],
        "g_loss_kl": [],
        "g_loss_gan": [],
        "g_loss_stats": [],
        "fm_loss": [],
        "smoothing_loss": [],
        "smoothing_loss2": [],
        "perceptual_loss": [],
        "diversity_loss": [],
        "beat_duration_cost": [],  # New generator cost
        "silence_cost": [],        # New generator cost
        "alternation_cost": [],    # New generator cost
        "enforce_rhythm_cost": [],
        "cb_percept_cost": [],
        "monotone_cost": [],
        "freq_outlier_cost": [],
        "d_loss": [],
        "beat_duration_cost_d": [],  # New discriminator cost
        "silence_cost_d": [],        # New discriminator cost
        "alternation_cost_d": [],     # New discriminator cost
        "enforce_rhythm_cost_d": [],
        "cb_percept_cost_d": [],
        "rythm_and_beats_cost": [],
        "freq_outlier_cost_d":[],
        "beat_timing_cost_d":[],
        "penalize_peaks_cost":[]
    }

    loss_weights = normalize_loss_weights(generator,
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
                               yaml_filepath = yaml_filepath,
                               batch_fraction = 1)
    
    print(f"Loss weights initialized: {loss_weights}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        generator.train()
        discriminator.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_g_loss_recon_sum = 0.0
        num_g_updates = 0
        num_d_updates = 0

        # Reset loss trackers for the epoch
        for key in loss_trackers:
            loss_trackers[key] = []

        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            # Unpack normalized tensors and stats
            target_norm, target_mean, target_std = data
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()
            noisy_real_music = real_music + 0.05 * torch.randn_like(real_music)
            real_band_signals, _ = split_signal_frequency_bands(noisy_real_music, fs=sample_rate)
        
            # Prepare labels once per batch
            real_labels = torch.ones(batch_size, 1, device=device) * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device)
        
            # === Generator update ===
            for _ in range(1):
                # Forward pass through generator (using new bandwise-aware forward method)
                outputs = generator(noise_g := torch.randn(batch_size, n_channels, noise_dim, device=device)) 
                (reconstructed_Super_Ultra_Low, reconstructed_Ultra_Low, reconstructed_Low_Middle,
                 reconstructed_Low, reconstructed_Middle, reconstructed_High, reconstructed_Ultra_High,
                 mu_g, logvar_g, fake_music_g) = outputs

                # Construct fake_band_signals and fband_signals as in the improved forward method
                band_names = [
                    "Super_Ultra_Low", "Ultra_Low", "Low_Middle", "Low", "Middle", "High", "Ultra_High"
                ]
                reconstructed_bands = [
                    reconstructed_Super_Ultra_Low, reconstructed_Ultra_Low, reconstructed_Low_Middle,
                    reconstructed_Low, reconstructed_Middle, reconstructed_High, reconstructed_Ultra_High
                ]
                fake_band_signals = {k: v for k, v in zip(band_names, reconstructed_bands)}
                fband_signals = {k: torch.fft.rfft(v, n=v.shape[-1], dim=-1) for k, v in fake_band_signals.items()}

                real_features = discriminator.get_intermediate_features(real_music).detach()
                fake_features = discriminator.get_intermediate_features(fake_music_g)        
                beats = compute_beats(fake_music_g, sample_rate)
                fake_pitch = compute_pitch(fake_music_g, sample_rate)          
                
                g_cost_reconstruction = loss_weights["g_cost_reconstruction"] * criterion_reconstruction(fake_music_g, noisy_real_music)
                g_cost_kl = loss_weights["g_cost_kl"] * kl_divergence_loss(mu_g, logvar_g) / batch_size
                g_cost = loss_weights["g_cost"] * criterion_gan(discriminator(fake_music_g), real_labels)
                g_cost_stats = loss_weights["g_cost_stats"] * compute_chunkwise_stats_loss(fake_music_g, real_music)
                fm_cost = loss_weights["fm_cost"] * torch.mean((real_features.mean(0) - fake_features.mean(0))**2)
                smoothing_cost = loss_weights["smoothing_cost"] * smoothing_loss(fake_music_g)
                smoothing_cost2 = loss_weights["smoothing_loss2"] * smoothing_loss2(fake_music_g) 
                perceptual_cost = loss_weights["perceptual_cost"] * stft_loss(fake_music_g, real_music)
                rythm_and_beats_cost = loss_weights["rythm_and_beats_cost"] * rythm_and_beats_loss(real_music, 
                                                                                                   fake_music_g,
                                                                                                   sample_rate=sample_rate,
                                                                                                   rhythm_weight=rhythm_weight,
                                                                                                   pitch_weight=pitch_weight,
                                                                                                   btw_weight=btw_weight)
                diversity_cost =  - 0.0003 * loss_weights["diversity_cost"] * torch.mean(torch.var(fake_music_g, dim=0))  # Penalize batch similarity
                silence_cost = loss_weights["silence_cost"] * silence_loss(fake_music_g, beats) 
                alternation_cost = loss_weights["alternation_cost"] * alternation_loss(beats, fake_music_g)
                beat_duration_cost = loss_weights["beat_duration_cost"] * beat_duration_loss(fake_music_g, sample_rate, min_duration=beat_min_duration, max_duration=beat_max_duration)
                 
                monotony_cost = loss_weights["monotony_cost"] * torch.mean((fake_pitch[1:] - fake_pitch[:-1]) ** 2)
                 
                freq_outlier_cost = loss_weights["freq_outlier_cost"] * spectral_regularization_loss(fake_music_g, sample_rate, threshold=outlier_threshold)
                 
                cp_cost = loss_weights["cb_percept_cost"] * combined_perceptual_loss(fake_music_g, real_music, sample_rate, scales=[512, 1024, 2048], 
                                                     mel_weight=comb_percept_lambda,
                                                     complex_weight=comb_percept_lambda,
                                                     multi_scale_weight=comb_percept_lambda
                                                     )
                rhythm_cost = loss_weights["enforce_rhythm_cost"] * rhythm_enforcement_loss(
                               fake_music_g, beats, large_duration_range=large_duration_range, short_duration_range=short_duration_range
                               )
                penalize_peaks_cost = loss_weights["penalize_peaks_cost"] * penalize_peaks_loss(fake_music_g, fs=sample_rate, music_band=(200, 8000), peak_height_ratio=peak_height_ratio)
                
                # Bandwise generator loss
                g_loss_bandwise, g_terms = bandwise_generator_loss(
                    fake_band_signals, real_band_signals, fband_signals, real_music, 
                    criterion_reconstruction, criterion_gan, discriminator, real_labels, loss_weights
                )

                # Print g_terms for this batch
                print(f"Batch {batch_idx} (Generator) bandwise loss terms: {g_terms}")

                # Combine all losses
                g_loss = (g_cost_reconstruction +
                           g_cost_kl + 
                           g_cost + 
                         #  g_cost_stats + 
                           fm_cost +
                         #  smoothing_cost +
                         #  smoothing_cost2 +
                         #  perceptual_cost + 
                           rythm_and_beats_cost +
                           diversity_cost +
                           beat_duration_cost +
                           silence_cost +
                           alternation_cost +
                           rhythm_cost + 
                           monotony_cost +
                        #   cp_cost + 
                        #   freq_outlier_cost +
                           penalize_peaks_cost +
                           g_loss_bandwise
                           ) / accumulation_steps
                g_loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_cost_reconstruction.item()
                num_g_updates += 1

                # Track generator losses
                loss_trackers["g_loss_reconstruction"].append(g_cost_reconstruction.item())
                loss_trackers["g_loss_kl"].append(g_cost_kl.item())
                loss_trackers["g_loss_gan"].append(g_cost.item())
                loss_trackers["g_loss_stats"].append(g_cost_stats.item())
                loss_trackers["fm_loss"].append(fm_cost.item())
                loss_trackers["smoothing_loss"].append(smoothing_cost.item())
                loss_trackers["smoothing_loss2"].append(smoothing_cost2.item())
                loss_trackers["perceptual_loss"].append(perceptual_cost.item())
                loss_trackers["rythm_and_beats_cost"].append(rythm_and_beats_cost.item())
                loss_trackers["diversity_loss"].append(diversity_cost.item())
                loss_trackers["beat_duration_cost"].append(beat_duration_cost.item())  # New tracker
                loss_trackers["silence_cost"].append(silence_cost.item())        # New tracker
                loss_trackers["alternation_cost"].append(alternation_cost.item())  # New tracker
                loss_trackers["enforce_rhythm_cost"].append(rhythm_cost.item())  # New tracker  
                loss_trackers["monotone_cost"].append(monotony_cost.item())  # New tracker                
                loss_trackers["cb_percept_cost"].append(cp_cost.item())                 
                loss_trackers["freq_outlier_cost"].append(freq_outlier_cost.item()) 
                loss_trackers["penalize_peaks_cost"].append(penalize_peaks_cost.item())                

            # === Discriminator update ===
            for d_substep in range(d_steps_per_g_step):
                d_real_logits = discriminator(noisy_real_music)
                with torch.no_grad():
                    noise = torch.randn(batch_size, n_channels, noise_dim, device=device)
                    outputs = generator(noise)
                    (reconstructed_Super_Ultra_Low, reconstructed_Ultra_Low, reconstructed_Low_Middle,
                    reconstructed_Low, reconstructed_Middle, reconstructed_High, reconstructed_Ultra_High,
                    mu_g, logvar_g, fake_music) = outputs

                    band_names = [
                        "Super_Ultra_Low", "Ultra_Low", "Low_Middle", "Low", "Middle", "High_Hiddle", "High", "Ultra_High"
                    ]
                    reconstructed_bands = [
                        reconstructed_Super_Ultra_Low, reconstructed_Ultra_Low, reconstructed_Low_Middle,
                        reconstructed_Low, reconstructed_Middle, reconstructed_High, reconstructed_Ultra_High
                    ]
                    fake_band_signals = {k: v for k, v in zip(band_names, reconstructed_bands)}
                d_fake_logits = discriminator(fake_music)
                
                beats = compute_beats(noisy_real_music, sample_rate)
                fake_beats = compute_beats(fake_music, sample_rate)
                
                beat_duration_d = loss_weights["beat_duration_cost_d"] * beat_duration_loss(noisy_real_music, sample_rate, min_duration=beat_min_duration, max_duration=2)
                silence_cost_d = loss_weights["silence_cost_d"] * silence_loss(noisy_real_music, beats)
                alternation_cost_d = loss_weights["alternation_cost_d"] * alternation_loss(beats, fake_music)
                
                detection_cost_d = loss_weights["detection_cost_d"] * rhythm_detection_penalty(
                    beats, fake_beats, large_duration_range=large_duration_range, short_duration_range=short_duration_range
                )

                cp_cost_d = loss_weights["cp_cost_d"] * combined_perceptual_loss(fake_music, noisy_real_music, sample_rate, scales=[512, 1024, 2048], 
                                                                                mel_weight=comb_percept_lambda,
                                                                                complex_weight=comb_percept_lambda,
                                                                                multi_scale_weight=comb_percept_lambda)
                freq_outlier_cost_d = loss_weights["freq_outlier_cost_d"] * spectral_outlier_discriminator_loss(noisy_real_music, fake_music, sample_rate)
                beat_timing_cost_d = beat_timing_loss(beats, fake_beats)

                d_loss_real = criterion_gan(d_real_logits, real_labels)
                d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
               
                # Bandwise discriminator loss
                d_loss_bandwise, d_terms = bandwise_discriminator_loss(fake_band_signals, real_band_signals, discriminator, criterion_gan, real_labels, fake_labels, loss_weights)

                # Print d_terms for this batch
                print(f"Batch {batch_idx} (Discriminator) bandwise loss terms: {d_terms}")

                d_loss = (d_loss_real +
                          d_loss_fake + 
                          beat_duration_d +
                          silence_cost_d +
                          alternation_cost_d +
                          detection_cost_d +
                          cp_cost_d +
                          freq_outlier_cost_d  +
                          beat_timing_cost_d +
                          d_loss_bandwise
                          ) / accumulation_steps
                d_loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    d_optimizer.step()
                    d_optimizer.zero_grad()
                epoch_loss_d += d_loss.item() * accumulation_steps
                num_d_updates += 1

                # Track discriminator losses
                loss_trackers["d_loss"].append(d_loss.item())
                loss_trackers["beat_duration_cost_d"].append(beat_duration_d.item())  # New tracker
                loss_trackers["silence_cost_d"].append(silence_cost_d.item())        # New tracker
                loss_trackers["alternation_cost_d"].append(alternation_cost_d.item())  # New tracker
                loss_trackers["enforce_rhythm_cost_d"].append(detection_cost_d.item())  # New tracker
                loss_trackers["cb_percept_cost_d"].append(cp_cost_d.item())  
                loss_trackers["beat_timing_cost_d"].append(beat_timing_cost_d.item())                 

        mean_g_loss_reconstruction = epoch_g_loss_recon_sum / num_g_updates if num_g_updates > 0 else 0.0
        d_loss_div = num_d_updates if num_d_updates > 0 else 1
        g_loss_div = num_g_updates if num_g_updates > 0 else 1

        # Print loss summaries every epoch
        if epoch % 1 == 0:
            print(f"\n==== Loss Summary for Epoch {epoch} ====")
            for loss_name, loss_values in loss_trackers.items():
                if len(loss_values) > 0:
                    print(f"{loss_name}: {sum(loss_values) / len(loss_values):.6f}")
                else:
                    print(f"{loss_name}: No values tracked for this loss.")

        print(f"Epoch {epoch}/{epochs} - D Loss: {epoch_loss_d/d_loss_div:.8f}, G Loss: {epoch_loss_g/g_loss_div:.8f}, Mean Recon: {mean_g_loss_reconstruction:.8f}")

        if (epoch % save_after_nepochs == 0):
            print(" ==== saving music samples and models ==== ")
            target_mean = target_mean.to(fake_music_g.device)
            target_std = target_std.to(fake_music_g.device)
            #save_sample_as_numpy(generator, device, music_out_folder, epoch, noise_dim, n_channels=n_channels, num_samples=10, prefix='')
            save_checkpoint(generator, g_optimizer, epoch, checkpoint_folder, "generator")
            save_checkpoint(discriminator, d_optimizer, epoch, checkpoint_folder, "discriminator")