#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:57:57 2025

@author: andrey
"""

import torch
from tqdm import tqdm
from vae_utilities import save_checkpoint
from loss_functions import kl_divergence_loss, no_seconds_loss, no_silence_loss, \
    melody_loss, melody_loss_d, silence_loss_d, compute_std_loss_d, min_max_stats_loss_d, kl_loss_songs_d
    

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
                  smoothing = 0,
                  d_steps_per_g_step=1,
                  batch_size = 8,
                  grad_clip_value = 1.,
                  freeze_discriminator = False,
                  loss_trackers = {},
                  loss_weights = {} 
                  ):
    
    generator.to(device)
    discriminator.to(device)
    
    if freeze_discriminator:
        for param in discriminator.parameters():
            param.requires_grad = False

    criterion_gan = torch.nn.BCEWithLogitsLoss()
    criterion_reconstruction = torch.nn.SmoothL1Loss()

    
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
        
            # Prepare labels once per batch
            real_labels = torch.ones(batch_size, 1, device=device) * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device)
        
            # === Generator update ===
            for _ in range(2):
                # Forward pass through generator (using new bandwise-aware forward method)
                outputs = generator(noise_g := torch.randn(batch_size, n_channels, noise_dim, device=device)) 
                (mu_g, logvar_g, fake_music_g) = outputs
                g_cost_kl = loss_weights["g_cost_kl"] * kl_divergence_loss(mu_g, logvar_g) / batch_size

                # Construct fake_band_signals and fband_signals as in the improved forward method                                                                                                 pitch_weight=pitch_weight,

                # Combine all losses
                g_cost_reconstruction = loss_weights["g_cost_reconstruction"] * criterion_reconstruction(fake_music_g, noisy_real_music)
                g_no_seconds_cost  = loss_weights["g_no_seconds_cost"] *  no_seconds_loss(fake_music_g)
                g_cost = loss_weights["g_cost"] * criterion_gan(discriminator(fake_music_g), real_labels)
                g_no_silence_cost = loss_weights["g_no_silence_cost"] * no_silence_loss(fake_music_g, 
                                                                                        keys=33, 
                                                                                        epsilon=0.001, 
                                                                                        silence_threshold=0.05, 
                                                                                        sharpness_threshold=0.1, 
                                                                                        sharpness_scaling=-0.001
                                                                                        )
                g_melody_cost = loss_weights["g_melody_cost"] * melody_loss(fake_music_g)
                g_compute_std_cost = loss_weights["g_compute_std_cost"] * compute_std_loss_d(fake_music_g, noisy_real_music)
                g_min_max_stats_cost = loss_weights["g_min_max_stats_cost"] * min_max_stats_loss_d(fake_music_g, noisy_real_music)
                g_melody_cost_2 = loss_weights["g_melody_cost_2"] * melody_loss_d(fake_music_g, noisy_real_music)
                
                
                g_loss = (g_cost_reconstruction +
                          g_cost +
                          g_cost_kl #+
                     #     g_no_seconds_cost +
                         # g_no_silence_cost +
                    #      g_melody_cost #+
                         # g_melody_cost_2 +
                        #  g_compute_std_cost + 
                        #  g_min_max_stats_cost
                           ) / accumulation_steps
              #  torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_value)
                g_loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_cost_reconstruction.item()
                num_g_updates += 1

                # Track generator losses
                # Add logs for bandwise generator losses

                if isinstance(g_cost_reconstruction, torch.Tensor):
                    loss_trackers["g_loss_reconstruction"].append(g_cost_reconstruction.item())
                else:
                    loss_trackers["g_loss_reconstruction"].append(g_cost_reconstruction)

                if isinstance(g_cost_kl, torch.Tensor):
                    loss_trackers["g_loss_kl"].append(g_cost_kl.item())
                else:
                    loss_trackers["g_loss_kl"].append(g_cost_kl)

                if isinstance(g_cost, torch.Tensor):
                    loss_trackers["g_loss_gan"].append(g_cost.item())
                else:
                    loss_trackers["g_loss_gan"].append(g_cost)

                if isinstance(g_cost, torch.Tensor):
                    loss_trackers["g_no_seconds_loss"].append(g_no_seconds_cost.item())
                else:
                    loss_trackers["g_no_seconds_loss"].append(g_no_seconds_cost)
                    
                if isinstance(g_cost, torch.Tensor):
                    loss_trackers["g_no_silence_loss"].append(g_no_silence_cost.item())
                else:
                    loss_trackers["g_no_silence_loss"].append(g_no_silence_cost)
                    
                if isinstance(g_cost, torch.Tensor):
                    loss_trackers["g_melody_loss"].append(g_melody_cost.item())
                else:
                    loss_trackers["g_melody_loss"].append(g_melody_cost)       
                    
                    
                    
                if isinstance(g_melody_cost_2, torch.Tensor):
                    loss_trackers["g_melody_loss_2"].append(g_melody_cost_2.item())
                else:
                    loss_trackers["g_melody_loss_2"].append(g_melody_cost_2) 
                    
                if isinstance(g_compute_std_cost, torch.Tensor):
                        loss_trackers["g_compute_std_loss"].append(g_compute_std_cost.item())
                else:
                    loss_trackers["g_compute_std_loss"].append(g_compute_std_cost)
                    
           
            # === Discriminator update ===
            for d_substep in range(d_steps_per_g_step):
                # Recompute fake_music to ensure fresh computation graph
                noise = torch.randn(batch_size, n_channels, noise_dim, device=device)
                outputs = generator(noise)  # Forward pass through generator
                (mu_g, logvar_g, fake_music) = outputs  # Recompute fake_music
            
                # Discriminator forward pass
                d_real_logits = discriminator(noisy_real_music)
                d_fake_logits = discriminator(fake_music)
            
                # Discriminator losses
                d_loss_real = criterion_gan(d_real_logits, real_labels)
                d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
            
                # Additional discriminator costs
                d_silence_cost = loss_weights["d_silence_cost"] * silence_loss_d(fake_music, noisy_real_music, d_fake_logits = d_fake_logits)
                d_melody_cost = loss_weights["d_melody_cost"] * melody_loss_d(fake_music, noisy_real_music, d_fake_logits = d_fake_logits)
                d_compute_std_cost = loss_weights["d_compute_std_cost"] * compute_std_loss_d(fake_music, noisy_real_music, d_fake_logits = d_fake_logits)
            
            
                d_min_max_stats_cost = loss_weights["d_min_max_stats_cost"] * min_max_stats_loss_d(fake_music, noisy_real_music,d_fake_logits = d_fake_logits)
                d_kl_song_cost = loss_weights["d_kl_song_cost"] * kl_loss_songs_d(noisy_real_music, fake_music, d_fake_logits=None, epsilon=1e-6)
                # Combine all discriminator losses
                d_loss = (d_loss_real +
                          d_loss_fake +
                          d_silence_cost +
                          d_melody_cost +
                          d_compute_std_cost +
                  #        d_min_max_stats_cost +
                          d_kl_song_cost
                          ) / accumulation_steps
            
                # Backward pass without retain_graph (graph is fresh)
             #   torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_value)
            
                # Optimizer step
                if not freeze_discriminator:
                    d_loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        d_optimizer.step()
                        d_optimizer.zero_grad()
            
                # Track losses
                epoch_loss_d += d_loss.item() * accumulation_steps
                num_d_updates += 1
                
                
                
                if isinstance(d_kl_song_cost, torch.Tensor):
                    loss_trackers["d_kl_songs_loss"].append(d_kl_song_cost.item())
                else:
                    loss_trackers["d_kl_songs_loss"].append(d_kl_song_cost)
                
            
                if isinstance(d_loss, torch.Tensor):
                    loss_trackers["d_loss"].append(d_loss.item())
                else:
                    loss_trackers["d_loss"].append(d_loss)
            
                if isinstance(d_silence_cost, torch.Tensor):
                    loss_trackers["d_silence_loss"].append(d_silence_cost.item())
                else:
                    loss_trackers["d_silence_loss"].append(d_silence_cost)
            
                if isinstance(d_melody_cost, torch.Tensor):
                    loss_trackers["d_melody_loss"].append(d_melody_cost.item())
                else:
                    loss_trackers["d_melody_loss"].append(d_melody_cost)
                    
                if isinstance(d_min_max_stats_cost, torch.Tensor):
                    loss_trackers["d_min_max_stats_loss"].append(d_min_max_stats_cost.item())
                else:
                    loss_trackers["d_min_max_stats_loss"].append(d_min_max_stats_cost)
                    
                if isinstance(d_compute_std_cost, torch.Tensor):
                    loss_trackers["d_compute_std_loss"].append(d_compute_std_cost.item())
                else:
                    loss_trackers["d_compute_std_loss"].append(d_compute_std_cost)
                                    
                                

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
