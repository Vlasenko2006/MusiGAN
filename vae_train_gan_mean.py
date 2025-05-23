import torch
from tqdm import tqdm
from vae_utilities import save_checkpoint, save_sample_as_numpy
from noise_fun import noise_fun

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
                  smoothing=0.1,
                  save_after_nepochs=10,
                  freeze_encoder_decoder_after=10,
                  beta=.0000251):
    generator.to(device)
    discriminator.to(device)

    criterion_gan = torch.nn.BCEWithLogitsLoss()
    criterion_reconstruction = torch.nn.SmoothL1Loss()
    freeze_decoder_weights(generator) #if you want full fine-tuning!

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        generator.train()
        discriminator.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_g_loss_recon_sum = 0.0
        num_g_updates = 0
        num_d_updates = 0

        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            # Unpack normalized tensors and stats
            target_norm, target_mean, target_std = data  # UPDATED: Only targets, no input
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()
            noisy_real_music = real_music + 0.05 * torch.randn_like(real_music)

            # === Generator update (always) ===
            for _ in range(2):
                noise_g = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
                fake_music_g, mu_g, logvar_g = generator(noise_g)
                real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)
                g_loss_reconstruction = criterion_reconstruction(fake_music_g, real_music)
                g_loss_kl = kl_divergence_loss(mu_g, logvar_g)
                g_loss_gan = criterion_gan(discriminator(fake_music_g), real_labels)
                g_loss = (g_loss_reconstruction + beta * g_loss_kl + g_loss_gan) / accumulation_steps
                g_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_loss_reconstruction.item()
                num_g_updates += 1

            # === Discriminator update (every other batch) ===
            if batch_idx % 2 == 0:
                noise = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
                fake_music, mu, logvar = generator(noise)
                real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)
                fake_labels = torch.zeros(batch_size, 1).to(device).float()
                d_real_logits = discriminator(noisy_real_music.float())
                d_fake_logits = discriminator(fake_music.detach().float())
                d_loss_real = criterion_gan(d_real_logits, real_labels)
                d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / accumulation_steps
                d_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    d_optimizer.step()
                    d_optimizer.zero_grad()
                epoch_loss_d += d_loss.item() * accumulation_steps
                num_d_updates += 1

        mean_g_loss_reconstruction = epoch_g_loss_recon_sum / num_g_updates if num_g_updates > 0 else 0.0
        d_loss_div = num_d_updates if num_d_updates > 0 else 1
        g_loss_div = num_g_updates if num_g_updates > 0 else 1

        print(f"Epoch {epoch}/{epochs} - D Loss: {epoch_loss_d/d_loss_div:.8f}, G Loss: {epoch_loss_g/g_loss_div:.8f}, Mean Recon: {mean_g_loss_reconstruction:.8f}")

        if (epoch % save_after_nepochs == 0):
            print(" ==== saving music samples and models ==== ")
            # If you want to denormalize before saving, do it here:
            target_mean = target_mean.to(fake_music_g.device)
            target_std = target_std.to(fake_music_g.device)
            fake_music_g_denorm = denormalize_waveform(fake_music_g, target_mean, target_std)
            save_sample_as_numpy(generator, device, music_out_folder, epoch, noise_dim, n_channels=n_channels, num_samples=10, prefix='')
            save_checkpoint(generator, g_optimizer, epoch, checkpoint_folder, "generator")
            save_checkpoint(discriminator, d_optimizer, epoch, checkpoint_folder, "discriminator")

def pretrain_generator(generator,
                       train_loader,
                       optimizer,
                       criterion, 
                       device,
                       noise_dim,
                       n_channels=2,
                       pretrain_epochs=20):
    generator.to(device)
    criterion = torch.nn.SmoothL1Loss()
    freeze_decoder_weights(generator)  # only if you want to freeze it!
    for epoch in range(pretrain_epochs):
        print(f"Pretraining Generator Epoch {epoch + 1}/{pretrain_epochs}")
        generator.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Pretraining"):
            # Unpack as targets only
            target_norm, target_mean, target_std = batch
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()

            noise = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
            fake_music, _, _ = generator(noise)

            loss = criterion(fake_music, real_music)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Pretraining Generator Epoch {epoch + 1}/{pretrain_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")
