import torch
from tqdm import tqdm
from vae_utilities import save_checkpoint, save_sample_as_numpy
from noise_fun import noise_fun
import librosa


def rythm_and_beats_cost(real_data,fake_data, rhythm_weight = 0.01, pitch_weight = 0.01):
    # Rhythm Loss
    real_beats = librosa.beat.beat_track(y=real_data)
    fake_beats = librosa.beat.beat_track(y=fake_data)
    rhythm_loss = torch.mean(torch.abs(real_beats - fake_beats))
    
    # Pitch Loss
    real_pitch = librosa.core.piptrack(y=real_data)
    fake_pitch = librosa.core.piptrack(y=fake_data)
    pitch_loss = torch.mean(torch.abs(real_pitch - fake_pitch))
    rythm_and_beats_loss =  rhythm_weight * rhythm_loss + pitch_weight * pitch_loss

    return rythm_and_beats_loss


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
                  beta=.0005, # beta  = .00005
                  lambda_fm = 20,
                  smoothing_weight = 0.0075,
                  perceptual_weight = 0.5,
                  rhythm_weight = 0.01,
                  pitch_weight = 0.01,
                  lambda_div= 0.01,
                  d_steps_per_g_step=25):
    generator.to(device)
    discriminator.to(device)

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

        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            # Unpack normalized tensors and stats
            target_norm, target_mean, target_std = data
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()
            noisy_real_music = real_music + 0.05 * torch.randn_like(real_music)
        
            # Prepare labels once per batch
            real_labels = torch.ones(batch_size, 1, device=device) * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device)
        
            # === Generator update (twice per batch) ===
            for _ in range(2):
                noise_g = torch.randn(batch_size, n_channels, noise_dim, device=device)
                fake_music_g, mu_g, logvar_g = generator(noise_g)
                g_loss_reconstruction = criterion_reconstruction(fake_music_g, real_music)
                g_loss_kl = kl_divergence_loss(mu_g, logvar_g) / batch_size
                g_loss_gan = criterion_gan(discriminator(fake_music_g), real_labels)
                g_loss_stats = compute_chunkwise_stats_loss(fake_music_g, real_music)
                real_features = discriminator.get_intermediate_features(real_music).detach()
                fake_features = discriminator.get_intermediate_features(fake_music_g)
                fm_loss = torch.mean((real_features.mean(0) - fake_features.mean(0))**2)
                smooth_loss = smoothing_loss2(fake_music_g, weight=smoothing_weight)
                perceptual_loss = stft_loss(fake_music_g, real_music, weight=perceptual_weight)
                rb_loss = rythm_and_beats_cost(real_data,fake_data, rhythm_weight = rhythm_weight, pitch_weight = pitch_weight)
                diversity_loss = -torch.mean(torch.var(fake_data, dim=0))  # Penalize batch similarity
                
                g_loss = (g_loss_reconstruction +
                          beta * g_loss_kl + 
                          g_loss_gan + g_loss_stats + 
                          lambda_fm * fm_loss +
                          smooth_loss +
                          perceptual_loss + 
                          rb_loss +
                          diversity_loss) / accumulation_steps

                g_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_loss_reconstruction.item()
                num_g_updates += 1

            # === Discriminator update (n times per generator update) ===
            for d_substep in range(d_steps_per_g_step):
                d_real_logits = discriminator(noisy_real_music)
                with torch.no_grad():
                    noise = torch.randn(batch_size, n_channels, noise_dim, device=device)
                    fake_music, _, _ = generator(noise)
                d_fake_logits = discriminator(fake_music)

                # === Minibatch Discrimination Integration ===
                # The discriminator automatically handles minibatch discrimination internally
                # No need to modify the loss calculation here

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

            noise = torch.randn(batch_size, n_channels,noise_dim ,device=device)# noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
            fake_music, _, _ = generator(noise)

            loss = criterion(fake_music, real_music)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Pretraining Generator Epoch {epoch + 1}/{pretrain_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")



def pretrain_discriminator(discriminator,
                           generator,
                           train_loader,
                           optimizer,
                           device,
                           noise_dim,
                           n_channels=2,
                           pretrain_epochs=5,
                           smoothing=0.1):
    """
    Pretrains the discriminator to distinguish between real and generated (from noise) audio,
    before starting adversarial training.
    """
    discriminator.to(device)
    generator.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(pretrain_epochs):
        print(f"Pretraining Discriminator Epoch {epoch + 1}/{pretrain_epochs}")
        discriminator.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Pretraining D"):
            target_norm, target_mean, target_std = batch
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()

            # === Efficient: Prepare labels ONCE per batch ===
            real_labels = torch.full((batch_size, 1), 1 - smoothing, device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # === Efficient: Generate fake samples using generator (no grad) ===
            with torch.no_grad():
                noise = torch.randn(batch_size, n_channels,noise_dim ,device=device) #noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
                fake_music, _, _ = generator(noise)

            # === Discriminator predictions ===
            d_real_logits = discriminator(real_music)
            d_fake_logits = discriminator(fake_music)

            # === Compute BCE losses and optimize ===
            d_loss_real = criterion(d_real_logits, real_labels)
            d_loss_fake = criterion(d_fake_logits, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

            epoch_loss += d_loss.item()

        print(f"Pretraining Discriminator Epoch {epoch + 1}/{pretrain_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")
