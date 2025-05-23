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

def compute_chunkwise_stats_loss(fake_music, real_music, num_chunks=100, lambda_mean=0.1, lambda_std=0.1, lambda_max=0.05):
    """
    Splits each audio sample in the batch into `num_chunks` and computes MSE between means, stds, and maxes of fake and real signals per chunk.
    Returns weighted sum of these losses.
    """
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
            target_norm, target_mean, target_std = data  # Only targets, no input
            batch_size = target_norm.size(0)
            real_music = target_norm.to(device).float()
            noisy_real_music = real_music + 0.05 * torch.randn_like(real_music)

            # === Generator update (always) ===
            for _ in range(1):
                noise_g = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
                fake_music_g, mu_g, logvar_g = generator(noise_g)
                real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)
                g_loss_reconstruction = criterion_reconstruction(fake_music_g, real_music)
                g_loss_kl = kl_divergence_loss(mu_g, logvar_g)
                g_loss_gan = criterion_gan(discriminator(fake_music_g), real_labels)

                # --- Chunkwise stats loss ---
                g_loss_stats = compute_chunkwise_stats_loss(fake_music_g, real_music)

                g_loss = (g_loss_reconstruction + beta * g_loss_kl + g_loss_gan + g_loss_stats) / accumulation_steps
                g_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_loss_reconstruction.item()
                num_g_updates += 1

            # === Discriminator update (every other batch) ===
            if batch_idx % 1 == 0:
                noise = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
                fake_music, mu, logvar = generator(noise)
                real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)
                fake_labels = torch.zeros(batch_size, 1).to(device).float()
                d_real_logits = discriminator(noisy_real_music.float())
                d_fake_logits = discriminator(fake_music.detach().float())

                # Debug: Print mean and std of discriminator outputs
               # print(f"Batch {batch_idx}:")
               # print("  d_real_logits mean:", d_real_logits.mean().item(), "std:", d_real_logits.std().item())
               # print("  d_fake_logits mean:", d_fake_logits.mean().item(), "std:", d_fake_logits.std().item())
               # print("  Real labels unique:", real_labels.unique().cpu().numpy(), "Fake labels unique:", fake_labels.unique().cpu().numpy())

                # Debug: Print shape of logits and labels (should match for BCEWithLogitsLoss)
               # print("  d_real_logits shape:", d_real_logits.shape, "real_labels shape:", real_labels.shape)
               # print("  d_fake_logits shape:", d_fake_logits.shape, "fake_labels shape:", fake_labels.shape)

                d_loss_real = criterion_gan(d_real_logits, real_labels)
                d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / accumulation_steps

                # Debug: Print discriminator losses
               # print("  d_loss_real:", d_loss_real.item(), "d_loss_fake:", d_loss_fake.item(), "d_loss:", d_loss.item())

                d_loss.backward()

                # Debug: Print gradient norms for discriminator parameters
                #for name, param in discriminator.named_parameters():
                #    if param.grad is not None:
                #        print(f"  Grad norm for {name}: {param.grad.norm().item()}")

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    d_optimizer.step()
                    d_optimizer.zero_grad()
                    # Debug: Print after optimizer step to see if parameters have changed
               #     print("  Discriminator optimizer step performed.")

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

            # Generate fake samples using the current generator
            noise = noise_fun(batch_size=batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
            with torch.no_grad():
                fake_music, _, _ = generator(noise)

            # Labels: real=1-smoothing, fake=0
            real_labels = torch.ones(batch_size, 1).to(device).float() * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1).to(device).float()

            # Discriminator predictions
            d_real_logits = discriminator(real_music)
            d_fake_logits = discriminator(fake_music)

            # Compute BCE losses
            d_loss_real = criterion(d_real_logits, real_labels)
            d_loss_fake = criterion(d_fake_logits, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2  # Average for reporting

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

            epoch_loss += d_loss.item()

        print(f"Pretraining Discriminator Epoch {epoch + 1}/{pretrain_epochs} - Loss: {epoch_loss / len(train_loader):.6f}")
