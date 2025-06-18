import torch
from tqdm import tqdm            
from loss_functions import rythm_and_beats_loss, compute_pitch, smoothing_loss, stft_loss



def pretrain_generator(generator,
                       train_loader,
                       optimizer,
                       criterion, 
                       device,
                       noise_dim,
                       n_channels=2,
                       sample_rate = 12000,
                       pretrain_epochs=20):
    generator.to(device)
    criterion = torch.nn.SmoothL1Loss()
    #freeze_decoder_weights(generator)  # only if you want to freeze it!
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
            (_, _,_, _, _, _,_, _, _, _, fake_music) = generator(noise) 


            loss = criterion(fake_music, real_music)
            smooth_loss = smoothing_loss(fake_music, weight=0.001)
            perceptual_loss = stft_loss(fake_music, real_music, weight=0.001)
            rb_loss = rythm_and_beats_loss(real_music,fake_music, sample_rate = sample_rate, rhythm_weight = 0.001, pitch_weight =0.001)
            diversity_loss = -torch.mean(torch.var(fake_music, dim=0))  # Penalize batch similarity
            loss += perceptual_loss #smooth_loss + rb_loss +perceptual_loss + 0.0001 * diversity_loss
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
                (_, _,_, _, _, _,_, _, _, _, fake_music) = generator(noise)

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
