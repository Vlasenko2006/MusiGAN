import torch
from tqdm import tqdm
from vae_utilities import save_checkpoint, save_sample_as_numpy

from torchaudio.transforms import Spectrogram, AmplitudeToDB




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


def rythm_and_beats_cost(real_data, fake_data, sample_rate=12000, rhythm_weight=0.01, pitch_weight=0.01, btw_weight=1.0):
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
                  beta=.00005,  # beta  = .00005
                  lambda_fm=20 * 2,
                  smoothing_weight = 50 * 5,  # *5
                  smoothing_weight2 = 1 * 4 * 400 * 5,# *5
                  perceptual_weight=0.05,
                  rhythm_weight=0.4 * 8 * 4,  # 0.005
                  pitch_weight=1.5 * 4 * 4,
                  btw_weight=0.3 * 8 * 4,
                  lambda_div=0.75,  # 0.01
                  lambda_rec=1 * 4 * 4,
                  lambda_duration = 30 * 2, # * 2
                  lambda_silence = 100 * 0.1,# *.1
                  lambda_alternation = 1.3 * 20 * 30, #### 1.5 # * 20
                  alpha_duration = 30 * 3,
                  alpha_silence = 100,
                  alpha_alternation = 30,
                  beat_min_duration = 0.4,
                  lambda_monotony = 1e-2,
                  rhythm_dist_weight = 10 * 1e-7,
                  rhythm_dist_weight_g = 2 * 1e-9,
                  large_duration_range=(8000, 12000), 
                  short_duration_range=(4000, 6000),
                  d_steps_per_g_step=5):
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
        "smooth_loss": [],
        "perceptual_loss": [],
        "rb_loss": [],
        "diversity_loss": [],
        "beat_duration_cost": [],  # New generator cost
        "silence_cost": [],        # New generator cost
        "alternation_cost": [],    # New generator cost
        "enforce_rhythm_cost": [],
        "monotone_cost": [],
        "d_loss": [],
        "beat_duration_cost_d": [],  # New discriminator cost
        "silence_cost_d": [],        # New discriminator cost
        "alternation_cost_d": [],     # New discriminator cost
        "enforce_rhythm_cost_d": []
    }

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
            noisy_real_music = real_music + 0.0 * torch.randn_like(real_music)
        
            # Prepare labels once per batch
            real_labels = torch.ones(batch_size, 1, device=device) * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1, device=device)
        
            # === Generator update ===
            for _ in range(1):
                noise_g = torch.randn(batch_size, n_channels, noise_dim, device=device)
                fake_music_g, mu_g, logvar_g = generator(noise_g)
                g_loss_reconstruction = criterion_reconstruction(fake_music_g, real_music)
                g_loss_kl = kl_divergence_loss(mu_g, logvar_g) / batch_size
                g_loss_gan = criterion_gan(discriminator(fake_music_g), real_labels)
                g_loss_stats = compute_chunkwise_stats_loss(fake_music_g, real_music)
                real_features = discriminator.get_intermediate_features(real_music).detach()
                fake_features = discriminator.get_intermediate_features(fake_music_g)
                fm_loss = torch.mean((real_features.mean(0) - fake_features.mean(0))**2)
                smooth_loss = smoothing_loss2(fake_music_g, weight=smoothing_weight2) + smoothing_loss(fake_music_g, weight=smoothing_weight)
                perceptual_loss = stft_loss(fake_music_g, real_music, weight=perceptual_weight)
                rb_loss = rythm_and_beats_cost(real_music, fake_music_g, sample_rate=sample_rate, rhythm_weight=rhythm_weight, pitch_weight=pitch_weight, btw_weight=btw_weight)
                diversity_loss = -torch.mean(torch.var(fake_music_g, dim=0))  # Penalize batch similarity
                
                beats = compute_beats(fake_music_g, sample_rate)
                
                beat_duration = lambda_duration * beat_duration_loss(fake_music_g, sample_rate, min_duration = beat_min_duration, max_duration = 2)
                silence_cost = lambda_silence * silence_loss(fake_music_g, beats) 
                alternation_cost = lambda_alternation * alternation_loss(beats, fake_music_g)
                
                fake_pitch = compute_pitch(fake_music_g, sample_rate)
                monotony_penalty = lambda_monotony * torch.mean((fake_pitch[1:] - fake_pitch[:-1]) ** 2)
               
                
                
                
                rhythm_loss = rhythm_dist_weight_g * rhythm_enforcement_loss(
                              fake_music_g, beats, large_duration_range=large_duration_range, short_duration_range=short_duration_range
                              )

                
                # beats_fake = compute_beats(fake_music_g, sample_rate)
                # beats_real = compute_beats(noisy_real_music, sample_rate)
                # print("Beats (fake):", beats_fake)
                # print("Beats (real):", beats_real)
                
                g_loss = (lambda_rec * g_loss_reconstruction +
                          beta * g_loss_kl + 
                          g_loss_gan + g_loss_stats + 
                          lambda_fm * fm_loss +
                          smooth_loss +
                          perceptual_loss + 
                          rb_loss +
                          lambda_div * diversity_loss +
                          beat_duration +
                          silence_cost +
                          alternation_cost +
                          rhythm_loss + 
                          monotony_penalty)/ accumulation_steps

                g_loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    g_optimizer.step()
                    g_optimizer.zero_grad()
                epoch_loss_g += g_loss.item() * accumulation_steps
                epoch_g_loss_recon_sum += g_loss_reconstruction.item()
                num_g_updates += 1

                # Track generator losses
                loss_trackers["g_loss_reconstruction"].append(lambda_rec * g_loss_reconstruction.item())
                loss_trackers["g_loss_kl"].append(beta * g_loss_kl.item())
                loss_trackers["g_loss_gan"].append(g_loss_gan.item())
                loss_trackers["g_loss_stats"].append(g_loss_stats.item())
                loss_trackers["fm_loss"].append(lambda_fm * fm_loss.item())
                loss_trackers["smooth_loss"].append(smooth_loss.item())
                loss_trackers["perceptual_loss"].append(perceptual_loss.item())
                loss_trackers["rb_loss"].append(rb_loss.item())
                loss_trackers["diversity_loss"].append(lambda_div * diversity_loss.item())
                loss_trackers["beat_duration_cost"].append(beat_duration.item())  # New tracker
                loss_trackers["silence_cost"].append(silence_cost.item())        # New tracker
                loss_trackers["alternation_cost"].append(alternation_cost.item())  # New tracker
                loss_trackers["enforce_rhythm_cost"].append(rhythm_loss.item())  # New tracker  
                loss_trackers["monotone_cost"].append(monotony_penalty.item())  # New tracker                
                
                

            # === Discriminator update ===
            for d_substep in range(d_steps_per_g_step):
                d_real_logits = discriminator(noisy_real_music)
                with torch.no_grad():
                    noise = torch.randn(batch_size, n_channels, noise_dim, device=device)
                    fake_music, _, _ = generator(noise)
                d_fake_logits = discriminator(fake_music)
                
                beats = compute_beats(noisy_real_music, sample_rate)
                fake_beats = compute_beats(fake_music, sample_rate)
                
                beat_duration_d = lambda_duration * beat_duration_loss(noisy_real_music, sample_rate, min_duration = beat_min_duration, max_duration = 2)
                silence_cost_d = lambda_silence * silence_loss(noisy_real_music, beats) 
                alternation_cost_d = lambda_alternation * alternation_loss(beats, fake_music)
                
                detection_penalty = rhythm_dist_weight * rhythm_detection_penalty(
                beats, fake_beats, large_duration_range=large_duration_range, short_duration_range=short_duration_range
                )

                

                d_loss_real = criterion_gan(d_real_logits, real_labels)
                d_loss_fake = criterion_gan(d_fake_logits, fake_labels)
                d_loss = (d_loss_real +
                          d_loss_fake + 
                          beat_duration_d +
                          silence_cost_d +
                          alternation_cost_d +
                          detection_penalty
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
                loss_trackers["enforce_rhythm_cost_d"].append(detection_penalty.item())  # New tracker

        mean_g_loss_reconstruction = epoch_g_loss_recon_sum / num_g_updates if num_g_updates > 0 else 0.0
        d_loss_div = num_d_updates if num_d_updates > 0 else 1
        g_loss_div = num_g_updates if num_g_updates > 0 else 1

        # Print loss summaries every 5 epochs
        if epoch % 5 == 0:
            print(f"\n==== Loss Summary for Epoch {epoch} ====")
            for loss_name, loss_values in loss_trackers.items():
                print(f"{loss_name}: {sum(loss_values) / len(loss_values):.6f}")

        print(f"Epoch {epoch}/{epochs} - D Loss: {epoch_loss_d/d_loss_div:.8f}, G Loss: {epoch_loss_g/g_loss_div:.8f}, Mean Recon: {mean_g_loss_reconstruction:.8f}")

        if (epoch % save_after_nepochs == 0):
            print(" ==== saving music samples and models ==== ")
            target_mean = target_mean.to(fake_music_g.device)
            target_std = target_std.to(fake_music_g.device)
            #fake_music_g_denorm = denormalize_waveform(fake_music_g, target_mean, target_std)
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
            fake_music, _, _ = generator(noise)

            loss = criterion(fake_music, real_music)
            smooth_loss = smoothing_loss(fake_music, weight=0.001)
            perceptual_loss = stft_loss(fake_music, real_music, weight=0.001)
            rb_loss = rythm_and_beats_cost(real_music,fake_music, sample_rate = sample_rate, rhythm_weight = 0.001, pitch_weight =0.001)
            diversity_loss = -torch.mean(torch.var(fake_music, dim=0))  # Penalize batch similarity
            loss += smooth_loss + rb_loss +perceptual_loss + 0.0001 * diversity_loss
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
