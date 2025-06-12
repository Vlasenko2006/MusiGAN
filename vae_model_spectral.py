import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vae_encoder_decoder import VariationalEncoderDecoder
from split_signal_frequency_bands import merge_band_signals


FREEZE_ENCODER_DECODER_AFTER = 10

sf = 4
do = 0.

# === Normalization Helpers ===
def normalize_waveform(waveform):
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True) + 1e-8
    return (waveform - mean) / std, mean, std

def denormalize_waveform(waveform, mean, std):
    return waveform * std + mean

# Dataset class: only yields normalized real stereo music + stats for denormalization
class AudioDataset(Dataset):
    """
    Yields:
      target_norm: [2, seq_len] - normalized stereo audio
      target_mean: [2, 1] - mean per channel
      target_std: [2, 1] - std per channel
    For GAN use: input is generated noise, not loaded from dataset.
    """
    def __init__(self, data):
        """
        data: numpy array of shape (N, 2, seq_len)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx]  # shape: (2, seq_len)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        target_norm, target_mean, target_std = normalize_waveform(target_tensor)
        return target_norm, target_mean, target_std

class VariationalAttentionModel(nn.Module):
    def __init__(
        self,
        num_heads=8,
        num_layers=1,
        n_channels=96,
        n_seq=3,
        sound_channels=2,
        seq_len=120000,
        latent_dim=128 * 8,
        band_count=7
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.band_count = band_count

        self.variational_encoder_decoder = VariationalEncoderDecoder(
            input_dim=sound_channels,
            n_channels=n_channels,
            n_seq=n_seq,
            latent_dim=latent_dim
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=2 * latent_dim,
            dropout=0.0
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # One head (projection/decoder) per band; each is a linear layer mapping from latent_dim to output size
        self.band_heads = nn.ModuleDict({
            name: nn.Linear(latent_dim, sound_channels * seq_len)  # assumes flattening channels and seq_len
            for name in [
                "Super_Ultra_Low", "Ultra_Low", "Low", "Low_Middle", "Middle", "High_Hiddle" ,"High", "Ultra_High"
            ]
        })
        self.band_names = list(self.band_heads.keys())
        self.sound_channels = sound_channels
        self.seq_len = seq_len

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")
        batch_size, input_dim, seq_len = x.shape

        mu, logvar = self.variational_encoder_decoder.encoder(x)
        z = self.variational_encoder_decoder.reparameterize(mu, logvar)  # [batch, latent_seq_len, latent_dim]
        z = z.permute(1, 0, 2)  # [latent_seq_len, batch, latent_dim]
        transformer_out = self.transformer(z)
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch, latent_seq_len, latent_dim]

        # For simplicity, take only the first token (CLS-style) for each band head
        band_latent = transformer_out[:, 0, :]  # [batch, latent_dim]
        reconstructed_bands = []
        fband_signals = {}
        for band_name in self.band_names:
            # Predict flattened output, then reshape to [batch, sound_channels, seq_len]
            band_wave = self.band_heads[band_name](band_latent)
            band_wave = band_wave.view(batch_size, self.sound_channels, self.seq_len)
            reconstructed_bands.append(band_wave)
            # Frequency domain
            fband_signals[band_name] = torch.fft.rfft(band_wave, n=band_wave.shape[-1], dim=-1)

        fake_music_g = merge_band_signals(fband_signals, fs=12000)

        return (*reconstructed_bands, mu, logvar, fake_music_g)