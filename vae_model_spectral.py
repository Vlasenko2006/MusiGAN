import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vae_encoder_decoder import VariationalEncoderDecoder
from split_signal_frequency_bands import merge_band_signals



class AttentionAggregation(nn.Module):
    def __init__(self, latent_dim, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads)
        self.latent_dim = latent_dim

    def forward(self, transformer_out):
        # transformer_out: [batch, latent_seq_len, latent_dim]
        transformer_out = transformer_out.permute(1, 0, 2)  # [latent_seq_len, batch, latent_dim]
        query = transformer_out.mean(dim=0, keepdim=True)  # Use the mean as the query vector [1, batch, latent_dim]
        attended_tokens, _ = self.attention(query, transformer_out, transformer_out)  # Perform attention
        attended_tokens = attended_tokens.squeeze(0)  # [batch, latent_dim]
        return attended_tokens


class TokenPooling(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, transformer_out):
        # transformer_out: [batch, latent_seq_len, latent_dim]
        max_pool = transformer_out.max(dim=1).values  # Max pooling across sequence dimension
        min_pool = transformer_out.min(dim=1).values  # Min pooling across sequence dimension
        pooled_tokens = torch.cat([max_pool, min_pool], dim=-1)  # Concatenate pooled features [batch, latent_dim * 2]
        return pooled_tokens



class LearnableAggregation(nn.Module):
    def __init__(self, latent_seq_len, latent_dim):
        super().__init__()
        self.latent_seq_len = latent_seq_len
        self.latent_dim = latent_dim
        self.weights = nn.Linear(latent_seq_len, latent_seq_len)  # Learnable weights for each token

    def forward(self, transformer_out):
        # transformer_out: [batch, latent_seq_len, latent_dim]
        transformer_out = transformer_out.permute(0, 2, 1)  # [batch, latent_dim, latent_seq_len]

        # Apply weights
        batch_size, latent_dim, latent_seq_len = transformer_out.shape
        if latent_seq_len != self.latent_seq_len:
            raise ValueError(f"Expected latent_seq_len={self.latent_seq_len}, but got {latent_seq_len}")

        weights = self.weights(transformer_out.reshape(batch_size * latent_dim, latent_seq_len))  # Flatten batch and latent_dim
        weights = weights.reshape(batch_size, latent_dim, latent_seq_len)  # Restore the original shape

        # Perform softmax and weighted summation
        weights = torch.softmax(weights, dim=-1)  # Normalize weights
        aggregated_tokens = torch.bmm(weights, transformer_out)  # Weighted summation
        aggregated_tokens = aggregated_tokens.sum(dim=1)  # Aggregate into a single vector [batch, latent_dim]

        return aggregated_tokens



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
        n_channels=64,
        n_seq=1,
        sound_channels=2,
        seq_len=120000,
        latent_dim=128 * 1 * 1,
        band_count=8,
        latent_seq_len = 1000
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
            dropout=0.
        )
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.TokenLinear = nn.Linear(latent_dim, latent_dim)  # Replace TokenPooling with nn.Linear
        #self.TokenLinear2 = nn.Linear(latent_seq_len * latent_dim, latent_dim)  # Replace TokenPooling with nn.Linear
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # One head (projection/decoder) per band; each is a linear layer mapping from latent_dim to output size
        self.band_heads = nn.ModuleDict({
            name: nn.Linear(latent_dim, sound_channels * seq_len)  # assumes flattening channels and seq_len
            for name in [
                "Super_Ultra_Low", "Ultra_Low", "Low", "Low_Middle", "Middle", "High_Middle", "High", "Ultra_High"
            ]
        })
        
        
        self.TokenLinear2 = nn.ModuleDict({
            name: nn.Linear(latent_seq_len * latent_dim, latent_dim) # assumes flattening channels and seq_len
            for name in [
                "Super_Ultra_Low", "Ultra_Low", "Low", "Low_Middle", "Middle", "High_Middle", "High", "Ultra_High"
            ]
        })
        
              
        
        self.band_names = list(self.band_heads.keys())
        self.sound_channels = sound_channels
        self.seq_len = seq_len

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")
        batch_size, input_dim, seq_len = x.shape

        # Encoder
        mu, logvar = self.variational_encoder_decoder.encoder(x)
        # Reparameterization
        z = self.variational_encoder_decoder.reparameterize(mu, logvar)  # [batch, latent_seq_len, latent_dim]
        # Transformer
        z = z.permute(1, 0, 2)  # [latent_seq_len, batch, latent_dim]
        transformer_out = self.transformer(z)
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch, latent_seq_len, latent_dim]

        # Band Reconstruction
        reconstructed_bands = []
        fband_signals = {}
           
        # Use nn.Linear with reduced dimensions
        
        for band_name in self.band_names:
            band_latent = self.TokenLinear2[band_name](transformer_out.flatten(start_dim=1, end_dim=-1))  # Use nn.Linear with reduced dimensions
            
            band_wave = self.band_heads[band_name](band_latent)
            band_wave = band_wave.view(batch_size, self.sound_channels, self.seq_len)
            reconstructed_bands.append(band_wave)
            fband_signals[band_name] = torch.fft.rfft(band_wave, n=band_wave.shape[-1], dim=-1)

        fake_music_g = merge_band_signals(fband_signals, fs=12000)
        return (*reconstructed_bands, mu, logvar, fake_music_g)



