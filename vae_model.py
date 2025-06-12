import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vae_encoder_decoder import VariationalEncoderDecoder

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
    def __init__(self, num_heads=8, num_layers=1, n_channels=96, n_seq=3, sound_channels=2, seq_len=120000, latent_dim=128 * 8): # *4
        super(VariationalAttentionModel, self).__init__()
        self.latent_dim = latent_dim
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
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")
        batch_size, input_dim, seq_len = x.shape
        mu, logvar = self.variational_encoder_decoder.encoder(x)   # [batch, latent_seq_len, latent_dim]
        z = self.variational_encoder_decoder.reparameterize(mu, logvar)  # [batch, latent_seq_len, latent_dim]
        z = z.permute(1, 0, 2)  # [latent_seq_len, batch, latent_dim]
        #print("z.shape", z.shape)
        transformer_out = self.transformer(z)  # [latent_seq_len, batch, latent_dim]
        #print("transformer_out.shape", transformer_out.shape)
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch, latent_seq_len, latent_dim]
        reconstructed_Super_Ultra_Low = self.variational_encoder_decoder.decoder(transformer_out)  # (decoder must accept sequence!)
        reconstructed_Ultra_Low = self.variational_encoder_decoder.decoder(transformer_out) 
        reconstructed_Low = self.variational_encoder_decoder.decoder(transformer_out)  
        reconstructed_Low_Middle = self.variational_encoder_decoder.decoder(transformer_out) 
        reconstructed_Middle = self.variational_encoder_decoder.decoder(transformer_out)
        reconstructed_High = self.variational_encoder_decoder.decoder(transformer_out)  
        reconstructed_Ultra_High = self.variational_encoder_decoder.decoder(transformer_out) 
        return reconstructed_Super_Ultra_Low, reconstructed_Ultra_Low, reconstructed_Low_Middle, reconstructed_Low, reconstructed_Middle, reconstructed_High, reconstructed_Ultra_High,mu, logvar
