import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

FREEZE_ENCODER_DECODER_AFTER = 10  # Number of steps after which encoder-decoder weights are frozen

sf = 4
do = 0.3

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)


# Variational Attention-based neural network with Cosine Decomposition
class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, compression_dim=64 * sf, num_cosines=16):
        super(VariationalAttentionModel, self).__init__()
        
        self.num_cosines = num_cosines  # Number of cosines in the decomposition
        
        # Encoder: Compress the input sequence into amplitudes for cosine decomposition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256 * sf),
            nn.ReLU(),
            nn.Linear(256 * sf, compression_dim * 2)  # Output both mean and log-variance
        )
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=compression_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder: Reconstruct the input sequence from cosine amplitudes
        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, 256 * sf),
            nn.ReLU(),
            nn.Linear(256 * sf, num_cosines)  # Predict amplitudes for num_cosines
        )
        
        # Feed-forward layers for the task-specific output (e.g., audio enhancement)
        self.fc = nn.Sequential(
            nn.Linear(compression_dim, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=do),
            nn.Linear(256 * sf, 256 * sf),
            nn.ReLU(),
            nn.Dropout(p=do),
            nn.Linear(256 * sf, input_dim * 2)  # Output layer
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Pass through the encoder to get mean and log-variance
        encoded = self.encoder(x)  # Shape: [batch_size, seq_len, compression_dim * 2]
        mu, logvar = torch.chunk(encoded, 2, dim=-1)  # Split into mean and log-variance

        # Reparameterize to get latent vector
        z = self.reparameterize(mu, logvar)  # Shape: [batch_size, seq_len, compression_dim]
        
        # Permute dimensions for Transformer (seq_len first)
        z_for_transformer = z.permute(1, 0, 2)  # Shape: [seq_len, batch_size, compression_dim]
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer(z_for_transformer)  # Shape: [seq_len, batch_size, compression_dim]
        
        # Permute back to original dimensions
        transformer_out = transformer_out.permute(1, 0, 2)  # Shape: [batch_size, seq_len, compression_dim]
        
        # Pass through the decoder to predict cosine amplitudes
        cosine_amplitudes = self.decoder(z)  # Shape: [batch_size, seq_len, num_cosines]
        
        # Reconstruct the waveform from the cosine amplitudes
        reconstructed = self.reconstruct_waveform(cosine_amplitudes, x.size(1))  # Shape: [batch_size, seq_len, input_dim]
        
        # Use mean pooling across sequence length for task-specific output
        context_vector = transformer_out.mean(dim=1)  # Shape: [batch_size, compression_dim]
        output = self.fc(context_vector)  # Shape: [batch_size, input_dim * 2]
        
        return reconstructed, output.view(x.size(0), 2, -1), mu, logvar  # Include mu and logvar for VAE loss

    def reconstruct_waveform(self, amplitudes, seq_len, num_channels=2):
        """
        Reconstruct the waveform using cosine decomposition.
        amplitudes: [batch_size, num_cosines]
        seq_len: Length of the input sequence
        num_channels: Number of audio channels (e.g., 2 for stereo audio)
        """
        batch_size, num_cosines = amplitudes.size()  # Ensure amplitudes are [batch_size, num_cosines]
    
        # Generate cosine basis functions
        t = torch.linspace(0, 1, steps=seq_len, device=amplitudes.device).unsqueeze(0).unsqueeze(-1)  # Shape: [1, seq_len, 1]
        frequencies = torch.arange(1, num_cosines + 1, device=amplitudes.device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_cosines]
        cosines = torch.cos(2 * torch.pi * frequencies * t)  # Shape: [1, seq_len, num_cosines]
    
        # Multiply amplitudes with cosine basis functions and sum over num_cosines
        reconstructed_single_channel = torch.sum(amplitudes.unsqueeze(1) * cosines, dim=-1)  # Shape: [batch_size, seq_len]
        print("reconstructed_single_channel.shape", reconstructed_single_channel.shape)
        # Expand to multiple channels (e.g., stereo audio)
        reconstructed = reconstructed_single_channel.unsqueeze(1).repeat(1, num_channels, 1)  # Shape: [batch_size, num_channels, seq_len]
    
        return reconstructed
