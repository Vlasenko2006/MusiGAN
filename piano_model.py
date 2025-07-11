#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:49:01 2025

@author: andrey
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from piano_encoder_decoder_gaussians import VariationalEncoderDecoder



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
        n_channels=128,
        latent_seq_len=None,
        sound_channels=2,
        seq_len=None,
        latent_dim=128 * 4,
        band_count=8,
        use_gaussians = False,
        dropout = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.band_count = band_count
        self.latent_seq_len = latent_seq_len
        self.sound_channels = sound_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.use_gaussians =use_gaussians


        assert self.latent_seq_len != None, "Specify latent_seq_len!"
        assert self.seq_len != None, "Specify seq_len!"

        self.variational_encoder_decoder = VariationalEncoderDecoder(
            input_dim=sound_channels,
            n_channels=n_channels,
            n_seq=self.latent_seq_len,
            latent_dim=self.latent_dim,
            song_duration = self.seq_len,
            use_gaussians = self.use_gaussians
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=2 * self.latent_dim,
            dropout=dropout
        )
        self.num_heads = num_heads
        self.latent_dim = self.latent_dim
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        
        # Adjusting 'self.TokenLinear2' to ensure compatibility with the decoder
        self.TokenLinear = nn.Linear(self.latent_seq_len *  self.latent_dim,  self.latent_dim)  # Adjusted latent_dim to match the decoder dimensions
        
    # Forward method [batch, latent_seq_len, latent_dim]
    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")
        batch_size, input_dim, seq_len = x.shape
        #print("batch_size, input_dim, seq_len  = ", batch_size, input_dim, seq_len)
    
        # Encoder
        mu, logvar = self.variational_encoder_decoder.encoder(x)
        # Reparameterization
        #print("mu.shape = ", mu.shape)
        #print("logvar.shape = ", logvar.shape)
        z = self.variational_encoder_decoder.reparameterize(mu, logvar)  # [batch, latent_seq_len, latent_dim]
        # Transformer
        z = z.permute(1, 0, 2)  # [latent_seq_len, batch, latent_dim]
        transformer_out = self.transformer(z)
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch, latent_seq_len, latent_dim]
    


        # Reshape transformer_out to match TokenLinear2 input
        # print("transformer_out.shape = ", transformer_out.shape)
        # print("LL input shape = ", self.latent_seq_len *  self.latent_dim)
        # print("LL output shape = ", self.latent_dim)
        # print("reshaped transformer_out.shape = ",transformer_out.view(batch_size, self.latent_seq_len *  self.latent_dim).shape)
        # band_latent = self.TokenLinear(transformer_out.view(batch_size, self.latent_seq_len *  self.latent_dim))
        # print("band_latent.shape = ", band_latent.shape)

        # Decoder
        fake_music_g = self.variational_encoder_decoder.decoder(transformer_out)
        #print("fake_music_g.shape = ", fake_music_g.shape)
        #fake_music_g = fake_music_g.view(batch_size, 2, seq_len)
        return (mu, logvar, fake_music_g)
