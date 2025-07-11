#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:09:30 2025

@author: andrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class VariationalEncoderDecoder(nn.Module):
    def __init__(self, input_dim,
                 n_channels=64,
                 n_seq=1000, 
                 latent_dim=128,
                 song_duration=16500,
                 dropout_rate=0.2, 
                 use_gaussians = False
                 ):
        
        super(VariationalEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_seq = n_seq
        self.song_duration = song_duration
        self.eps = 10e-12
        self.use_gaussians = use_gaussians
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Encoder: Conv1D, BatchNorm, and Pooling layers
        self.encoder_conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=128, kernel_size=9, stride=2, padding=4
        )
        self.encoder_bn1 = nn.BatchNorm1d(128, eps=self.eps)
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder_conv2 = nn.Conv1d(
            in_channels=128, out_channels=n_channels, kernel_size=7, stride=2, padding=3
        )
        self.encoder_bn2 = nn.BatchNorm1d(n_channels, eps=self.eps)
        self.final_pooling = nn.AdaptiveAvgPool1d(self.n_seq)

        self.fc_mu = nn.Linear(n_channels, latent_dim)
        self.fc_logvar = nn.Linear(n_channels, latent_dim)

        # Decoder
        self.latent_to_decoder = nn.Linear(latent_dim, n_channels)
        self.latent_to_gaussians = nn.Linear(latent_dim * self.n_seq, 8)
        self.latent_to_gaussians_activation = nn.ReLU()
                
        
        

        self.decoder_conv1 = nn.ConvTranspose1d(
            in_channels=n_channels, out_channels=128, kernel_size=7, stride=2, padding=3, output_padding=1
        )
        self.decoder_bn1 = nn.BatchNorm1d(128, eps=self.eps)

        self.unpooling1 = nn.ConvTranspose1d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2
        )
        self.decoder_bn2 = nn.BatchNorm1d(128, eps=self.eps)

        self.decoder_conv2 = nn.ConvTranspose1d(
            in_channels=128, out_channels=128, kernel_size=3, stride=5, padding=1, output_padding=4
        )
        self.decoder_bn3 = nn.BatchNorm1d(128, eps=self.eps)

        self.decoder_conv3 = nn.ConvTranspose1d(
            in_channels=128, out_channels=input_dim, kernel_size=9, stride=2, padding=4, output_padding=1
        )
        self.fp = nn.AdaptiveAvgPool1d(self.song_duration)

    def encoder(self, x):
        """
        Encoder: Takes input and encodes it into a sequence of latent vectors suitable for transformer input.
        
        Outputs mean (mu) and log-variance (logvar) for the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu: Mean of the latent space. [batch_size, n_seq, latent_dim]
                - logvar: Log-variance of the latent space. [batch_size, n_seq, latent_dim]
        """
        # First encoder layer
        x1 = self.encoder_conv1(x)  # [batch, 128, L]
        x1 = self.encoder_bn1(x1)
        x1 = self.dropout(x1)  # Apply dropout
        x1_pooled = self.pooling1(x1)  # [batch, 128, L//2]
    
        # Resize x1 to match x1_pooled's sequence length
        x1_resized = F.interpolate(x1, size=x1_pooled.size(2), mode='linear', align_corners=False)
    
        # Second encoder layer with skip connection
        x2 = self.encoder_conv2(x1_pooled + x1_resized)  # Skip connection: Add resized x1 to x1_pooled
        x2 = self.encoder_bn2(x2)
        x2 = self.dropout(x2)  # Apply dropout
    
        # Final pooling
        x_final = self.final_pooling(x2)  # [batch, n_channels, n_seq]
        x_final = x_final.permute(0, 2, 1)  # [batch, n_seq, n_channels]
    
        mu = self.fc_mu(x_final)  # [batch, n_seq, latent_dim]
        logvar = self.fc_logvar(x_final)  # [batch, n_seq, latent_dim]
    
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def learnable_filter(self, x, kernel_sizes, kernel_centers, num_of_kernels=4):
        batch, nchannels, nlength = x.shape
     #   print("kernel_sizes, ", kernel_sizes)
     #   print("kernel_centers = ", kernel_centers)
        # Create a Gaussian distribution over a large enough range
        ones = torch.arange(-nlength * 1.5, nlength * 1.5, device=x.device)  # Large enough range
        
        # Initialize result tensor
        filtered_output = torch.zeros_like(x)
        
        # Apply Gaussian filters for each kernel center across batches
        for b in range(batch):
            for kernel_center, kernel_size in zip(kernel_centers[b, :], kernel_sizes[b, :]):
              #  print("kernel_center, kernel_size", kernel_center, kernel_size)
                center_value = -8000 + 16000 * kernel_center
                gaussian = torch.exp(-0.5 * ((ones - center_value) / (kernel_size * 4000))**2)  # Center Gaussian at kernel_center
                
                # Slice Gaussian to match input length
                start = len(ones) // 2 - nlength // 2
                end = start + nlength
                gaussian = gaussian[start:end]
                
                # Reshape Gaussian to match input dimensions
                gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, nlength]
                
                # Accumulate filtered results for the current batch
                filtered_output[b, 0, :] += x[b, 0, :] * gaussian.squeeze(0).squeeze(0)
                filtered_output[b, 1, :] += x[b, 1, :] * gaussian.squeeze(0).squeeze(0)

        # Normalize filtered_output for each batch
        batch_sums = torch.amax(filtered_output, dim=(1, 2), keepdim=True)  # Compute max for each batch
        filtered_output = filtered_output / batch_sums  # Normalize by batch max
        
        return filtered_output
    

    def decoder(self, z):
        
        if self.use_gaussians == True:
            zr = z.reshape(z.size(0), -1)
            x_g = self.latent_to_gaussians(zr)
            x_g = torch.abs(x_g) #self.latent_to_gaussians_activation(x_g) 
            #x_g[:,4:] = F.hardsigmoid((x_g[:,4:]-1)*3 )
#            x_g = torch.cat([x_g[:, :4], F.hardsigmoid((x_g[:, 4:] - 1) * 3)], dim=1)
            x_g = torch.cat([2 * F.sigmoid( x_g[:, :4] - 10 ), x_g[:, 4:] /(torch.max(x_g[:, 4:])+0.0001) ], dim=1)        
        
        # Latent to decoder
        x = self.latent_to_decoder(z)  # [batch, n_seq, n_channels]
        x = x.permute(0, 2, 1)  # [batch, n_channels, n_seq]
        
        # Decoder layers with skip connections
        x1 = self.decoder_conv1(x)
        x1 = self.decoder_bn1(x1)
        x1 = self.dropout(x1)  # Apply dropout
        
        # Resize x to match x1's sequence length
        if x.size(2) != x1.size(2):
            x = F.interpolate(x, size=x1.size(2), mode='linear', align_corners=False)
        
        x2 = self.unpooling1(x1 + x)  # Skip connection: Add resized x to x1
        x2 = self.decoder_bn2(x2)
        x2 = self.dropout(x2)  # Apply dropout
        
        # Resize x1 to match x2's sequence length before addition
        if x1.size(2) != x2.size(2):
            x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        
        x3 = self.decoder_conv2(x2 + x1)  # Skip connection: Add resized x1 to x2
        x3 = self.decoder_bn3(x3)
        x3 = self.dropout(x3)  # Apply dropout
        
        # Resize x2 to match x3's sequence length
        if x2.size(2) != x3.size(2):
            x2 = F.interpolate(x2, size=x3.size(2), mode='linear', align_corners=False)
        
        x4 = self.decoder_conv3(x3 + x2)  # Skip connection: Add resized x2 to x3
        x4 = x4 / (torch.max(x4) + 0.001)
        x4 = F.hardsigmoid(x4)
        
        reconstructed = self.fp(x4)
        if self.use_gaussians:
            
            reconstructed = self.learnable_filter(reconstructed, x_g[:,:4], x_g[:,4:])
        
        return reconstructed
    


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
