import torch
import torch.nn as nn

class VariationalEncoderDecoder(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3, latent_dim=128):
        super(VariationalEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_seq = n_seq

        # Encoder: Conv1D, BatchNorm, and Pooling layers
        self.encoder_conv1 = nn.Conv1d(
            in_channels=input_dim,  # Matches the input dimension (e.g., 2 channels)
            out_channels=128,
            kernel_size=9,
            stride=2,
            padding=4
        )
        self.encoder_bn1 = nn.BatchNorm1d(128)  # Batch Normalization for first convolution
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder_conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=n_channels,  # Final encoded channels
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.encoder_bn2 = nn.BatchNorm1d(n_channels)  # Batch Normalization for second convolution

        # Final pooling to reduce sequence length
        self.final_pooling = nn.AdaptiveAvgPool1d(n_seq * 1000)  # Encoded length

        # Fully connected layers for latent space
        self.fc_mu = nn.Linear(n_channels * n_seq * 1000, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(n_channels * n_seq * 1000, latent_dim)  # Log-variance of latent space

        # Projection layer to map latent_dim to decoder input size
        self.latent_to_decoder = nn.Linear(latent_dim, n_channels * n_seq * 1000)

        # Decoder: Designed as the inverse of the encoder
        self.decoder_conv1 = nn.ConvTranspose1d(
            in_channels=n_channels,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=3,
            output_padding=1
        )
        self.decoder_bn1 = nn.BatchNorm1d(128)  # Batch Normalization for first transposed convolution

        self.unpooling1 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.decoder_bn2 = nn.BatchNorm1d(128)  # Batch Normalization after unpooling

        self.decoder_conv2 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=5,  # Keep stride for larger upsampling
            padding=1,
            output_padding=4  # Adjust output padding to achieve the correct size
        )
        self.decoder_bn3 = nn.BatchNorm1d(128)  # Batch Normalization for second transposed convolution

        self.decoder_conv3 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=input_dim,
            kernel_size=9,
            stride=2,
            padding=4,
            output_padding=1
        )

    def encoder(self, x):
        """
        Encoder: Takes input and encodes it into a smaller representation.
        
        Outputs mean (mu) and log-variance (logvar) for the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu: Mean of the latent space.
                - logvar: Log-variance of the latent space.
        """
        batch_size = x.size(0)

        x = self.encoder_conv1(x)  # Shape: [batch_size, 128, reduced_seq_len]
        x = self.encoder_bn1(x)  # Apply batch normalization
        x = self.pooling1(x)  # Shape: [batch_size, 128, reduced_seq_len // 2]

        x = self.encoder_conv2(x)  # Shape: [batch_size, n_channels, smaller_seq_len]
        x = self.encoder_bn2(x)  # Apply batch normalization
        x = self.final_pooling(x)  # Shape: [batch_size, n_channels, n_seq * 1000]

        # Flatten for fully connected layers
        x = x.view(batch_size, -1)  # Shape: [batch_size, n_channels * n_seq * 1000]

        # Latent space mean and log-variance
        mu = self.fc_mu(x)  # Shape: [batch_size, latent_dim]
        logvar = self.fc_logvar(x)  # Shape: [batch_size, latent_dim]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: Allows backpropagation through the stochastic sampling process.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log-variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Sampled latent vector

    def decoder(self, z):
        """
        Decoder: Reconstructs the input from the latent representation.

        Args:
            z (torch.Tensor): Sampled latent vector.

        Returns:
            torch.Tensor: Reconstructed tensor of the same shape as the input.
        """
        batch_size = z.size(0)

        # Project latent space to decoder input size
        x = self.latent_to_decoder(z)  # Shape: [batch_size, n_channels * n_seq * 1000]
        x = x.view(batch_size, self.n_channels, self.n_seq * 1000)  # Reshape to match ConvTranspose1D input

        x = self.decoder_conv1(x)  # Shape: [batch_size, 128, upsampled_seq_len]
        x = self.decoder_bn1(x)  # Apply batch normalization
        x = self.unpooling1(x)  # Shape: [batch_size, 128, further_upsampled_seq_len]
        x = self.decoder_bn2(x)  # Apply batch normalization
        x = self.decoder_conv2(x)  # Shape: [batch_size, 128, even_further_upsampled_seq_len]
        x = self.decoder_bn3(x)  # Apply batch normalization
        reconstructed = self.decoder_conv3(x)  # Shape: [batch_size, input_dim, original_seq_len]

        return reconstructed

    def forward(self, x):
        """
        Forward pass through the Variational Encoder-Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - reconstructed: Reconstructed tensor of the same shape as the input.
                - mu: Mean of the latent space.
                - logvar: Log-variance of the latent space.
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterization trick to sample from latent space
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
