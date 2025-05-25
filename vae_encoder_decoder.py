import torch
import torch.nn as nn

class VariationalEncoderDecoder(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=1, latent_dim=128):
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

        # Fully connected layers for latent space (per position)
        self.fc_mu = nn.Linear(n_channels, latent_dim)
        self.fc_logvar = nn.Linear(n_channels, latent_dim)

        # Projection layer to map latent_dim to decoder input size, PER POSITION
        self.latent_to_decoder = nn.Linear(latent_dim, n_channels)
        
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
        Encoder: Takes input and encodes it into a sequence of latent vectors suitable for transformer input.
        
        Outputs mean (mu) and log-variance (logvar) for the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu: Mean of the latent space. [batch_size, n_seq * 1000, latent_dim]
                - logvar: Log-variance of the latent space. [batch_size, n_seq * 1000, latent_dim]
        """
        batch_size = x.size(0)

        x = self.encoder_conv1(x)  # [batch, 128, L]
        x = self.encoder_bn1(x)
        x = self.pooling1(x)       # [batch, 128, L//2]

        x = self.encoder_conv2(x)  # [batch, n_channels, L']
        x = self.encoder_bn2(x)
        x = self.final_pooling(x)  # [batch, n_channels, n_seq*1000]

        # Now we want to treat the sequence dimension as sequence, and apply FC per position
        x = x.permute(0, 2, 1)  # [batch, n_seq*1000, n_channels]

        mu = self.fc_mu(x)      # [batch, n_seq*1000, latent_dim]
        logvar = self.fc_logvar(x)  # [batch, n_seq*1000, latent_dim]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: Allows backpropagation through the stochastic sampling process.

        Args:
            mu (torch.Tensor): Mean of the latent space. [batch, seq, latent_dim]
            logvar (torch.Tensor): Log-variance of the latent space. [batch, seq, latent_dim]

        Returns:
            torch.Tensor: Sampled latent vector. [batch, seq, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        # z: [batch, n_seq*1000, latent_dim]
        batch_size = z.size(0)
        x = self.latent_to_decoder(z)  # [batch, n_seq*1000, n_channels]
        x = x.permute(0, 2, 1)  # [batch, n_channels, n_seq*1000]

        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)
        x = self.unpooling1(x)
        x = self.decoder_bn2(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn3(x)
        reconstructed = self.decoder_conv3(x)
        return reconstructed

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
