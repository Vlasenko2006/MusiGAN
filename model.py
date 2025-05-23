import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class

FREEZE_ENCODER_DECODER_AFTER = 10  # Number of steps after which encoder-decoder weights are frozen

sf = 4
do = 0.

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)


class AttentionModel(nn.Module):
    def __init__(self, num_heads=8, num_layers=1, n_channels=96, n_seq=3, sound_channels=2, seq_len=120000):
        super(AttentionModel, self).__init__()

        # Encoder and Decoder from encoder-decoder
        self.encoder_decoder = encoder_decoder(input_dim=sound_channels, n_channels=n_channels, n_seq=n_seq)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=2 *2 *128,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Reconstructed tensor of shape [batch_size, input_dim, seq_len].
                - Output tensor of shape [batch_size, input_dim, seq_len].
        """
        # Debug: input shape

        # Verify input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")

        batch_size, input_dim, seq_len = x.shape

        # Encoding
        encoded = self.encoder_decoder.encoder(x)  # Use encoder from encoder-decoder

        # Permute for Transformer
        x = encoded.permute(2, 0, 1)  # Shape: [output_seq_len, batch_size, n_channels]

        # Pass through Transformer Encoder
        transformer_out = self.transformer(x)  # Shape: [output_seq_len, batch_size, n_channels]

        # Permute back for Decoder
        transformer_out = transformer_out.permute(1, 2, 0)  # Shape: [batch_size, n_channels, output_seq_len]

        output = self.encoder_decoder.decoder(transformer_out)  # Decode to match input shape

        M = torch.max(torch.abs(output))

        #if M> 1: output = output/M

        return output

