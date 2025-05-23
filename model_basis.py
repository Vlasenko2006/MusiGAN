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

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

sf = 4
do = 0.3

class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3):
        super(VariationalAttentionModel, self).__init__()

        # Calculate the required output sequence length
        self.output_seq_len = n_seq * 1000

        # Pooling Layer to downsample the input sequence first
        self.initial_pooling = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)

        # Encoder: Conv1D and Pooling layers
        self.encoder_conv1 = nn.Conv1d(
            in_channels=2,  # Adjusted to match the number of channels from the pooling output
            out_channels=128,
            kernel_size=9,
            stride=2,
            padding=4
        )
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder_conv2 = nn.Conv1d(
            in_channels=128,  # Matches the output of encoder_conv1
            out_channels=n_channels,  # Adjust output channels
            kernel_size=7,
            stride=2,
            padding=3
        )

        # Final pooling to ensure output sequence length matches `n_seq * 1000`
        self.final_pooling = nn.AdaptiveAvgPool1d(self.output_seq_len)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: Transposed Conv1D for upsampling
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=n_channels,
            out_channels=2,  # Reconstruct input_dim directly
            kernel_size=9,
            stride=2,
            padding=4,
            output_padding=1
        )

        # Feed-forward layers for task-specific output
        self.fc = nn.Sequential(
            nn.Linear(n_channels, 128),
            nn.ReLU(),
            nn.Dropout(p=do),
            nn.Linear(128, n_channels)
        )

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
        # Debugging: Print input shape
        print(f"Input shape before processing: {x.shape}")

        # Verify input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")
        
        batch_size, input_dim, seq_len = x.shape

        # Ensure input has the correct number of channels
        if input_dim != 2:  # Adjust this if more channels are expected
            raise ValueError(f"Expected input to have 2 channels (dim=1), but got {input_dim} channels.")

        # Split the input tensor into two parts along the channel dimension
        x1, x2 = torch.split(x, 1, dim=1)  # Each part will have shape [batch_size, 1, seq_len]
        print(f"Split input into two parts: x1 {x1.shape}, x2 {x2.shape}")

        # Apply pooling to each part separately
        x1 = self.initial_pooling(x1.squeeze(1))  # Shape: [batch_size, reduced_seq_len]
        x2 = self.initial_pooling(x2.squeeze(1))  # Shape: [batch_size, reduced_seq_len]

        # Concatenate the two parts back together along the channel dimension
        x = torch.stack((x1, x2), dim=1)  # Shape: [batch_size, 2, reduced_seq_len]
        print(f"After initial_pooling and stacking: {x.shape}")

        # Encode input
        x = self.encoder_conv1(x)  # Shape: [batch_size, 128, reduced_seq_len]
        print(f"After encoder_conv1: {x.shape}")
        x = self.pooling1(x)  # Downsample further
        print(f"After pooling1: {x.shape}")

        x = self.encoder_conv2(x)  # Shape: [batch_size, n_channels, smaller_seq_len]
        print(f"After encoder_conv2: {x.shape}")

        # Ensure sequence length matches `n_seq * 1000`
        x = self.final_pooling(x)
        print(f"After final_pooling: {x.shape}")

        # Save encoded representation for reconstruction
        encoded = x

        # Permute for Transformer
        x = x.permute(2, 0, 1)  # Shape: [output_seq_len, batch_size, n_channels]
        print(f"After permute for Transformer: {x.shape}")

        # Pass through Transformer Encoder
        transformer_out = self.transformer(x)  # Shape: [output_seq_len, batch_size, n_channels]
        print(f"After Transformer Encoder: {transformer_out.shape}")

        # Permute back for Transposed Conv1D
        transformer_out = transformer_out.permute(1, 2, 0)  # Shape: [batch_size, n_channels, output_seq_len]
        print(f"After permute back for Transposed Conv1D: {transformer_out.shape}")

        # Decode using Transposed Conv1D for reconstructed output
        reconstructed = self.decoder_conv(encoded)  # Reconstruct directly from encoded features
        print(f"After decoder_conv (reconstructed): {reconstructed.shape}")

        # Process transformer output for task-specific output
        task_specific = self.fc(transformer_out.mean(dim=2))  # Shape: [batch_size, n_channels]
        print(f"After task-specific feed-forward: {task_specific.shape}")
        task_specific = task_specific.unsqueeze(2).expand(-1, -1, seq_len)  # Expand back to sequence length
        task_specific = self.decoder_conv(task_specific)  # Decode to match input shape
        task_specific = task_specific.permute(0, 2, 1)  # Shape: [batch_size, seq_len, input_dim]
        print(f"After decoder_conv (task-specific output): {task_specific.shape}")

        return reconstructed, task_specific
