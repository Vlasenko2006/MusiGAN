import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class

FREEZE_ENCODER_DECODER_AFTER = 10  # Number of steps after which encoder-decoder weights are frozen

sf = 4
do = 0.3


class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3, sound_channels=2, batch_size=64, seq_len=120000):
        super(VariationalAttentionModel, self).__init__()

        # Encoder and Decoder from encoder-decoder
        self.encoder_decoder = encoder_decoder(input_dim=sound_channels, n_channels=n_channels, n_seq=n_seq)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=128,
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
        print(f"Input shape: {x.shape}")

        # Verify input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")

        batch_size, input_dim, seq_len = x.shape

        # Encoding
        encoded = self.encoder_decoder.encoder(x)  # Use encoder from encoder-decoder
        print(f"Encoded shape: {encoded.shape}")

        # Permute for Transformer
        x = encoded.permute(2, 0, 1)  # Shape: [output_seq_len, batch_size, n_channels]
        print(f"Permuted for Transformer shape: {x.shape}")

        # Pass through Transformer Encoder
        transformer_out = self.transformer(x)  # Shape: [output_seq_len, batch_size, n_channels]
        print(f"Transformer output shape: {transformer_out.shape}")

        # Permute back for Decoder
        transformer_out = transformer_out.permute(1, 2, 0)  # Shape: [batch_size, n_channels, output_seq_len]
        print(f"Permuted back for Decoder shape: {transformer_out.shape}")

        # Decoding
        reconstructed = self.encoder_decoder.decoder(encoded)  # Use decoder from encoder-decoder
        print(f"Reconstructed shape: {reconstructed.shape}")


        task_specific = self.encoder_decoder.decoder(transformer_out)  # Decode to match input shape
        print(f"Task-specific decoded shape: {task_specific.shape}")

        return reconstructed, task_specific


# Test function
def test_variational_attention_model():
    """
    Test the VariationalAttentionModel to ensure that the decoded output matches the original input shape.
    """
    # Define model parameters
    batch_size = 64
    input_dim = 2
    seq_len = 120000
    n_channels = 64
    n_seq = 3

    # Create a random input tensor
    x = torch.rand(batch_size, input_dim, seq_len)

    # Initialize the model
    print("Initialize model")
    model = VariationalAttentionModel(input_dim=input_dim, n_channels=n_channels, n_seq=n_seq)
    print("computing molde")
    # Run through the forward pass
    reconstructed, task_specific = model(x)

    # Assert that the input and reconstructed shapes are the same
    assert x.shape == reconstructed.shape, f"Shape mismatch: input shape {x.shape}, reconstructed shape {reconstructed.shape}"

    # Assert that the task-specific output shape matches the input shape
    assert x.shape == task_specific.shape, f"Shape mismatch: input shape {x.shape}, task-specific shape {task_specific.shape}"

    # Print success message
    print("Test passed! Input, reconstructed, and task-specific output shapes match.")


# Run the test
if __name__ == "__main__":
    test_variational_attention_model()
