import torch
import torch.nn as nn

class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3):
        super(VariationalAttentionModel, self).__init__()

        # Encoder: Conv1D and Pooling layers (unchanged)
        self.encoder_conv1 = nn.Conv1d(
            in_channels=input_dim,  # Matches the input dimension (e.g., 2 channels)
            out_channels=128,
            kernel_size=9,
            stride=2,
            padding=4
        )
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder_conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=n_channels,  # Final encoded channels
            kernel_size=7,
            stride=2,
            padding=3
        )

        # Final pooling to reduce sequence length
        self.final_pooling = nn.AdaptiveAvgPool1d(n_seq * 1000)  # Encoded length

        # Decoder: Designed as the inverse of the encoder
        self.decoder_conv1 = nn.ConvTranspose1d(
            in_channels=n_channels,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=3,
            output_padding=1
        )
        self.unpooling1 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.decoder_conv2 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=5,  # Keep stride for larger upsampling
            padding=1,
            output_padding=4  # Adjust output padding to achieve the correct size
        )
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

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            torch.Tensor: Encoded tensor.
        """
        print(f"[DEBUG] Encoder input: {x.shape}")
        x = self.encoder_conv1(x)  # Shape: [batch_size, 128, reduced_seq_len]
        print(f"[DEBUG] After encoder_conv1: {x.shape}")
        x = self.pooling1(x)  # Shape: [batch_size, 128, reduced_seq_len // 2]
        print(f"[DEBUG] After pooling1: {x.shape}")
        x = self.encoder_conv2(x)  # Shape: [batch_size, n_channels, smaller_seq_len]
        print(f"[DEBUG] After encoder_conv2: {x.shape}")
        encoded = self.final_pooling(x)  # Shape: [batch_size, n_channels, n_seq * 1000]
        print(f"[DEBUG] After final_pooling: {encoded.shape}")
        return encoded

    def decoder(self, encoded):
        """
        Decoder: Reconstructs the input from the encoded representation.

        Args:
            encoded (torch.Tensor): Encoded tensor.

        Returns:
            torch.Tensor: Reconstructed tensor of the same shape as the input.
        """
        print(f"[DEBUG] Decoder input: {encoded.shape}")
        x = self.decoder_conv1(encoded)  # Shape: [batch_size, 128, upsampled_seq_len]
        print(f"[DEBUG] After decoder_conv1: {x.shape}")
        x = self.unpooling1(x)  # Shape: [batch_size, 128, further_upsampled_seq_len]
        print(f"[DEBUG] After unpooling1: {x.shape}")
        x = self.decoder_conv2(x)  # Shape: [batch_size, 128, even_further_upsampled_seq_len]
        print(f"[DEBUG] After decoder_conv2: {x.shape}")
        reconstructed = self.decoder_conv3(x)  # Shape: [batch_size, input_dim, original_seq_len]
        print(f"[DEBUG] After decoder_conv3: {reconstructed.shape}")
        return reconstructed


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
    model = VariationalAttentionModel(input_dim=input_dim, n_channels=n_channels, n_seq=n_seq)

    # Run through the encoder and decoder
    encoded = model.encoder(x)
    decoded = model.decoder(encoded)

    # Assert that the input and decoded shapes are the same
    assert x.shape == decoded.shape, f"Shape mismatch: input shape {x.shape}, decoded shape {decoded.shape}"

    # Print success message
    print("Test passed! Input and decoded shapes match.")

# Run the test
if __name__ == "__main__":
    test_variational_attention_model()
