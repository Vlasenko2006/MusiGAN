import torch
import torch.nn as nn
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class

class VariationalAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, n_channels=64, n_seq=3):
        super(VariationalAttentionModel, self).__init__()

        # Encoder and Decoder from encoder-decoder
        self.encoder_decoder = encoder_decoder(input_dim=input_dim, n_channels=n_channels, n_seq=n_seq)

    def encoder(self, x):
        """
        Encoder: Takes input and encodes it into a smaller representation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            torch.Tensor: Encoded tensor.
        """
        return self.encoder_decoder.encoder(x)

    def decoder(self, encoded):
        """
        Decoder: Reconstructs the input from the encoded representation.

        Args:
            encoded (torch.Tensor): Encoded tensor.

        Returns:
            torch.Tensor: Reconstructed tensor of the same shape as the input.
        """
        return self.encoder_decoder.decoder(encoded)


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
