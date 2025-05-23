import torch
from vae_model import VariationalAttentionModel  # Import the VariationalAttentionModel

def test_variational_attention_model():
    # Hyperparameters for the test
    sound_channels = 2  # Example: Stereo audio with 2 channels
    seq_len = 120000  # Example: Sequence length (e.g., 5 seconds of audio at 24 kHz)
    n_channels = 96  # Number of channels in the encoder's output
    n_seq = 3  # Sequence reduction factor
    latent_dim = 128  # Latent dimension size
    num_heads = 8  # Number of attention heads in the transformer
    num_layers = 1  # Number of transformer layers

    # Create a random input tensor with shape [batch_size, sound_channels, seq_len]
    batch_size = 16  # Test with a batch size of 16
    input_tensor = torch.randn(batch_size, sound_channels, seq_len)

    # Initialize the VariationalAttentionModel
    model = VariationalAttentionModel(
        num_heads=num_heads,
        num_layers=num_layers,
        n_channels=n_channels,
        n_seq=n_seq,
        sound_channels=sound_channels,
        seq_len=seq_len,
        latent_dim=latent_dim
    )

    # Inspect the Transformer's configuration
    transformer_layer = model.transformer.layers[0]  # Get the first Transformer layer
    d_model = transformer_layer.self_attn.embed_dim  # Embedding dimension (d_model)
    num_heads = transformer_layer.self_attn.num_heads  # Number of attention heads
    print(f"Transformer's input layer configuration:")
    print(f" - Expected sequence length (seq_len): Any length (depends on input)")
    print(f" - Batch size (batch_size): Any size (depends on input)")
    print(f" - Embedding dimension (d_model): {d_model}")
    print(f" - Number of attention heads: {num_heads}")

    # Forward pass through the model
    print("\nRunning the test...")
    reconstructed, mu, logvar = model(input_tensor)

    # Print shapes of the outputs
    print(f"Input shape: {input_tensor.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Assertions to validate correctness
    assert reconstructed.shape == input_tensor.shape, \
        f"Reconstructed shape {reconstructed.shape} does not match input shape {input_tensor.shape}"
    assert mu.shape == (batch_size, latent_dim), \
        f"Mu shape {mu.shape} does not match expected shape {(batch_size, latent_dim)}"
    assert logvar.shape == (batch_size, latent_dim), \
        f"Logvar shape {logvar.shape} does not match expected shape {(batch_size, latent_dim)}"

    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_variational_attention_model()
