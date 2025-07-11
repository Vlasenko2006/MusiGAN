import torch

def load_decoder_weights(model, checkpoint_path):
    """
    Load decoder weights from a checkpoint into an encoder_decoder model.

    Args:
        model (encoder_decoder): An instance of the encoder_decoder class.
        checkpoint_path (str): Path to the checkpoint file containing weights.

    Returns:
        encoder_decoder: The model with updated decoder weights.
    """
    # Step 1: Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load checkpoint from file
    
    # Step 2: Extract decoder weights from the checkpoint
    # Assuming the checkpoint contains a key "decoder" or similar structure
    decoder_weights = {k: v for k, v in checkpoint.items() if "decoder" in k}

    # Step 3: Get the model's current decoder state_dict
    model_decoder_state = model.state_dict()

    # Step 4: Filter and update decoder weights (ignoring missing or mismatched keys)
    decoder_state_to_load = {}
    for key, weight in decoder_weights.items():
        if key in model_decoder_state and model_decoder_state[key].size() == weight.size():
            decoder_state_to_load[key] = weight
        else:
            print(f"Skipping {key}: shape mismatch or not found in model.")

    # Load the filtered weights into the model's decoder
    model_decoder_state.update(decoder_state_to_load)
    model.load_state_dict(model_decoder_state, strict=False)  # Use strict=False to allow partial loading

    print("Decoder weights loaded successfully.")
    return model

# Example usage:
# Initialize your model
#model = encoder_decoder(input_dim=2, n_channels=64, n_seq=3)

# Path to the checkpoint file
#checkpoint_path = "path/to/your/checkpoint.pth"

# Load decoder weights
#model = load_decoder_weights(model, checkpoint_path)
