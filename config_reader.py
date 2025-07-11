import yaml

def load_config_from_yaml(yaml_path):
    """
    Load configuration parameters from a YAML file.
    Args:
        yaml_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration parameters.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Example usage for integration:
# config = load_config_from_yaml("config.yaml")
# Now you can access parameters like config['batch_size'], config['epochs'], etc.

# Update your main script to use these parameters.
# For example, in your training or initialization functions, replace hardcoded values:
# Instead of: batch_size = 64
# Use: batch_size = config['batch_size']

# Example code update in your main training script:
# -------------------------------------------------
# # Instead of hardcoding parameters:
# batch_size = 64
# epochs = 30000
# checkpoint_folder = "checkpoint_piano_hards4"
#
# # Use loaded config:
# config = load_config_from_yaml("config.yaml")
# batch_size = config['batch_size']
# epochs = config['epochs']
# checkpoint_folder = config['checkpoint_folder']
# learning_rate = config['learning_rate']
# etc.
#
# # Pass 'config' or its individual parameters to the rest of your code:
# train_vae_gan(..., batch_size=config['batch_size'], ...)