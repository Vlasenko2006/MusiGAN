import os
import torch
import torch.nn as nn

def update_checkpoint(checkpoint_path, model, optimizer, strict=False):
    """
    Load the model and optimizer state from a checkpoint file. Handles missing keys in the model's state_dict.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        strict (bool): Whether to strictly enforce that the keys in the checkpoint match the model's keys.

    Returns:
        int: The epoch number to resume training from.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load the model's state_dict with strict handling of missing keys
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # Update the optimizer only if possible (to avoid issues with new parameters)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError:
            print("Optimizer state_dict could not be loaded. Reinitializing optimizer.")
            pass

        # Initialize missing parameters if strict=False
        if not strict:
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint['model_state_dict']
            missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
            for key in missing_keys:
                if "minibatch_discrimination.T" in key:
                    nn.init.normal_(model_state_dict[key], mean=0, std=0.02)
                elif "fc.weight" in key:
                    nn.init.normal_(model_state_dict[key], mean=0, std=0.02)
                elif "fc.bias" in key:
                    nn.init.constant_(model_state_dict[key], 0)
                else:
                    # Generic initialization for other missing keys
                    nn.init.normal_(model_state_dict[key], mean=0, std=0.02)
            model.load_state_dict(model_state_dict)

        start_epoch = checkpoint.get('epoch', 0) + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from epoch 1.")
        return 1  # Start from the first epoch if no checkpoint is found
