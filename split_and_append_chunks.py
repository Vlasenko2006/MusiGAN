import numpy as np

def split_and_append_chunks(data):
    """
    Splits input-target pairs from the dataset into individual music chunks
    and appends them to a list.

    Parameters:
    - data: List of input-target pairs (loaded from .npy files).

    Returns:
    - List of individual music chunks.
    """
    music_chunks = []

    for input_chunk, target_chunk in data:
        # Append both input and target chunks to the list
        music_chunks.append(input_chunk)
        music_chunks.append(target_chunk)

    return music_chunks

