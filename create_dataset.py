#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:12:20 2025

@author: andreyvlasenko
"""

import os
import numpy as np
import random

def create_dataset(output_folder="output", dataset_folder="dataset", chunk_duration=10, target_sr=16000):
    """
    Creates a dataset from .npy files. Splits each file into chunks of equal duration,
    combines odd and even chunks into input-target pairs, scrambles them, and splits
    into training and validation sets.

    Parameters:
    - output_folder: Folder containing source .npy files.
    - dataset_folder: Folder to save the dataset.
    - chunk_duration: Duration of each chunk in seconds.
    - target_sr: Sample rate in Hz (default 16 kHz).
    """
    # Ensure the dataset folder exists
    os.makedirs(dataset_folder, exist_ok=True)

    # List all .npy files in the output folder
    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])[:]  # First 20 files

    # Initialize containers for training and validation data
    training_data = []
    validation_data = []

    # Process each .npy file
    chunk_count = 0  # Counter for chunks
    for file_idx, npy_file in enumerate(npy_files, start=1):
        # Load the .npy file
        file_path = os.path.join(output_folder, npy_file)
        audio_array = np.load(file_path)

        # Calculate the number of samples in a chunk
        samples_per_chunk = chunk_duration * target_sr

        # Ensure the audio is long enough and can be split into 12 chunks
        total_samples = audio_array.shape[-1]
        expected_chunks = 12
        if total_samples < expected_chunks * samples_per_chunk:
            print(f"Skipping {npy_file}: Not enough samples to create 12 \
                  chunks of {chunk_duration} seconds.\
                  Sample per chunks { expected_chunks * samples_per_chunk}")
            continue

        # Split the audio into 12 chunks
        chunks = [
            audio_array[..., i * samples_per_chunk:(i + 1) * samples_per_chunk]
            for i in range(expected_chunks)
        ]

        # Combine odd and even chunks into input-target pairs
        pairs = [
            (chunks[i], chunks[i + 1]) for i in range(0, len(chunks) - 1, 2)
        ]
        chunk_count += len(pairs)

        # Shuffle the pairs
        random.shuffle(pairs)

        # Add to the combined list
        training_data.extend(pairs)

    # Split into training and validation sets (90/10 split)
    random.shuffle(training_data)  # Shuffle the entire dataset
    split_idx = int(0.9 * len(training_data))  # 90% for training
    validation_data = training_data[split_idx:]
    training_data = training_data[:split_idx]

    # Save the training and validation sets
    training_file = os.path.join(dataset_folder, "training_set.npy")
    validation_file = os.path.join(dataset_folder, "validation_set.npy")
    np.save(training_file, training_data)
    np.save(validation_file, validation_data)

    print(f"Dataset created successfully!")
    print(f"Training set size: {len(training_data)} pairs")
    print(f"Validation set size: {len(validation_data)} pairs")
    print(f"Saved to: {dataset_folder}")

# Example usage
if __name__ == "__main__":
    create_dataset(output_folder="output", dataset_folder="dataset", chunk_duration=10, target_sr=12000)