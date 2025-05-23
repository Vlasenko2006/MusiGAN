#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:13:59 2025

@author: andreyvlasenko
"""

import os
import numpy as np
from numpy_2_mp3 import mp3_to_numpy


def process_music_files(input_folder, output_folder, max_files=200, clip_duration=120, target_sr=16000):
    """
    Searches for music files in a folder (including subfolders), processes them into NumPy arrays,
    and saves the first `clip_duration` seconds of audio for each file into the output folder.
    Skips files shorter than `clip_duration` seconds.

    Parameters:
    - input_folder: Folder to search for music files.
    - output_folder: Folder to save processed NumPy arrays.
    - max_files: Maximum number of files to process.
    - clip_duration: Duration (in seconds) of audio to keep.
    - target_sr: Target sample rate for the waveform.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect all music files (e.g., MP3 format) in the folder and subfolders
    music_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".mp3"):
                music_files.append(os.path.join(root, file))
                if len(music_files) >= max_files:
                    break
        if len(music_files) >= max_files:
            break

    print(f"Found {len(music_files)} music files to process.")

    # Process each file
    for idx, music_file in enumerate(music_files, start=1):
        try:
            # Convert MP3 to NumPy array
            waveform, sample_rate = mp3_to_numpy(music_file, target_sr=target_sr)
            
            # Calculate the number of samples for the desired clip duration
            num_samples = clip_duration * target_sr

            # Check if the audio is long enough
            if waveform.shape[-1] < num_samples or waveform.ndim < 2:
                print(f"Skipping {music_file}: Less than {clip_duration} seconds.")
                continue

            # Take only the first `clip_duration` seconds
            truncated_waveform = waveform[:, :num_samples] if waveform.ndim == 2 else waveform[:num_samples]
            truncated_waveform = truncated_waveform.astype(np.float16)
            # Save the processed waveform as a NumPy file
            output_file = os.path.join(output_folder, f"{idx}.npy")
            np.save(output_file, truncated_waveform)
            print(f"Processed and saved: {output_file}")

        except Exception as e:
            print(f"Error processing {music_file}: {e}")

# Example Usage
if __name__ == "__main__":
    path = "path to your music"  # Input folder containing music files
    output_folder = "output"  # Output folder to save processed files
    
    
# Check if the folder exists
    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    else:
        print(f"Folder '{output_folder}' already exists.")

    # Process music files
    process_music_files(input_folder=path, output_folder=output_folder, max_files=400, clip_duration=120, target_sr=12000)