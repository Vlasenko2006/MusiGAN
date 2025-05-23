#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:23:09 2025

@author: andreyvlasenko
"""


import numpy as np
from numpy_2_mp3 import numpy_to_mp3
import os


# Paths
sample_rate = 12000
path = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_output/"
path_output = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_music_output"

# Ensure the output directory exists
os.makedirs(path_output, exist_ok=True)

# Process all .npy files
for file in os.listdir(path):
    if file.endswith(".npy"):
        file_path = os.path.join(path, file)
        print(f"Processing file: {file_path}")
        
        # Load the NumPy array
        array = np.load(file_path)
        
        # Subtract 1 and convert to float32
        array = array - 1.0
        array = array.astype(np.float32)
        
        # Generate output MP3 file path
        output_mp3_file = os.path.join(path_output, file.replace(".npy", ".mp3"))
        
        # Convert to MP3
        numpy_to_mp3(array, sample_rate, output_mp3_file)

print("Processing complete. MP3 files saved to:", path_output)