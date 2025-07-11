#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:05:35 2025

@author: andreyvlasenko
"""

import numpy as np
import torch
import os
from numpy_2_mp3 import numpy_to_mp3
import matplotlib.pyplot as plt
from numpy_2_piano_keys import from_loudness_to_signal


def postprocessing(signal, ratio = 0.95):
    r = 1
    l = len(signal.flatten())
    i = 0
    while r > ratio:
        i += 1
        signal[signal> 0.01 * i] = i
        signal[signal < i] = 0 
        nz = np.count_nonzero(signal.flatten())
        r = nz / l
    print(r)
    return r
        




path = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/piano/"
output_path = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/GAN_music/"


ndev = 5
keys = list(range(27, 63))
fs = 12000         # sample rate frequencey
seq_len = 16500
min_note_duration = 0.02
loudness_threshold_ratio = 0.05
r = 0.4 

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)


# Process all .npy files in the input directory
for file in os.listdir(path):
    if file.endswith(".npy"):
        file_path = os.path.join(path, file)
        print(f"Processing file: {file_path}")

        # Load the NumPy array
        signal = np.load(file_path)
        #
        signal = np.asarray(signal).squeeze()
        #r = postprocessing(signal, ratio  = r)
        signal = signal - r  #0.5
        signal[signal<0] = 0
        signal = signal / np.max(signal)
        plt.plot(signal[0,:])
        plt.title(str(file))
        plt.show()
        signal = torch.tensor(signal).reshape([1,2,len(keys),500])
        array = from_loudness_to_signal(
            signal, fs=fs, keys=keys, ndev=ndev,
            min_note_duration=min_note_duration, seq_len=seq_len
        )
        
        # Generate output MP3 file path (preserve base name)
        base_name = os.path.splitext(file)[0]
        output_mp3_file = os.path.join(output_path, f"{base_name}.mp3")

        # Convert to MP3
        array = array.squeeze()
        numpy_to_mp3(array, fs, output_mp3_file)




print("Processing complete. MP3 files saved to:", output_path)