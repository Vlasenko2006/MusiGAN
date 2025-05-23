#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:37:20 2025

@author: andreyvlasenko
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import AudioDataset
from utilities import load_checkpoint
from tqdm import tqdm

# Constants
dataset_folder = "../dataset"
sample_rate = 16000
checkpoint_path = "checkpoints_trans2/model_epoch_610.pt"  # Path to the checkpoint
validation_output_folder = "validation_output"
train_output_folder = "train_output"

batch_size = 16

# Ensure output folders exist
os.makedirs(validation_output_folder, exist_ok=True)
os.makedirs(train_output_folder, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

